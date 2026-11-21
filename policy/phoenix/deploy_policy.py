# ----------------------------------------------------------------------------
# Phoenix-MWA policy adapter: the get_model / eval / reset_model interface of the RoboSynChallenge evaluation
# harness (a real-robot runner offering the same environment interface can drive it as well).
#
# The model (submodule policy/phoenix/Phoenix-MWA, torch 2.7.1 + its own deps) runs in phoenix_worker.py under the
# Phoenix venv, driven over a stdio JSON pipe. Phoenix conditions on the
# current frame only, so there is no history to send back.
#
# phoenix_async real (default off): the next chunk is computed in the background while the arm keeps executing the
# current one, so it does not stop for every inference (AsyncChunks). phoenix_rtc_prefix True also hands the model
# the steps already committed (Training RTC, upstream's prometheus branch); only a checkpoint trained that way takes it
# (rtc_worker.py, picked for a run dir with model.yaml). With phoenix_async off (serial inference) that is
# SerialPrefix: each chunk continues the phoenix_prefix_steps targets the previous chunk planned after its executed ones.
# phoenix_rtc_infer inpaint | guidance (with phoenix_async real, any checkpoint, default off): the background chunk is
# drawn continuing the running chunk's targets from its observation on, the d that execute meanwhile at weight 1 and
# the rest by RTC's soft mask: by inpainting (phoenix_worker.InpaintScheduler) or by RTC's own guidance
# (phoenix_worker.RtcGuidance, upstream's implementation of Black et al. 2025).
# phoenix_smooth inpaint2soft (chunk_inpaint.py): each synchronous chunk is drawn continuing the previous chunk's steps
# that were not executed (inpainting at inference, any checkpoint).
#
# Unified eval interface:
#   get_model(usr_args) -> model
#   eval(env, model, obs) -> (obs, info, truncated, inference_times)
#   reset_model(model) -> None
# ----------------------------------------------------------------------------

from __future__ import annotations

import atexit
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import torch

from policy.inference_timing import finish_inference
from policy.inference_timing import start_inference
from policy.phoenix import chunk_inpaint
from policy.phoenix import chunk_smoother

POLICY_ROOT = Path(__file__).resolve().parent
REPO_ROOT = POLICY_ROOT.parents[1]
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
GRIPPER_DIMS = [6, 13]
ARM_DIMS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
ACTION_HORIZON = 32
ASYNC_MODES = ("off", "real")
# phoenix_rtc_infer: how a phoenix_async chunk is made to continue the running one at inference (any checkpoint)
RTC_INFER_MODES = ("off", "inpaint", "guidance")
# phoenix_stall_steps: no progress = every joint within STALL_RAD and both grippers within STALL_GRIPPER (opening
# fraction) of where they were that many env steps before
STALL_RAD = 0.02
STALL_GRIPPER = 0.05
# phoenix_stall_fix, what a stalled chunk is drawn with: none (counted only), reseed (seed phoenix_seed + the stalled
# chunks in a row), cover (cam_high black) or mask
# (cam_high encoded as missing); cover and mask for phoenix_stall_cover_chunks chunks, then one as is, and again
STALL_FIXES = ("none", "reseed", "cover", "mask")
# phoenix_nudge: when NUDGE_CHUNKS chunks in a row plan almost no motion -- every arm
# joint's range inside the chunk and its step from the last command below NUDGE_STILL_RAD, the grippers within
# NUDGE_STILL_GRIPPER (opening fraction) -- the next chunk is replaced by a small move along the last clear motion
# direction (the joint that moved most goes NUDGE_RAD, the others in proportion; grippers as planned).
NUDGE_STILL_RAD = 0.01
NUDGE_STILL_GRIPPER = 0.02
NUDGE_CHUNKS = 2
NUDGE_RAD = 0.03


def stalled(before: np.ndarray, after: np.ndarray) -> bool:
    """True when the arms made no progress from state ``before`` to state ``after`` (14-D, observation.state units)."""
    moved = np.abs(np.asarray(after, dtype=np.float32) - np.asarray(before, dtype=np.float32))
    return bool(moved[ARM_DIMS].max() < STALL_RAD and moved[GRIPPER_DIMS].max() < STALL_GRIPPER)


def rtc_infer_mode(value) -> str:
    """phoenix_rtc_infer as one of RTC_INFER_MODES (YAML reads a bare off as False)."""
    mode = "off" if value in (None, False, "") else str(value).strip().lower()
    mode = "off" if mode in ("false", "none", "0") else mode
    if mode not in RTC_INFER_MODES:
        raise ValueError(f"phoenix_rtc_infer must be one of {RTC_INFER_MODES}, got {value!r}")
    return mode


class AsyncChunks:
    """The chunk timeline of phoenix_async (upstream prometheus's H32 / s / d execution, at the control rate).

    A chunk predicted from the observation at env step L has its index 0 at step L. The first chunk of an episode is
    computed synchronously and runs from index 0; every later one is launched once the running chunk has executed
    ``steps`` (s) of its own actions, from the observation then, and takes over when it arrives, at index (arrival
    step - L): on time that is after ``delay`` (d) steps, the ones the old chunk still had committed (the prefix a
    Training RTC model is handed). A late chunk takes over later and the old one keeps running meanwhile (holding its
    last target past its end); an early one takes over at once.
    """

    def __init__(self, steps: int, delay: int, horizon: int = ACTION_HORIZON):
        if not (0 < delay < horizon and steps >= 1 and steps + delay <= horizon):
            raise ValueError(f"phoenix_async needs 0 < d and s + d <= {horizon} (s={steps}, d={delay})")
        self.steps, self.delay, self.horizon = int(steps), int(delay), int(horizon)
        self.reset()

    def reset(self) -> None:
        self.step = 0  # env steps executed this episode
        self.chunk = None  # (horizon, 14) env-unit targets of the running chunk
        self.t0 = 0  # env step of its index 0
        self.start = 0  # its first own index: 0 for the first chunk, d for the others
        self.launches = self.late = 0
        self.delays = []  # steps from launch to takeover

    def launch_due(self) -> bool:
        return self.chunk is not None and self.step - self.t0 >= self.start + self.steps

    def committed(self) -> np.ndarray:
        """The running chunk's next d targets: what executes while the next one is computed."""
        index = self.step - self.t0
        rows = [min(i, self.horizon - 1) for i in range(index, index + self.delay)]
        return self.chunk[rows]

    def running(self) -> np.ndarray:
        """The running chunk's targets from the current step on (row 0: the one executed now), what a chunk launched
        now overlaps (phoenix_rtc_infer); past its end, its last target."""
        return self.chunk[min(self.step - self.t0, self.horizon - 1) :]

    def land(self, chunk: np.ndarray, launch_step: int) -> None:
        first = self.chunk is None
        self.chunk, self.t0, self.start = np.asarray(chunk), int(launch_step), 0 if first else self.delay
        if not first:
            self.launches += 1
            self.delays.append(self.step - launch_step)
            self.late += self.step - launch_step > self.delay

    def action(self) -> np.ndarray:
        return self.chunk[min(self.step - self.t0, self.horizon - 1)]

    def describe(self) -> str:
        if not self.delays:
            return "no background chunk"
        return (
            f"{self.launches} background chunks, took over after {np.median(self.delays):g} steps (median, "
            f"max {max(self.delays)}; planned {self.delay}), {self.late} late"
        )


class SerialPrefix:
    """Training RTC with serial inference (no inference while the arm moves).

    The first chunk of an episode is planned without a prefix. Every later one is planned from the observation after
    the previous chunk executed ``steps`` (s) targets, and is handed that chunk's next ``delay`` (d) planned targets
    as its prefix: the model holds them fixed while it denoises the rest, they come back as the new chunk's first d
    steps, and the arm executes them, so consecutive chunks join where the old plan was heading instead of jumping to
    a new plan. d <= the trained maximum (10), 0 < d < s, s + d <= 32.
    """

    def __init__(self, delay: int, horizon: int = ACTION_HORIZON):
        self.delay, self.horizon = int(delay), int(horizon)
        self.reset()

    def check(self, steps: int) -> None:
        if not (0 < self.delay < steps and steps + self.delay <= self.horizon):
            raise ValueError(f"serial Training RTC needs 0 < d < s and s + d <= {self.horizon} "
                             f"(s=phoenix_steps {steps}, d=phoenix_prefix_steps {self.delay})")  # fmt: skip

    def reset(self) -> None:
        self.chunk = None  # (horizon, 14) env-unit targets of the last chunk
        self.executed = 0  # how many of them the arm executed
        self.chunks = 0

    def prefix(self) -> np.ndarray | None:
        """The last chunk's d targets after the ones the arm executed of it (None for an episode's first chunk, and
        when the last chunk's plan was left)."""
        return None if self.chunk is None else self.chunk[self.executed : self.executed + self.delay]

    def keep(self, chunk: np.ndarray | None, executed: int) -> None:
        """The chunk planned last and how many of its targets the arm executes (the phoenix_steps it was planned with, which
        may be changed before the next chunk); None when the arm did not execute its plan (a nudge replaced it),
        so the next chunk does not go back to it."""
        self.chunk = None if chunk is None else np.asarray(chunk)
        self.executed = int(executed)
        self.chunks += 1


def default_worker_python() -> str:
    """Phoenix's interpreter: $PHOENIX_PYTHON, else WS_ROOT/venvs/phoenix."""
    if os.environ.get("PHOENIX_PYTHON"):
        return os.environ["PHOENIX_PYTHON"]
    ws_root = Path(os.environ.get("WS_ROOT", REPO_ROOT.parent))
    venv_python = ws_root / "venvs" / "phoenix" / "bin" / "python"
    return str(venv_python) if venv_python.exists() else sys.executable


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _any_true(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.any().item())
    if isinstance(value, np.ndarray):
        return bool(value.any())
    return bool(value)


def _extract_image(obs, camera: str) -> np.ndarray:
    """One camera of env 0 as contiguous HWC uint8 RGB."""
    image = _to_numpy(obs["sensor"][camera]["color"])[0]
    if image.ndim != 3:
        raise ValueError(f"expected HWC image for {camera}, got {image.shape}")
    image = image[..., :3]
    if np.issubdtype(image.dtype, np.floating):
        if float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255)
    return np.ascontiguousarray(image.astype(np.uint8))


def encode_obs(obs) -> dict[str, np.ndarray]:
    """Environment observation -> the arrays phoenix_worker.py reads (state = robot/qpos = observation.state)."""
    encoded = {cam: _extract_image(obs, cam) for cam in CAMERAS}
    encoded["state"] = _to_numpy(obs["robot"]["qpos"])[0].astype(np.float32).reshape(-1)
    return encoded


def format_env_action(action, env) -> torch.Tensor:
    """Absolute 14-D joint target -> (1, D) tensor on the env device."""
    action_array = np.asarray(action, dtype=np.float32).reshape(-1)
    env_action_dim = int(np.prod(env.unwrapped.single_action_space.shape))
    if action_array.shape[0] < env_action_dim:
        raise ValueError(f"policy action has dim {action_array.shape[0]}, env expects {env_action_dim}")
    action_tensor = torch.as_tensor(action_array[:env_action_dim], dtype=torch.float32, device=env.unwrapped.device)
    return action_tensor.unsqueeze(0)


def worker_command(usr_args: dict, python_bin: str) -> list[str]:
    """The worker command line for a deploy_policy.yml / --overrides dict: phoenix_worker.py for a post-train run
    (config.yaml), rtc_worker.py for a Training RTC run (model.yaml, export_rtc.py's layout)."""
    run_dir = Path(str(usr_args.get("checkpoint_dir") or "")).expanduser()
    if not str(usr_args.get("checkpoint_dir") or ""):
        raise ValueError("set checkpoint_dir (a Phoenix training output dir, or a copy of it)")
    run_dir = run_dir if run_dir.is_absolute() else REPO_ROOT / run_dir
    if (run_dir / "config.yaml").is_file():
        script = "phoenix_worker.py"
    elif (run_dir / "model.yaml").is_file():
        script = "rtc_worker.py"
    else:
        raise FileNotFoundError(f"{run_dir} is not a Phoenix run dir (no config.yaml, no Training RTC model.yaml)")
    cmd = [python_bin, str(POLICY_ROOT / script), "--run-dir", str(run_dir), "--device", "cuda:0",
           "--step", str(int(usr_args.get("phoenix_step", 0) or 0)),
           "--replan-steps", str(int(usr_args.get("phoenix_steps", 10))),
           "--denoise-steps", str(int(usr_args.get("phoenix_denoise_steps", 5))),
           "--seed", str(int(usr_args.get("phoenix_seed", 0))),
           "--text-encoder", str(usr_args.get("phoenix_text_encoder") or "cache")]  # fmt: skip
    for key, flag in (("phoenix_encoders", "--encoders"), ("phoenix_wan_base", "--wan-base")):
        if usr_args.get(key):
            cmd += [flag, str(Path(str(usr_args[key])).expanduser())]
    if str(usr_args.get("phoenix_compile", False)).lower() in ("true", "1", "yes", "default"):
        if script != "phoenix_worker.py":
            raise ValueError("phoenix_compile is for post-train runs (phoenix_worker.py), not a Training RTC run")
        cmd += ["--compile", "default"]  # torch.compile around the MoT blocks (phoenix_worker.compile_trunk)
    if str(usr_args.get("phoenix_frame_cache", False)).lower() in ("true", "1", "yes"):
        if script != "phoenix_worker.py":
            raise ValueError("phoenix_frame_cache is for post-train runs (phoenix_worker.py), not a Training RTC run")
        cmd += ["--frame-cache"]  # the current frame's trunk tokens once per inference (phoenix_worker.FrameCache)
    return cmd


def worker_key(usr_args: dict) -> tuple:
    """What a worker is started with, but for what a running one takes (its execution and denoising steps, its noise
    seed): a client for another task with the same key can drive the same worker (a runner can keep it across a task
    switch)."""
    cmd = worker_command(usr_args, usr_args.get("phoenix_python") or default_worker_python())
    for flag in ("--replan-steps", "--denoise-steps", "--seed"):
        index = cmd.index(flag)
        del cmd[index : index + 2]
    return (*cmd, str(usr_args.get("phoenix_cuda_visible_devices", usr_args.get("gpu_id", 0))))


class PhoenixClient:
    """Drives phoenix_worker.py over a stdio JSON pipe."""

    stall_steps, stall_fix, stall_counts = 0, "none", None  # phoenix_stall_* off, also for clients built bare
    smoother, smooth_history = None, 8  # phoenix_smooth off for clients built bare
    sampling, last_tail = "single", None  # phoenix_sampling single for clients built bare
    inpaint, inpaint_last, joins, stall_change = None, None, None, None  # phoenix_smooth inpaint<h> off likewise
    nudge, nudge_counts, still_chunks, last_motion, last_command = False, None, 0, None, None  # phoenix_nudge off
    serial = None  # serial Training RTC off for clients built bare
    rtc_infer, rtc_beta = "off", 5.0  # phoenix_rtc_infer off likewise

    def __init__(self, usr_args: dict, worker: subprocess.Popen | None = None):
        """``worker``: the running worker of a client with the same worker_key, taken over instead of starting one."""
        self.python_bin = usr_args.get("phoenix_python") or default_worker_python()
        prompts = usr_args.get("phoenix_prompts") or {}
        # the trained prompt of this task (the T5 cache is keyed by it), else the environment's instruction
        self.prompt = str(usr_args.get("phoenix_prompt") or prompts.get(usr_args.get("task_name"), "") or "")
        # trained on opening fractions (0 closed, 1 open): the environment takes metres per finger (0.05)
        self.gripper_scale = float(usr_args.get("phoenix_gripper_scale", 0.05))
        self.steps = int(usr_args.get("phoenix_steps", 10))
        # the model sees one frame and every chunk is drawn with the same noise, so where it predicts "wait" it gets the
        # same frame back and waits again: count those chunks (phoenix_stall_steps > 0) and draw them differently
        # (phoenix_stall_fix)
        self.seed = int(usr_args.get("phoenix_seed", 0))
        self.stall_steps = int(usr_args.get("phoenix_stall_steps", 0) or 0)
        self.stall_fix = str(usr_args.get("phoenix_stall_fix", "none") or "none").lower()
        if self.stall_fix not in STALL_FIXES:
            raise ValueError(f"phoenix_stall_fix must be one of {STALL_FIXES}, got {self.stall_fix!r}")
        if self.stall_fix != "none" and self.stall_steps <= 0:
            raise ValueError("phoenix_stall_fix needs phoenix_stall_steps > 0 (the steps without progress)")
        self.cover_chunks = max(1, int(usr_args.get("phoenix_stall_cover_chunks", 2)))
        self.smoother = chunk_smoother.from_usr_args(usr_args)  # phoenix_smooth: each chunk smoothed before execution
        self.smooth_history = int(usr_args.get("phoenix_smooth_history", 8) or 8)  # for a smoother set at run time
        self.inpaint = chunk_inpaint.from_usr_args(usr_args)  # phoenix_smooth inpaint<h>: chunks continue the last tail
        self.inpaint_last = self.inpaint
        self.joins = chunk_inpaint.JoinSteps()
        # phoenix_sampling: single, or bidN (Bidirectional Decoding's backward coherence, arXiv 2408.17355): N chunks
        # with different noise seeds, the one closest to the previous chunk's unexecuted tail is executed
        self.sampling = self.set_sampling(str(usr_args.get("phoenix_sampling", "single") or "single"))
        self.last_tail = None  # the executed chunk's planned but unexecuted targets (bid compares, lipo blends)
        self.set_nudge(usr_args.get("phoenix_nudge", False))
        self.history: list[tuple[int, np.ndarray]] = []  # (env steps executed before the chunk, state then)
        self.executed = self.stalls = self.episode_stalls = 0
        # never reset (reset_model may also run inside an episode): a runner labels an episode with the difference
        if self.stall_steps > 0:
            self.stall_counts = {"fix": self.stall_fix, "steps": self.stall_steps, "stalled": 0, "changed": 0}
        self.async_mode = str(usr_args.get("phoenix_async", "off")).lower()
        if self.async_mode not in ASYNC_MODES:
            raise ValueError(f"phoenix_async must be one of {ASYNC_MODES}, got {self.async_mode!r}")
        self.rtc_prefix = str(usr_args.get("phoenix_rtc_prefix", False)).lower() in ("true", "1", "yes")
        if (self.smoother is not None or self.inpaint is not None or self.sampling != "single") \
                and self.async_mode != "off":  # fmt: skip
            raise ValueError("phoenix_smooth / phoenix_sampling need synchronous chunks: set phoenix_async off")
        # RTC at inference: a background chunk continues the running one (inpainting, or RTC's guidance)
        self.rtc_infer = rtc_infer_mode(usr_args.get("phoenix_rtc_infer"))
        self.rtc_beta = float(usr_args.get("phoenix_rtc_beta", 5.0))  # RTC's guidance weight clip (PI's default)
        if self.rtc_infer != "off" and self.async_mode == "off":
            raise ValueError("phoenix_rtc_infer joins asynchronous chunks: set phoenix_async real "
                             "(synchronous chunks: phoenix_smooth inpaint<h>soft)")  # fmt: skip
        if self.rtc_infer != "off" and self.rtc_prefix:
            raise ValueError("one of phoenix_rtc_prefix (a Training RTC checkpoint) and phoenix_rtc_infer, not both")
        # d: control steps one inference takes (0.42 s of model + the pipe at 10 Hz: 6 leaves some margin)
        self.timeline = AsyncChunks(self.steps, int(usr_args.get("phoenix_inference_delay", 6))) \
            if self.async_mode != "off" else None  # fmt: skip
        # synchronous Training RTC (inference and execution alternate): each chunk continues the
        # phoenix_prefix_steps targets the previous one planned after the ones it executed
        if self.rtc_prefix and self.timeline is None:
            if self.inpaint is not None or self.sampling != "single":
                raise ValueError("serial Training RTC already joins the chunks: leave inpaint and bidN out")
            self.serial = SerialPrefix(int(usr_args.get("phoenix_prefix_steps", 4)))
            self.serial.check(self.steps)
        self.executor = ThreadPoolExecutor(max_workers=1) if self.timeline is not None else None
        self.pending_future = None  # a runner waits for it before reset_model
        self.rpc_lock = threading.Lock()  # one request at a time on the pipe (the async chunk thread uses it too)
        self.pending_launch = None
        self.cuda_visible_devices = str(usr_args.get("phoenix_cuda_visible_devices", usr_args.get("gpu_id", 0)))
        self.worker_key = worker_key(usr_args)
        if worker is None:
            worker_env = os.environ.copy()
            # DexSim cannot be pinned with CUDA_VISIBLE_DEVICES; only the worker gets the mask
            worker_env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
            worker_env.pop("PYTHONPATH", None)  # the sim venv's paths must not leak into the Phoenix interpreter
            self.proc = subprocess.Popen(worker_command(usr_args, self.python_bin), stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1, env=worker_env)  # fmt: skip
        else:  # the model is loaded: only this task's steps go to it
            self.proc = worker
            self._rpc({"cmd": "set", "replan_steps": self.steps,
                       "denoise_steps": int(usr_args.get("phoenix_denoise_steps", 5)), "seed": self.seed})  # fmt: skip
            self._rpc({"cmd": "reset"})
        atexit.register(self.close)
        self.info = self._rpc({"cmd": "info"})["info"]
        print(f"[phoenix] worker={self.python_bin} gpu={self.cuda_visible_devices} checkpoint={self.info['checkpoint']} "
              f"step={self.info['step']} prompt slot={self.info['embodiment_id']}\n"
              f"[phoenix] steps/inference={self.info['replan_steps']} chunk={self.info['action_horizon']} "
              f"denoise={self.info['denoise_steps']} seed={self.info['seed']} text={self.info['text_encoder']} "
              f"gripper_scale={self.gripper_scale} relative_dims={self.info.get('relative_dims') or 'none'} "
              f"prompt={self.prompt or '(environment instruction)'}", flush=True)  # fmt: skip
        if self.stall_steps > 0:
            print(f"[phoenix] stall: no progress over {self.stall_steps} steps, then {self.stall_fix}", flush=True)
        if self.smoother is not None:
            print(f"[phoenix] smooth: {self.smoother.describe()}", flush=True)
        if self.inpaint is not None:
            self._check_inpaint()
            print(f"[phoenix] inpaint: {self._inpaint_note()}", flush=True)
        if self.rtc_prefix and not self.info.get("rtc_prefix"):
            raise ValueError(
                f"{self.info['checkpoint']} is not a Training RTC checkpoint: leave phoenix_rtc_prefix off"
            )
        if self.serial is not None:
            if self.serial.delay > int(self.info.get("max_prefix_steps", 0)):
                raise ValueError(f"phoenix_prefix_steps {self.serial.delay} > the trained maximum "
                                 f"{self.info.get('max_prefix_steps')}")  # fmt: skip
            print(f"[phoenix] serial Training RTC: each chunk continues the {self.serial.delay} steps the previous "
                  f"one planned after its {self.steps}", flush=True)  # fmt: skip
        if self.timeline is not None:
            steps, delay = self.timeline.steps, self.timeline.delay
            print(f"[phoenix] async: s={steps} d={delay} "
                  f"prefix={'committed steps (Training RTC)' if self.rtc_prefix else 'none'}", flush=True)  # fmt: skip
            if steps + 2 * delay > ACTION_HORIZON:  # a chunk runs from its index d to d + s + d
                print(f"[phoenix] async: s + 2 d = {steps + 2 * delay} > {ACTION_HORIZON}: every chunk ends "
                      f"{steps + 2 * delay - ACTION_HORIZON} steps before the next takes over (the arm holds its last "
                      "target meanwhile); lower phoenix_steps", flush=True)  # fmt: skip
        if self.info.get("compile", "off") != "off":  # (with phoenix_frame_cache: its second token shape too)
            took = self._warm_up()
            print("[phoenix] compiled trunk: " + ("no phoenix_prompt to warm up with, the first inference compiles"
                                                  if took is None else f"warmed up in {took:.1f} s"), flush=True)  # fmt: skip
        if self.rtc_infer != "off":
            self._start_rtc_infer()

    def _check_inpaint(self) -> None:
        if not self.info.get("inpaint"):
            raise ValueError(f"this Phoenix worker cannot inpaint ({self.info['checkpoint']}): leave inpaint out of "
                             "phoenix_smooth")  # fmt: skip

    def _inpaint_note(self) -> str:
        inpaint, steps = self.inpaint, self.execution_steps
        left = ACTION_HORIZON - steps
        if left <= 0:
            return f"{inpaint.key}, but phoenix_steps {steps} executes the whole chunk: nothing to continue"
        rest = max(0, left - inpaint.steps)
        soft = rest if inpaint.soft_steps is None else min(inpaint.soft_steps, rest)
        held = f"{min(inpaint.steps, left)} exactly" + (f", {soft} more softly" if inpaint.soft else "")
        return f"{inpaint.key}: each chunk continues the {left} steps of the last one not executed ({held})"

    def _report_joins(self) -> None:
        summary = self.joins.summary() if self.joins is not None else None
        if summary is not None:
            print(f"[phoenix] episode: {summary} (phoenix_smooth now {self.smoothing})", flush=True)
        if self.joins is not None:
            self.joins.reset()

    def _joined(self, rows: list[np.ndarray]) -> list[np.ndarray]:
        """Counts the step from the last executed target into this chunk (JoinSteps), for every setting, on the targets
        handed to execution (a nudge's when it replaced the plan's; before any smoother)."""
        if self.joins is not None and len(rows):
            self.joins.chunk(rows, len(rows))
        return rows

    def detach(self) -> subprocess.Popen | None:
        """Give the worker up to the next client (PhoenixClient(..., worker=)) instead of closing it; None if it exited."""
        self._report_joins()
        if getattr(self, "history", None):
            self._report_stalls()
        if getattr(self, "executor", None) is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
        proc, self.proc = getattr(self, "proc", None), None
        return proc if proc is not None and proc.poll() is None else None

    def close(self) -> None:
        self._report_joins()
        if getattr(self, "history", None):
            self._report_stalls()
        if getattr(self, "executor", None) is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
        proc = getattr(self, "proc", None)
        if proc is None or proc.poll() is not None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
            proc.wait(timeout=10)

    def _start_rtc_infer(self) -> None:
        """phoenix_rtc_infer: what the worker has to offer for it, what it will do, and RTC guidance's warm-up."""
        need = {"inpaint": "inpaint", "guidance": "rtc_guidance"}[self.rtc_infer]
        if not self.info.get(need):
            raise ValueError(f"this Phoenix worker has no {need} ({self.info['checkpoint']}): leave phoenix_rtc_infer "
                             "off (a Training RTC run takes phoenix_rtc_prefix instead)")  # fmt: skip
        steps, delay = self.timeline.steps, self.timeline.delay
        soft = max(0, ACTION_HORIZON - steps - 2 * delay)  # the running chunk's steps after the d it still executes
        print(f"[phoenix] RTC at inference ({self.rtc_infer}): a background chunk continues the running one, its next "
              f"{delay} steps at weight 1 and {soft} more by the soft mask (the first chunk's: "
              f"{max(0, ACTION_HORIZON - steps - delay)})"
              + ("" if soft else "; no soft steps: the join moves to the takeover, lower phoenix_steps"), flush=True)  # fmt: skip
        if self.rtc_infer != "guidance":
            return
        took = self._warm_up(rtc={"prev": np.zeros((delay + 1, 14)).tolist(), "delay": delay, "beta": self.rtc_beta})
        if took is None:
            print("[phoenix] RTC guidance: no phoenix_prompt to warm up with, the first background chunk will be late "
                  "(it compiles the attention's backward kernel)", flush=True)  # fmt: skip
        else:
            print(f"[phoenix] RTC guidance warmed up in {took:.1f} s", flush=True)

    def _warm_up(self, **extra) -> float | None:
        """Two inferences on a blank observation, so that what compiles at first use (phoenix_compile's trunk, RTC
        guidance's backward kernel) does so before the episode; the seconds it took, None without a prompt to use."""
        if not self.prompt:
            return None
        started = time.monotonic()
        blank = {camera: np.zeros((480, 640, 3), dtype=np.uint8) for camera in CAMERAS}
        blank["state"] = np.zeros(14, dtype=np.float32)
        for _ in range(2):
            self._infer(blank, "", full=True, **extra)
        self._rpc({"cmd": "reset"})
        return time.monotonic() - started

    def _rpc(self, payload: dict) -> dict:
        with self.rpc_lock:
            return self._rpc_unlocked(payload)

    def _rpc_unlocked(self, payload: dict) -> dict:
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("Phoenix worker exited; its traceback is on stderr above")
            line = line.strip()
            if not line:
                continue
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not response.get("ok", False):
                raise RuntimeError(response.get("error", "Phoenix worker error"))
            return response

    def _stall_change(self, state: np.ndarray) -> str | None:
        """phoenix_stall_fix's change for this chunk ("reseed", "cover", "mask"), or None: no stall, or its pause."""
        if self.stall_steps <= 0:
            return None
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        before = [past for executed, past in self.history if executed <= self.executed - self.stall_steps]
        self.history.append((self.executed, state))
        self.executed += self.execution_steps  # what runs of this chunk before the next request
        if not (before and stalled(before[-1], state)):
            self.stalls = 0
            return None
        self.stalls += 1
        self.episode_stalls += 1
        self.stall_counts["stalled"] += 1
        change = None
        if self.stall_fix == "reseed" and self.seed >= 0:
            change = "reseed"
        elif self.stall_fix in ("cover", "mask") and (self.stalls - 1) % (self.cover_chunks + 1) < self.cover_chunks:
            change = self.stall_fix  # a while, then one chunk as is, then again while it stays stalled
        if change is not None:
            self.stall_counts["changed"] += 1
        return change

    def _report_stalls(self) -> None:
        if self.stall_steps > 0 and self.history:
            print(f"[phoenix] episode: {self.episode_stalls} of {len(self.history)} chunks after no progress over "
                  f"{self.stall_steps} steps ({self.stall_fix})", flush=True)  # fmt: skip
        self.history, self.executed, self.stalls, self.episode_stalls = [], 0, 0, 0

    def _prepare(self, obs: dict[str, np.ndarray], extra: dict) -> dict[str, np.ndarray]:
        """phoenix_stall_fix's change of this chunk's request (once per chunk: it counts the chunk)."""
        change = self.stall_change = self._stall_change(obs["state"])
        if change == "reseed":
            extra["seed"] = self.seed + self.stalls
        elif change == "cover":
            obs = {**obs, "cam_high": np.zeros_like(obs["cam_high"])}
        elif change == "mask":
            extra["pad_cameras"] = ["cam_high"]
        return obs

    def _request(self, obs: dict[str, np.ndarray], instruction: str, **extra) -> np.ndarray:
        return self._infer(self._prepare(obs, extra), instruction, **extra)

    def _infer(self, obs: dict[str, np.ndarray], instruction: str, **extra) -> np.ndarray:
        fd, path = tempfile.mkstemp(prefix="phoenix_obs_", suffix=".npz")
        os.close(fd)
        try:
            np.savez(path, **obs)
            response = self._rpc({"cmd": "act", "obs_path": path, "prompt": self.prompt or instruction, **extra})
        finally:
            os.remove(path)
        actions = np.asarray(response["actions"], dtype=np.float32)
        actions[:, GRIPPER_DIMS] *= self.gripper_scale
        return actions

    def act(self, obs: dict[str, np.ndarray], instruction: str) -> list[np.ndarray]:
        """One replan chunk of absolute joint targets, grippers scaled for the environment."""
        return list(self._request(obs, instruction))

    def plan(self, obs: dict[str, np.ndarray], instruction: str) -> tuple[list[np.ndarray], np.ndarray | None]:
        """The steps to execute of the next chunk and the previous chunk's unexecuted tail (for lipo). With bidN or lipo
        the worker returns whole chunks: bidN draws N with seeds 1000 apart and keeps the one whose start is closest to
        the previous chunk's tail on the arm joints (the first: no tail yet). With phoenix_smooth inpaint<h> too, and
        each chunk is drawn continuing that tail."""
        # a UI may change these from another thread: read once, one setting per chunk
        inpaint, smoother, sampling = self.inpaint, self.smoother, self.sampling
        if self.serial is not None:  # serial Training RTC: the chunk continues what the previous one planned next
            extra = {"full": True}
            obs = self._prepare(obs, extra)
            tail, steps = self.last_tail, self.execution_steps
            prefix = self.serial.prefix()
            if prefix is not None and len(prefix) and self.stall_change is None:  # a stall fix leaves the stalled plan
                held = np.array(prefix, dtype=np.float32)
                held[:, GRIPPER_DIMS] /= self.gripper_scale  # the model's opening fractions
                extra["prefix"] = held.tolist()
            chunk = self._infer(obs, instruction, **extra)
            actions, left = self._to_run(list(chunk[:steps]), chunk[steps:])
            self.serial.keep(None if left else chunk, steps)
            return actions, tail
        samples = int(sampling[3:]) if sampling.startswith("bid") else 1
        lipo = smoother is not None and smoother.method == "lipo"
        if samples == 1 and not lipo and inpaint is None:
            self.last_tail = None
            return self._joined(self._after_plan(self.act(obs, instruction))), None
        extra: dict = {"full": True}
        obs = self._prepare(obs, extra)
        reseeded = "seed" in extra  # phoenix_stall_fix reseed: this chunk's own seed
        base = int(extra.pop("seed", self.seed))
        tail, steps = self.last_tail, self.execution_steps
        if inpaint is not None and tail is not None and len(tail) and self.stall_change is None:
            held = np.array(tail, dtype=np.float32)  # a stall fix is there to leave the stalled plan: not continued
            held[:, GRIPPER_DIMS] /= self.gripper_scale  # the model's opening fractions
            extra.update(inpaint=held.tolist(), inpaint_steps=inpaint.steps, inpaint_soft=inpaint.soft,
                         inpaint_soft_steps=inpaint.soft_steps)  # fmt: skip
        best, best_cost = None, np.inf
        for i in range(samples):
            seed = {"seed": base + 1000 * i if base >= 0 else base} if i or reseeded else {}
            chunk = self._infer(obs, instruction, **extra, **seed)
            if tail is None or not len(tail) or samples == 1:  # nothing to compare with (phoenix_steps 32): one
                best = chunk
                break
            n = min(len(tail), len(chunk))
            weights = 0.9 ** np.arange(n)  # the nearest steps count most
            arm = chunk_smoother.ARM_DIMS
            cost = float(weights @ np.abs(chunk[:n][:, arm] - tail[:n][:, arm]).sum(1))
            if cost < best_cost:
                best, best_cost = chunk, cost
        return self._to_run(list(best[:steps]), best[steps:])[0], tail

    def _to_run(self, rows: list[np.ndarray], rest: np.ndarray) -> tuple[list[np.ndarray], bool]:
        """The ``rows`` of a whole chunk that execute, after phoenix_nudge (_after_plan) and counted at the join as such
        (_joined), and whether a nudge replaced them. ``rest``, the chunk's targets after them, is the next chunk's tail
        unless one did: the arm left that plan, so the next chunk does not continue (inpaint), compare with (bidN) or
        blend from (lipo) it."""
        nudged = (self.nudge_counts or {}).get("nudged", 0)
        actions = self._joined(self._after_plan(rows))
        left = (self.nudge_counts or {}).get("nudged", 0) != nudged  # a nudge replaced the plan's targets
        self.last_tail = None if left else np.asarray(rest)
        return actions, left

    sampling_choices = ("single", "bid2", "bid3")

    def set_nudge(self, on) -> bool:
        """phoenix_nudge on / off from the next chunk on."""
        self.nudge = str(on).lower() in ("1", "true", "yes", "on")
        if self.nudge and self.nudge_counts is None:
            self.nudge_counts = {"nudged": 0}  # never reset: a runner labels an episode with the difference
        return self.nudge

    def _after_plan(self, actions: list[np.ndarray]) -> list[np.ndarray]:
        """phoenix_nudge on the chunk about to run; keeps the last command and the last clear motion direction."""
        planned = np.asarray(actions, dtype=np.float64)
        arm = list(chunk_smoother.ARM_DIMS)
        if self.nudge and self.last_command is not None and len(planned):
            scale = max(float(getattr(self, "gripper_scale", 0.05)), 1e-9)
            inside = float(np.ptp(planned[:, arm], axis=0).max())
            step = float(np.abs(planned[0, arm] - self.last_command[arm]).max())
            grip_inside = float(np.ptp(planned[:, GRIPPER_DIMS], axis=0).max())
            grip_step = float(np.abs(planned[0, GRIPPER_DIMS] - self.last_command[GRIPPER_DIMS]).max())
            grip = max(grip_inside, grip_step) / scale
            still = inside < NUDGE_STILL_RAD and step < NUDGE_STILL_RAD and grip < NUDGE_STILL_GRIPPER
            self.still_chunks = self.still_chunks + 1 if still else 0
            if self.still_chunks >= NUDGE_CHUNKS and self.last_motion is not None:
                direction = self.last_motion / max(float(np.abs(self.last_motion).max()), 1e-9)
                ramp = (np.arange(1, len(planned) + 1) / len(planned))[:, None]
                planned = planned.copy()
                planned[:, arm] = self.last_command[arm] + NUDGE_RAD * ramp * direction
                self.still_chunks = 0
                self.nudge_counts["nudged"] += 1
                print(f"[phoenix] nudge: {NUDGE_CHUNKS} still chunks, {NUDGE_RAD} rad on along the last motion",
                      flush=True)  # fmt: skip
        if len(planned):
            start = self.last_command if self.last_command is not None else planned[0]
            motion = planned[-1, arm] - start[arm]
            if float(np.abs(motion).max()) >= NUDGE_STILL_RAD:
                self.last_motion = motion
            self.last_command = planned[-1].copy()
        return list(planned.astype(np.float32))

    def set_sampling(self, value: str) -> str:
        """phoenix_sampling from the next chunk on: single, bid2, bid3 (or bidN)."""
        value = str(value or "single").strip().lower()
        if value != "single" and not (value.startswith("bid") and value[3:].isdigit() and int(value[3:]) >= 2):
            raise ValueError(f"phoenix_sampling must be single or bidN (N >= 2), got {value!r}")
        if value != "single" and getattr(self, "timeline", None) is not None:
            raise ValueError("phoenix_sampling works on synchronous chunks only: set phoenix_async off")
        if value != "single" and self.serial is not None:
            raise ValueError("serial Training RTC already joins the chunks: phoenix_sampling stays single")
        self.sampling = value
        return value

    def chunk(self, obs: dict[str, np.ndarray], instruction: str, prefix: np.ndarray | None = None,
              running: np.ndarray | None = None) -> np.ndarray:  # fmt: skip
        """All 32 targets of a chunk (env units); ``prefix``: committed env-unit targets it has to continue (Training
        RTC); ``running``: the running chunk's env-unit targets from this observation on, which phoenix_rtc_infer
        makes it continue (the first d, executed meanwhile, at weight 1; the rest by RTC's soft mask)."""
        extra: dict = {"full": True}
        if prefix is not None:
            prefix = np.array(prefix, dtype=np.float32)
            prefix[:, GRIPPER_DIMS] /= self.gripper_scale  # the model's opening fractions
            extra["prefix"] = prefix.tolist()
        obs = self._prepare(obs, extra)
        mode = self.rtc_infer  # read once
        # a stall fix is there to leave the stalled plan: that chunk does not continue it
        if mode != "off" and running is not None and len(running) and self.stall_change is None:
            rows = np.array(running, dtype=np.float32)
            rows[:, GRIPPER_DIMS] /= self.gripper_scale
            delay = min(self.timeline.delay, len(rows))
            if mode == "guidance":
                extra["rtc"] = {"prev": rows.tolist(), "delay": delay, "beta": self.rtc_beta}
            else:
                extra.update(inpaint=rows.tolist(), inpaint_steps=delay, inpaint_soft=True, inpaint_soft_steps=None)
        return self._infer(obs, instruction, **extra)

    def describe_async(self) -> dict | None:
        """What an episode's label records as policy_async (None: synchronous chunks)."""
        if getattr(self, "timeline", None) is None:
            return None
        described = {"steps": self.timeline.steps, "delay": self.timeline.delay, "rtc_infer": self.rtc_infer,
                     "rtc_prefix": bool(getattr(self, "rtc_prefix", False)), "launches": self.timeline.launches,
                     "late": self.timeline.late, "delays": [int(d) for d in self.timeline.delays]}  # fmt: skip
        if self.rtc_infer == "guidance":
            described["rtc_beta"] = self.rtc_beta
        return described

    @property
    def denoise_steps(self) -> int:
        """The worker's denoising steps now."""
        return int(self.info["denoise_steps"])

    def set_denoise_steps(self, steps: int) -> int:
        """Denoising steps from the next chunk on, without reloading the model."""
        self.info = self._rpc({"cmd": "set", "denoise_steps": int(steps)})["info"]
        return self.denoise_steps

    @property
    def noise_seed(self) -> int:
        """Every chunk's noise seed now: >= 0 the same noise every chunk, < 0 fresh noise."""
        return int((getattr(self, "info", None) or {}).get("seed", getattr(self, "seed", 0)))  # an older worker's info

    def set_noise_seed(self, seed: int) -> int:
        """The noise seed from the next chunk on, without reloading the model (bidN and the stall reseed count from
        it too)."""
        self.info = self._rpc({"cmd": "set", "seed": int(seed)})["info"]
        self.seed = self.noise_seed
        return self.seed

    @property
    def execution_steps(self) -> int:
        """Steps of each chunk executed before the next inference (phoenix_steps)."""
        return self.timeline.steps if self.timeline is not None else self.steps

    @property
    def execution_steps_max(self) -> int:
        delay = self.timeline.delay if self.timeline is not None else self.serial.delay if self.serial else 0
        return ACTION_HORIZON - delay

    @property
    def smoothing(self) -> str:
        """phoenix_smooth now (none, gauss1, lipo1, inpaint2soft, ...)."""
        return chunk_inpaint.join(self.inpaint, self.smoother_key)

    smoothing_choices = chunk_smoother.CHOICES  # the smoothers; the inpainting has its own setter (set_inpaint)
    inpaint_hold_choices, inpaint_soft_choices = chunk_inpaint.HOLD_CHOICES, chunk_inpaint.SOFT_CHOICES

    @property
    def smoother_key(self) -> str:
        """The smoothing part of phoenix_smooth : none, gauss1, lipo1, ..."""
        smoother = self.smoother  # read once, as in plan()
        return smoother.key if smoother is not None else "none"

    @property
    def inpaint_setting(self) -> dict:
        """The inpainting part: steps held exactly (0: off), soft steps (None: all)."""
        inpaint = self.inpaint
        if inpaint is None:  # off; switched on at run time, it blends over all the steps not executed
            return {"steps": 0, "soft_steps": None}
        return {"steps": inpaint.steps, "soft_steps": inpaint.soft_steps if inpaint.soft else 0}

    def describe_smoothing(self) -> dict | None:
        """What an episode's label records as policy_smooth (None: neither smoothing nor inpainting)."""
        smoother, inpaint = self.smoother, self.inpaint  # read once, as in plan()
        if smoother is None and inpaint is None:
            return None
        described = smoother.describe() if smoother is not None else {"method": "none"}
        return {**described, "inpaint": inpaint.describe()} if inpaint is not None else described

    def set_smoothing(self, value: str) -> str:
        """phoenix_smooth from the next chunk on, without reloading the model; the new smoother continues from the
        commands already sent, inpainting from the last chunk's tail."""
        if getattr(self, "timeline", None) is not None:
            raise ValueError("phoenix_smooth works on synchronous chunks only: set phoenix_async off")
        inpaint, _ = chunk_inpaint.split(value)
        smoother = chunk_smoother.build(value, history=self.smooth_history)
        if inpaint is not None and self.serial is not None:
            raise ValueError("serial Training RTC already joins the chunks: no inpaint")
        if inpaint is not None:
            self._check_inpaint()
            self.inpaint_last = inpaint  # what set_smooth_method("inpaint") brings back after another choice
        if smoother is not None and self.smoother is not None:
            smoother._sent = self.smoother._sent  # noqa: SLF001 - the same commands, a new fit
        self.smoother, self.inpaint = smoother, inpaint
        if inpaint is not None:
            print(f"[phoenix] inpaint: {self._inpaint_note()}", flush=True)
        return self.smoothing

    def set_smoother(self, value: str) -> str:
        """The smoothing only, at run time; the inpainting kept."""
        return self.set_smoothing(chunk_inpaint.join(self.inpaint, chunk_inpaint.split(value)[1]))

    def set_inpaint(self, steps: int, soft_steps: int | None) -> str:
        """The inpainting only, at run time (chunk_inpaint.compose); the smoothing kept."""
        return self.set_smoothing(chunk_inpaint.join(chunk_inpaint.compose(steps, soft_steps), self.smoother_key))

    smooth_method_choices = (*chunk_smoother.CHOICES, "inpaint", "bid2", "bid3")

    @property
    def smooth_method(self) -> str:
        """One way of joining chunks as a single choice: the smoother, "inpaint" or bidN when only that one is on, "none", or the
        ones on joined with + (a deploy_policy.yml combination)."""
        parts = [self.smoother_key] if self.smoother_key != "none" else []
        parts += ["inpaint"] if self.inpaint is not None else []
        parts += [self.sampling] if self.sampling != "single" else []
        return "+".join(parts) or "none"

    def set_smooth_method(self, method: str) -> str:
        """One way of joining chunks at a time, set at run time -- a smoother, "inpaint" (with the held / soft steps it
        had last, else 2 held and the rest soft) or bidN (phoenix_sampling) -- the others off; returns phoenix_smooth."""
        method = str(method or "none").strip().lower()
        if method.startswith("bid"):
            self.set_sampling(method)  # checks it first
            return self.set_smoothing("none")
        if method == "inpaint":
            inpaint = self.inpaint or self.inpaint_last or chunk_inpaint.compose(2, None)
            method = chunk_inpaint.join(inpaint, "none")
        now = self.set_smoothing(method)
        self.set_sampling("single")
        return now

    def set_execution_steps(self, steps: int) -> int:
        """phoenix_steps from the next chunk on, without reloading the model."""
        steps = int(steps)
        if not 1 <= steps <= self.execution_steps_max:
            raise ValueError(f"steps per chunk must be 1..{self.execution_steps_max}, got {steps}")
        if self.serial is not None:
            self.serial.check(steps)
        if self.timeline is not None:  # async: the timeline launches the next chunk after s of its own steps
            self.timeline.steps = steps
        else:  # sync: the worker returns that many steps of each chunk
            self.info = self._rpc({"cmd": "set", "replan_steps": steps})["info"]
        self.steps = steps
        return steps

    def reset(self) -> None:
        self._report_stalls()
        self._report_joins()
        self.last_tail, self.still_chunks, self.last_motion, self.last_command = None, 0, None, None
        smoother = self.smoother  # read once, as in plan()
        if smoother is not None:
            smoother.reset()
        if self.pending_future is not None:
            self.pending_future.cancel()
            self.pending_future = None
        if self.timeline is not None:
            if self.timeline.launches:
                print(f"[phoenix] episode: {self.timeline.describe()}", flush=True)
            self.timeline.reset()
        if self.serial is not None:
            self.serial.reset()
        self._rpc({"cmd": "reset"})


def get_model(usr_args):
    return PhoenixClient(usr_args)


def obs_state(obs) -> np.ndarray:
    """The measured 14-D state of an environment observation (grippers as opening fractions)."""
    return _to_numpy(obs["robot"]["qpos"])[0].astype(np.float64).reshape(-1)


def execute_chunk(env, actions, obs, smoother=None, tail=None):
    """Step through one chunk (smoothed first, phoenix_smooth; ``tail``: the previous chunk's unexecuted targets),
    stopping at success or termination; returns (obs, info, truncated)."""
    final_obs, info, truncated = obs, {}, False
    if smoother is not None:
        actions = smoother(actions, obs_state(obs), tail=tail)
    for action in actions:
        if smoother is not None:
            smoother.record(action)
        final_obs, _reward, terminated, truncated, info = env.step(format_env_action(action, env))
        if _any_true(env.get_wrapper_attr("is_task_success")()):
            break
        if _any_true(terminated) or _any_true(truncated):
            break
    return final_obs, info, truncated


def eval_async(env, model, obs):
    """phoenix_async: up to s env steps, the next chunk computed in the background (AsyncChunks)."""
    instruction = getattr(env, "_current_instruction", None) or ""
    timeline, inference_times = model.timeline, []
    final_obs, info, truncated = obs, {}, False
    for _ in range(timeline.steps):
        if model.pending_future is not None and model.pending_future.done():
            launch_step, started_at = model.pending_launch
            chunk = model.pending_future.result()  # a worker error surfaces here
            finish_inference(started_at, inference_times, None)
            timeline.land(chunk, launch_step)
            model.pending_future = model.pending_launch = None
        if timeline.chunk is None:  # the episode's first chunk: nothing runs yet, so it is computed inline
            started_at = start_inference(None)
            timeline.land(model.chunk(encode_obs(final_obs), instruction), timeline.step)
            finish_inference(started_at, inference_times, None)
        elif model.pending_future is None and timeline.launch_due():
            prefix = timeline.committed() if model.rtc_prefix else None
            running = timeline.running() if getattr(model, "rtc_infer", "off") != "off" else None
            model.pending_launch = (timeline.step, start_inference(None))
            model.pending_future = model.executor.submit(
                model.chunk, encode_obs(final_obs), instruction, prefix, running
            )
        final_obs, _reward, terminated, truncated, info = env.step(format_env_action(timeline.action(), env))
        timeline.step += 1
        if _any_true(env.get_wrapper_attr("is_task_success")()):
            break
        if _any_true(terminated) or _any_true(truncated):
            break
    return final_obs, info, truncated, inference_times


def eval(env, model, obs):
    if getattr(model, "timeline", None) is not None:
        return eval_async(env, model, obs)
    instruction = getattr(env, "_current_instruction", None) or ""
    inference_times: list[float] = []
    started_at = start_inference(None)  # includes obs serialization and the pipe (bidN: all N)
    if hasattr(model, "plan"):
        actions, tail = model.plan(encode_obs(obs), instruction)
    else:
        actions, tail = model.act(encode_obs(obs), instruction), None
    finish_inference(started_at, inference_times, None)
    final_obs, info, truncated = execute_chunk(env, actions, obs, getattr(model, "smoother", None), tail)
    return final_obs, info, truncated, inference_times


def reset_model(model):
    model.reset()


def close_model(model):
    model.close()
