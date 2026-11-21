"""Tests of the adapter and the worker that need no GPU and no weights. Two of them import upstream code and are
skipped without a Phoenix-MWA checkout at policy/phoenix/Phoenix-MWA.

    python -m pytest -q tests
"""

import importlib
import importlib.util
from pathlib import Path
import time
import types

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeEnv:
    """The environment interface deploy_policy.eval needs: observations with three cameras and a 14-D state, steps
    that only count."""

    _current_instruction = ""

    def __init__(self):
        self.unwrapped = types.SimpleNamespace(single_action_space=types.SimpleNamespace(shape=(14,)), device="cpu")
        self.step_count = 0

    def _obs(self):
        import torch  # noqa: PLC0415

        image = np.zeros((1, 4, 4, 3), dtype=np.uint8)
        return {"sensor": {camera: {"color": image} for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist")},
                "robot": {"qpos": torch.zeros(1, 14)}}  # fmt: skip

    def reset(self):
        self.step_count = 0
        return self._obs()

    def get_wrapper_attr(self, name):
        return lambda: False

    def step(self, action):
        time.sleep(0.002)  # a control period, so that a background inference gets to run between steps
        self.step_count += 1
        return self._obs(), 0.0, False, False, {}


def test_phoenix_async_timeline_launches_after_s_steps_and_takes_over_after_d(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    adapter = importlib.import_module("policy.phoenix.deploy_policy")
    with pytest.raises(ValueError, match="s \\+ d"):
        adapter.AsyncChunks(steps=28, delay=6)
    tl = adapter.AsyncChunks(steps=16, delay=6)
    first = np.arange(32, dtype=np.float32)[:, None].repeat(14, 1)
    tl.land(first, 0)  # the episode's first chunk runs from index 0
    assert not tl.launch_due()
    tl.step = 16
    assert tl.launch_due()  # s of its own actions done
    np.testing.assert_array_equal(tl.committed()[:, 0], np.arange(16, 22))  # the next d it still runs
    tl.step = 22  # on time: d steps later
    tl.land(first + 100, 16)
    assert tl.action()[0] == 106  # the new chunk at index d
    assert not tl.launch_due()
    tl.step = 16 + 6 + 16
    assert tl.launch_due()  # s of the new chunk's own actions after its takeover
    tl.step = 48
    tl.land(first + 200, 38)  # 10 steps after its launch: late
    assert tl.late == 1
    assert tl.delays == [6, 10]
    tl.step = 38 + 40
    assert tl.action()[0] == 231  # past its end: its last target held


def test_phoenix_async_keeps_the_arm_moving_while_the_next_chunk_is_computed(monkeypatch):
    import threading  # noqa: PLC0415
    import time  # noqa: PLC0415
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    monkeypatch.syspath_prepend(str(REPO_ROOT))
    adapter = importlib.import_module("policy.phoenix.deploy_policy")
    env = FakeEnv()
    calls, launches = [], []

    class Executor(ThreadPoolExecutor):  # notes the step each background inference is launched at
        def submit(self, fn, *args):
            launches.append(env.step_count)
            return super().submit(fn, *args)

    def chunk(obs, instruction, prefix=None, running=None):
        launch = launches[-1] if calls else env.step_count
        if calls:  # a background chunk: the arm has to keep stepping while it is computed
            deadline = time.monotonic() + 5
            while env.step_count < launch + 3 and time.monotonic() < deadline:
                time.sleep(0.001)
        calls.append((launch, threading.current_thread() is threading.main_thread(), prefix, running))
        out = np.zeros((32, 14), dtype=np.float32)
        out[:, 0] = 0.001 * launch + 0.0001 * np.arange(32)
        out[:, [6, 13]] = 0.05  # env units (the client scaled them)
        return out

    client = adapter.PhoenixClient.__new__(adapter.PhoenixClient)
    client.__dict__.update(prompt="", gripper_scale=0.05, rtc_prefix=True, chunk=chunk,
                           timeline=adapter.AsyncChunks(steps=8, delay=4), executor=Executor(1),
                           pending_future=None, pending_launch=None)  # fmt: skip
    obs = env.reset()
    for _ in range(3):
        obs, _info, _truncated, _times = adapter.eval(env, client, obs)
    assert env.step_count == 24  # s steps per call, never held up by an inference
    assert calls[0][:2] == (0, True)  # the first chunk inline
    assert [c[0] for c in calls[1:3]] == [8, 20]  # launched at L, runs from its index d for s: next at L + d + s
    assert not any(c[1] for c in calls[1:])  # the others in the background
    prefix = calls[1][2]
    assert prefix.shape == (4, 14)
    np.testing.assert_allclose(prefix[:, 0], 0.0001 * np.arange(8, 12), atol=1e-7)  # the first chunk's next d
    assert client.timeline.delays[0] in (3, 4)
    assert calls[1][3] is None  # phoenix_rtc_infer off: the running chunk is not handed over
    client.executor.shutdown(wait=True)

    # phoenix_rtc_infer: the background chunk also gets the running chunk's targets from its launch on
    calls.clear(), launches.clear()
    env = FakeEnv()
    client.__dict__.update(rtc_prefix=False, rtc_infer="inpaint", timeline=adapter.AsyncChunks(steps=8, delay=4),
                           executor=Executor(1), pending_future=None, pending_launch=None)  # fmt: skip
    obs = env.reset()
    for _ in range(2):
        obs, _info, _truncated, _times = adapter.eval(env, client, obs)
    client.executor.shutdown(wait=True)
    assert calls[1][2] is None and calls[1][3].shape == (24, 14)  # the first chunk from its index 8 to its end
    np.testing.assert_allclose(calls[1][3][:, 0], 0.0001 * np.arange(8, 32), atol=1e-7)

    # synchronous Training RTC (serial inference) is checked before any worker starts: s + d <= 32,
    # and no inpaint / bidN on top of it (a post-train checkpoint is refused once its worker reports no prefix)
    with pytest.raises(ValueError, match="serial Training RTC needs"):
        adapter.PhoenixClient({"phoenix_rtc_prefix": True, "phoenix_steps": 32, "checkpoint_dir": "/nowhere"})
    with pytest.raises(ValueError, match="leave inpaint and bidN out"):
        adapter.PhoenixClient({"phoenix_rtc_prefix": True, "phoenix_sampling": "bid2", "checkpoint_dir": "/nowhere"})


def test_phoenix_rtc_at_inference_is_asked_of_the_worker_per_background_chunk(monkeypatch):
    """phoenix_rtc_infer (with phoenix_async real): chunk() sends the running chunk's rows in model units, the d that
    execute meanwhile at weight 1 -- as inpainting with RTC's soft mask, or as RTC guidance -- and not after a stall
    fix; the settings are checked before any worker starts and the label names them."""
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    adapter = importlib.import_module("policy.phoenix.deploy_policy")
    assert [adapter.rtc_infer_mode(v) for v in (None, False, "off", "Guidance", "inpaint")] == [
        "off", "off", "off", "guidance", "inpaint"]  # fmt: skip
    with pytest.raises(ValueError, match="phoenix_rtc_infer must be one of"):
        adapter.rtc_infer_mode("rtc")
    with pytest.raises(ValueError, match="set phoenix_async real"):
        adapter.PhoenixClient({"phoenix_rtc_infer": "inpaint", "checkpoint_dir": "/nowhere"})
    # phoenix_steps is the re-planning period with phoenix_async too: longer than d, and period + d within the chunk
    for steps, delay in ((6, 6), (28, 6)):
        with pytest.raises(ValueError, match="phoenix_inference_delay < phoenix_steps"):
            adapter.PhoenixClient({"phoenix_async": "real", "phoenix_steps": steps, "phoenix_inference_delay": delay,
                                   "checkpoint_dir": "/nowhere"})  # fmt: skip
    with pytest.raises(ValueError, match="not both"):
        adapter.PhoenixClient({"phoenix_rtc_infer": "guidance", "phoenix_async": "real", "phoenix_rtc_prefix": True,
                               "checkpoint_dir": "/nowhere"})  # fmt: skip

    tl = adapter.AsyncChunks(steps=14, delay=6)
    first = np.arange(32, dtype=np.float32)[:, None].repeat(14, 1)
    tl.land(first, 0)
    tl.step = 14
    assert tl.running()[:, 0].tolist() == list(range(14, 32))
    tl.step = 40  # past its end: its last target
    assert tl.running()[:, 0].tolist() == [31]

    sent = []

    def infer(obs, instruction, **extra):
        sent.append(extra)
        return np.zeros((32, 14), dtype=np.float32)

    client = adapter.PhoenixClient.__new__(adapter.PhoenixClient)
    client.__dict__.update(prompt="", gripper_scale=0.05, _infer=infer, timeline=adapter.AsyncChunks(14, 6),
                           rtc_infer="inpaint")  # fmt: skip
    obs = {"state": np.zeros(14, dtype=np.float32)}
    running = np.zeros((18, 14), dtype=np.float32)
    running[:, 0] = np.arange(18)
    running[:, adapter.GRIPPER_DIMS] = 0.03  # env units: metres per finger
    client.chunk(obs, "", None, running)
    assert sent[-1]["full"] and "rtc" not in sent[-1]
    assert (sent[-1]["inpaint_steps"], sent[-1]["inpaint_soft"], sent[-1]["inpaint_soft_steps"]) == (6, True, None)
    rows = np.array(sent[-1]["inpaint"])
    assert rows.shape == (18, 14) and rows[:, 0].tolist() == list(range(18))
    assert np.allclose(rows[:, adapter.GRIPPER_DIMS], 0.6)  # the model's opening fractions
    assert np.allclose(running[:, adapter.GRIPPER_DIMS], 0.03)  # the timeline's chunk is left as it was

    client.rtc_infer, client.rtc_beta = "guidance", 5.0
    client.chunk(obs, "", None, running[:4])  # fewer rows left than d: all of them at weight 1
    assert "inpaint" not in sent[-1]
    assert (sent[-1]["rtc"]["delay"], sent[-1]["rtc"]["beta"], len(sent[-1]["rtc"]["prev"])) == (4, 5.0, 4)
    client.chunk(obs, "", None, None)  # an episode's first chunk: nothing runs yet
    assert set(sent[-1]) == {"full"}
    client.stall_steps, client.stall_fix, client.cover_chunks = 2, "mask", 2  # a stalled chunk leaves the plan
    client.history, client.executed, client.stalls, client.episode_stalls = [(0, np.zeros(14))], 2, 0, 0
    client.stall_counts = {"stalled": 0, "changed": 0}
    client.chunk(obs, "", None, running)
    assert sent[-1].get("pad_cameras") == ["cam_high"] and "rtc" not in sent[-1] and "inpaint" not in sent[-1]

    label = client.describe_async()
    assert (label["period"], label["steps"], label["delay"], label["rtc_infer"], label["rtc_beta"]) == (
        20, 14, 6, "guidance", 5.0)  # fmt: skip
    client.timeline = None
    assert client.describe_async() is None


def test_phoenix_steps_is_the_replanning_period_with_async_chunks_too(monkeypatch):
    """set_execution_steps: the worker's steps with synchronous chunks; with phoenix_async the timeline's, the next
    chunk launched d steps before the period is over."""
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    adapter = importlib.import_module("policy.phoenix.deploy_policy")
    sent = []
    client = adapter.PhoenixClient.__new__(adapter.PhoenixClient)
    client.__dict__.update(steps=20, timeline=None, serial=None, info={},
                           _rpc=lambda payload: sent.append(payload) or {"ok": True, "info": {"replan_steps": 10}})  # fmt: skip
    assert client.set_execution_steps(10) == 10
    assert sent == [{"cmd": "set", "replan_steps": 10}]
    with pytest.raises(ValueError, match="1..32"):
        client.set_execution_steps(33)
    client.timeline = adapter.AsyncChunks(steps=16, delay=6)  # async: the timeline, not the worker
    assert client.execution_steps == 22 and client.execution_steps_max == 26  # the period: 16 of its own + 6
    client.set_execution_steps(20)  # every 20 steps: launched after 14 of the running chunk's own
    assert client.timeline.steps == 14 and client.execution_steps == 20
    with pytest.raises(ValueError, match="7..26"):
        client.set_execution_steps(6)  # no longer than one inference
    assert len(sent) == 1


def test_phoenix_rtc_guidance_wraps_upstreams_function_around_the_joint_loop(monkeypatch):
    """phoenix_worker.RtcGuidance on a stand-in of upstream's joint loop (action pre_dit -> mot -> action post_dit):
    beta 0 leaves the loop's result as it is, a positive beta gives upstream's corrected velocity (checked against the
    closed form of a linear model), the attention runs with the training kernel options meanwhile, and everything
    patched is back afterwards."""
    import torch  # noqa: PLC0415

    upstream = REPO_ROOT / "policy/phoenix/Phoenix-MWA/src/wam/inference/online_action_policy.py"
    if not upstream.is_file():
        pytest.skip("Phoenix-MWA submodule not checked out")
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    worker = importlib.import_module("policy.phoenix.phoenix_worker")
    monkeypatch.syspath_prepend(str(upstream.parents[2]))  # the worker's own path entry, gone with an earlier test
    pytest.importorskip("wam.inference.online_action_policy")
    horizon, dim, gain, bias = 6, 4, 0.3, 0.2
    seen = []

    class Expert(torch.nn.Module):
        action_dim = dim

        def pre_dit(self, action_tokens, timestep, context=None):
            return {"tokens": action_tokens * 1.0, "timestep": timestep}

        def post_dit(self, tokens, pre_state):
            return tokens

    class MoT(torch.nn.Module):
        flex_training_kernel_options = {"bwd_BLOCK_M1": 64}

        def __init__(self):
            super().__init__()
            self._flex_attention = lambda q, k, v, mask, options, *rest: seen.append(options) or q

        def forward(self, embeds_all, attention_mask=None):
            self._flex_attention(embeds_all["action"], None, None, None, {"BLOCK_M": 64})
            return {"video": embeds_all["video"] + 1.0, "action": gain * embeds_all["action"] + bias}

    class Model:
        device, torch_dtype = torch.device("cpu"), torch.float32

        def __init__(self):
            self.action_expert, self.mot = Expert(), MoT()
            self.infer_action_scheduler = type("S", (), {"num_train_timesteps": 1000})()

    model = Model()
    torch.manual_seed(0)
    noise, target, video = torch.randn(1, horizon, dim), torch.randn(1, horizon, dim), torch.zeros(1, 3, 2)
    pad = torch.zeros(1, dim, dtype=torch.bool)
    pad[0, 3] = True  # a padded action dim: never guided
    sigmas = [1.0, 0.8, 0.6, 0.4, 0.2]

    def loop():  # upstream's frame_joint loop, reduced to the calls RtcGuidance stands on
        x, velocities, videos = noise.clone(), [], []
        with torch.no_grad():
            for sigma in sigmas:
                pre = model.action_expert.pre_dit(action_tokens=x, timestep=torch.tensor([sigma * 1000.0]))
                out = model.mot(embeds_all={"video": video, "action": pre["tokens"]}, attention_mask=None)
                velocity = model.action_expert.post_dit(out["action"], pre)
                velocities.append(velocity)
                videos.append(out["video"])
                x = x - 0.2 * velocity
        return x, velocities, videos

    plain, plain_velocities, plain_videos = loop()
    assert seen and all(options == {"BLOCK_M": 64} for options in seen)
    seen.clear()
    with worker.RtcGuidance(model, target, 2, 4, pad, beta=0.0) as guided:
        unguided, _, videos = loop()
    # (upstream's function zeroes the velocity of a padded dim, as its joint loop zeroes those dims after every step)
    assert guided.steps == len(sigmas) and torch.allclose(unguided[..., :3], plain[..., :3], atol=1e-6)
    assert all(torch.equal(a, b) for a, b in zip(videos, plain_videos, strict=True))  # the video tokens as they were
    assert all(options == {"bwd_BLOCK_M1": 64} for options in seen)  # the training kernel options meanwhile
    for module, name in ((model.action_expert, "pre_dit"), (model.action_expert, "post_dit"), (model.mot, "forward")):
        assert name not in vars(module)  # the class's own methods again
    seen.clear()
    model.mot(embeds_all={"video": video, "action": noise}, attention_mask=None)
    assert seen == [{"BLOCK_M": 64}]

    with worker.RtcGuidance(model, target, 2, 4, pad, beta=5.0):
        pulled, velocities, _ = loop()
    # the linear model's closed form: clean = x - sigma (gain x + bias), d clean / d x = 1 - sigma gain
    from wam.inference.online_action_policy import _rtc_guidance_weight  # noqa: PLC0415
    from wam.inference.online_action_policy import get_rtc_prefix_weights  # noqa: PLC0415

    weights = get_rtc_prefix_weights(2, 4, horizon, "exp").view(1, horizon, 1)
    assert weights.flatten().tolist()[:2] == [1.0, 1.0] and weights.flatten().tolist()[4:] == [0.0, 0.0]
    x = noise.clone()
    for sigma, velocity in zip(sigmas, velocities, strict=True):
        own = (gain * x + bias).masked_fill(pad.view(1, 1, -1), 0.0)
        error = ((target.masked_fill(pad.view(1, 1, -1), 0.0) - (x - sigma * own)) * weights)
        error = error.masked_fill(pad.view(1, 1, -1), 0.0)
        weight = _rtc_guidance_weight(torch.tensor([sigma]), max_guidance_weight=5.0)
        expected = own - weight * (1.0 - sigma * gain) * error
        assert torch.allclose(velocity, expected, atol=1e-5)
        x = x - 0.2 * velocity
    first = lambda chunk: (chunk[:, :2, :3] - target[:, :2, :3]).abs().max()  # noqa: E731
    assert first(pulled) < 0.5 * first(plain)  # the first d steps drawn towards the running chunk
    assert torch.allclose(pulled[:, 4:, :3], plain[:, 4:, :3], atol=1e-6)  # past the rows it has: as unguided
    with pytest.raises(RuntimeError, match="no longer calls action pre_dit"), worker.RtcGuidance(
        model, target, 2, 4, pad
    ):
        model.mot(embeds_all={"video": video, "action": noise}, attention_mask=None)


def test_phoenix_compile_wraps_each_block_and_goes_back_to_eager(monkeypatch, tmp_path):
    """phoenix_compile: the adapter starts phoenix_worker.py with --compile; compile_trunk gives every MoT block its
    own compiled pair around upstream's two per-layer functions (the same results), restore_trunk takes them off and
    RTC guidance runs without them."""
    import torch  # noqa: PLC0415

    monkeypatch.syspath_prepend(str(REPO_ROOT))
    adapter = importlib.import_module("policy.phoenix.deploy_policy")
    worker = importlib.import_module("policy.phoenix.phoenix_worker")
    (tmp_path / "config.yaml").write_text("{}")
    command = adapter.worker_command({"checkpoint_dir": str(tmp_path), "phoenix_compile": True}, "python")
    assert command[-2:] == ["--compile", "default"]
    assert "--compile" not in adapter.worker_command({"checkpoint_dir": str(tmp_path)}, "python")
    assert adapter.worker_key({"checkpoint_dir": str(tmp_path), "phoenix_compile": True}) != adapter.worker_key(
        {"checkpoint_dir": str(tmp_path)})  # fmt: skip
    assert worker.parse_args(["--run-dir", "x", "--compile", "default"]).compile == "default"
    for part in ("trunk", "encoder", "reduce-overhead"):  # one of the two parts, or CUDA graph replay
        assert worker.parse_args(["--run-dir", "x", "--compile", part]).compile == part

    names = []

    def fake_compile(function, **kwargs):
        names.append((function.__code__.co_name, kwargs))
        return function

    monkeypatch.setattr(torch, "compile", fake_compile)

    class Block(torch.nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale

    class Expert(torch.nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.blocks = torch.nn.ModuleList(Block(scale + i) for i in range(2))

    class MoT(torch.nn.Module):
        expert_order = ["video", "action"]

        def __init__(self):
            super().__init__()
            self.mixtures = torch.nn.ModuleDict({"video": Expert(1.0), "action": Expert(10.0)})

        def _build_expert_attention_io(self, expert, block, x, freqs, t_mod):
            return x * block.scale, freqs, t_mod

        @staticmethod
        def _apply_expert_post_block(block, residual_x, mixed_attn_out, gate_msa, shift_mlp, scale_mlp, gate_mlp,
                                     context_payload):  # fmt: skip
            context = 0.0 if context_payload is None else context_payload["context"] + context_payload["mask"]
            return residual_x + block.scale * mixed_attn_out + context

    model = type("Model", (), {})()
    model.mot = mot = MoT()
    x = torch.arange(3.0)

    def layer(block, payload):
        q, _, _ = mot._build_expert_attention_io(None, block, x, 1, 2)  # noqa: SLF001
        return mot._apply_expert_post_block(block, x, q, 0, 0, 0, 0, payload)  # noqa: SLF001

    blocks = [b for name in mot.expert_order for b in mot.mixtures[name].blocks]
    payload = {"context": torch.ones(3), "mask": torch.ones(3)}
    eager = [layer(b, p) for b in blocks for p in (None, payload)]
    assert worker.compile_trunk(model) == 4
    assert sorted(n for n, _ in names) == sorted(
        f"phoenix_{half}_{name}_{i}" for half in ("pre", "post") for name in ("video", "action") for i in range(2))  # fmt: skip
    assert all(kwargs == {"dynamic": False, "fullgraph": False} for _, kwargs in names)
    compiled = [layer(b, p) for b in blocks for p in (None, payload)]
    assert all(torch.equal(a, b) for a, b in zip(eager, compiled, strict=True))
    taken = worker.restore_trunk(model)
    assert set(taken) == {"_build_expert_attention_io", "_apply_expert_post_block"}
    assert "_build_expert_attention_io" not in vars(mot) and worker.restore_trunk(model) == {}

    # compile_encoder: the nine copies of the current frame go through DINO's backbone once
    assert worker.compile_encoder(model).startswith("left as it is")  # no such encoder here
    calls = []

    class Backbone(torch.nn.Module):
        def forward_features(self, images, masks):
            return images

    class Student(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dino = Backbone()

        def _dino_features(self, frames, patch_mask, require_grad=True):
            calls.append(tuple(frames.shape))
            return frames.flatten(2) * 2.0, frames.flatten(2).sum(-1, keepdim=True)

    student = Student()
    model.joint_visual_encoder = type("E", (), {"csvae_encoder": type("C", (), {"dino": type("D", (), {
        "student": student})()})()})()  # fmt: skip
    names.clear()
    assert "encoded once" in worker.compile_encoder(model)
    assert [n for n, _ in names] == ["forward_features"]
    frame = torch.rand(1, 1, 3, 4, 4)
    repeated = frame.repeat(1, 9, 1, 1, 1)
    patches, cls = student._dino_features(repeated, None, require_grad=False)  # noqa: SLF001
    assert calls == [(1, 1, 3, 4, 4)] and patches.shape == (1, 9, 48) and cls.shape == (1, 9, 1)
    assert torch.equal(patches, repeated.flatten(2) * 2.0)
    different = torch.rand(1, 9, 3, 4, 4)
    student._dino_features(different, None)  # noqa: SLF001
    student._dino_features(repeated, torch.ones(1, 9, 4, dtype=torch.bool))  # noqa: SLF001
    student._dino_features(frame, None)  # noqa: SLF001
    assert calls[1:] == [(1, 9, 3, 4, 4), (1, 9, 3, 4, 4), (1, 1, 3, 4, 4)]  # anything else: as upstream


def test_phoenix_frame_cache_gives_the_full_forwards_result_from_the_kept_current_frame(monkeypatch, tmp_path):
    """phoenix_worker.FrameCache on a small stand-in of upstream's MoT (same hooks, same joint96 mask): a step whose
    current-frame inputs are the kept ones runs on the future and action tokens only and gives what the full forward
    gives; a new frame, a masked token or a backward pass runs in full; uninstalled, upstream's methods are back.
    The adapter passes phoenix_frame_cache on as --frame-cache."""
    import torch  # noqa: PLC0415

    monkeypatch.syspath_prepend(str(REPO_ROOT))
    adapter = importlib.import_module("policy.phoenix.deploy_policy")
    worker = importlib.import_module("policy.phoenix.phoenix_worker")
    (tmp_path / "config.yaml").write_text("{}")
    assert adapter.worker_command({"checkpoint_dir": str(tmp_path), "phoenix_frame_cache": True}, "python")[-1] == (
        "--frame-cache")  # fmt: skip
    assert "--frame-cache" not in adapter.worker_command({"checkpoint_dir": str(tmp_path)}, "python")
    assert worker.parse_args(["--run-dir", "x", "--frame-cache"]).frame_cache

    dim, per_frame, frames, actions, layers = 5, 2, 3, 3, 3
    video_len = per_frame * frames
    torch.manual_seed(0)

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q, self.k, self.v, self.o = (torch.nn.Linear(dim, dim).double() for _ in range(4))
            self.c = torch.nn.Linear(dim, dim).double()

    class Expert(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList(Block() for _ in range(layers))

    class MoT(torch.nn.Module):
        expert_order = ["video", "action"]
        num_layers = layers

        def __init__(self):
            super().__init__()
            self.mixtures = torch.nn.ModuleDict({"video": Expert(), "action": Expert()})
            self.block_masks = []

        def _build_expert_attention_io(self, expert, block, x, freqs, t_mod):
            h = x * (1 + t_mod[..., 0, :] if t_mod.ndim == 4 else 1 + t_mod)
            return block.q(h) * freqs[:, :, 0], block.k(h) * freqs[:, :, 0], block.v(h), x, None, None, None, None, False

        def _build_flex_block_mask(self, *, attention_mask, batch_size, query_len, key_len, device):
            self.block_masks.append((query_len, key_len))
            return attention_mask.clone()

        def _mixed_attention(self, q_cat, k_cat, v_cat, attention_mask, flex_block_mask=None, flex_key_mask=None,
                             query_mask=None, allow_self_when_all_masked=False, force_sdpa=False):  # fmt: skip
            mask = attention_mask if flex_block_mask is None else flex_block_mask
            assert tuple(mask.shape) == (q_cat.shape[1], k_cat.shape[1])
            scores = (q_cat @ k_cat.transpose(1, 2)).masked_fill(~mask, float("-inf"))
            return scores.softmax(-1) @ v_cat

        def forward(self, embeds_all, attention_mask, freqs_all, context_all, t_mod_all, key_masks_all=None,
                    query_masks_all=None):  # fmt: skip
            tokens = dict(embeds_all)
            assert attention_mask.shape[0] == attention_mask.shape[1] == sum(t.shape[1] for t in tokens.values())
            for layer in range(self.num_layers):
                io = {name: self._build_expert_attention_io(expert=self.mixtures[name], block=self.mixtures[name].blocks[layer],
                                                            x=tokens[name], freqs=freqs_all[name], t_mod=t_mod_all[name])
                      for name in self.expert_order}  # fmt: skip
                mixed = self._mixed_attention(
                    q_cat=torch.cat([io[n][0] for n in self.expert_order], 1),
                    k_cat=torch.cat([io[n][1] for n in self.expert_order], 1),
                    v_cat=torch.cat([io[n][2] for n in self.expert_order], 1), attention_mask=attention_mask,
                    flex_block_mask=None, flex_key_mask=None, query_mask=None, force_sdpa=False)  # fmt: skip
                start = 0
                for name in self.expert_order:
                    block, length = self.mixtures[name].blocks[layer], tokens[name].shape[1]
                    context = context_all[name]  # the text rows each token may read
                    read = (context["mask"].double() @ block.c(context["context"])) / context["mask"].shape[-1]
                    tokens[name] = tokens[name] + block.o(mixed[:, start : start + length]) + read
                    start += length
            return tokens

    model = type("Model", (), {})()
    model.mot = mot = MoT()
    total = video_len + actions
    mask = torch.zeros(total, total, dtype=torch.bool)  # upstream's joint96 mask
    mask[:per_frame, :per_frame] = True
    mask[per_frame:video_len, :video_len] = True
    mask[video_len:, :] = True

    def inputs(current, future, action, step):
        video_t = torch.cat([torch.zeros(1, per_frame, 6, dim), torch.full((1, video_len - per_frame, 6, dim), 0.1 * step)], 1)
        return {
            "embeds_all": {"video": torch.cat([current, future], 1), "action": action},
            "attention_mask": mask.clone(),
            "freqs_all": {"video": freqs["video"], "action": freqs["action"]},
            "context_all": {"video": {"context": text, "mask": context_mask["video"]},
                            "action": {"context": text, "mask": context_mask["action"]}},
            "t_mod_all": {"video": video_t.double(), "action": torch.full((1, dim), 0.1 * step).double()},
            "key_masks_all": {"video": torch.ones(1, video_len, dtype=torch.bool), "action": torch.ones(1, actions, dtype=torch.bool)},
            "query_masks_all": {"video": torch.ones(1, video_len, dtype=torch.bool), "action": torch.ones(1, actions, dtype=torch.bool)},
        }  # fmt: skip

    rand = lambda *shape: torch.randn(*shape, dtype=torch.float64)  # noqa: E731
    freqs = {"video": rand(1, video_len, 1, dim), "action": rand(1, actions, 1, dim)}
    text = rand(1, 4, dim)
    context_mask = {"video": torch.rand(1, video_len, 4) > 0.3, "action": torch.rand(1, actions, 4) > 0.3}
    current = rand(1, per_frame, dim)
    steps = [inputs(current, rand(1, video_len - per_frame, dim), rand(1, actions, dim), k) for k in (5, 4, 3)]
    assert worker.FrameCache.current_tokens(mask, video_len) == per_frame
    assert worker.FrameCache.current_tokens(torch.ones(total, total, dtype=torch.bool), video_len) == 0

    with torch.no_grad():
        full = [mot(**step) for step in steps]
        cache = worker.FrameCache(model).install()
        got = [mot(**step) for step in steps]
        assert cache.counts == {"full": 1, "cached": 2} and len(cache.keys) == layers
        assert mot.block_masks == [(total - per_frame, total)]  # one mask for the live queries over every key
        for a, b in zip(full, got, strict=True):
            assert a["video"].shape == b["video"].shape == (1, video_len, dim)
            assert torch.allclose(a["video"], b["video"], atol=1e-10) and torch.allclose(a["action"], b["action"], atol=1e-10)

        other = inputs(rand(1, per_frame, dim), steps[0]["embeds_all"]["video"][:, per_frame:], steps[0]["embeds_all"]["action"], 5)
        expected = MoT.forward(mot, **other)
        again = mot(**other)  # a new current frame: in full, and the cache is its
        assert cache.counts == {"full": 2, "cached": 2} and torch.allclose(again["video"], expected["video"], atol=1e-10)
        padded = {**steps[1], "embeds_all": {**other["embeds_all"]}}
        padded["key_masks_all"] = {**padded["key_masks_all"], "video": torch.zeros(1, video_len, dtype=torch.bool)}
        mot(**padded)  # a masked token: never from the cache
        assert cache.counts["cached"] == 2
    with torch.enable_grad():  # a backward pass (RTC guidance) goes through upstream's forward as it is
        before = dict(cache.counts)
        mot(**steps[0])
        assert cache.counts == before
    cache.uninstall()
    for name in ("forward", "_build_expert_attention_io", "_mixed_attention"):
        assert name not in vars(mot)
    with torch.no_grad():
        assert torch.equal(mot(**steps[2])["action"], full[2]["action"])


def test_phoenix_inpainting_holds_the_known_steps_and_leaves_the_others_as_upstream(monkeypatch):
    """phoenix_worker.InpaintScheduler around upstream's flow-matching scheduler: the known steps come out exactly
    (weight 1), the free ones bit for bit as upstream's Euler steps; RTC's soft mask decays to 0 by the last known."""
    import torch  # noqa: PLC0415

    path = REPO_ROOT / "policy/phoenix/Phoenix-MWA/src/wam/model/backbones/wan22/schedulers/scheduler_continuous.py"
    if not path.is_file():
        pytest.skip("Phoenix-MWA submodule not checked out")
    spec = importlib.util.spec_from_file_location("phoenix_scheduler_continuous", path)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    worker = importlib.import_module("policy.phoenix.phoenix_worker")

    w = worker.inpaint_weights(12, 4, soft=False)
    assert w[:4].tolist() == [1.0] * 4 and not w[4:].any()
    soft = worker.inpaint_weights(12, 4, soft=True)
    assert soft[:4].tolist() == [1.0] * 4 and np.all(np.diff(soft[3:12]) < 0) and 0 < soft[11] < 0.2
    assert not soft[12:].any()
    assert worker.inpaint_weights(2, 4, soft=True)[:3].tolist() == [1.0, 1.0, 0.0]
    short = worker.inpaint_weights(12, 2, soft=True, soft_steps=4)  # soft_steps 4: the soft mask ends 4 steps after
    assert short[:2].tolist() == [1.0, 1.0] and np.all(np.diff(short[1:6]) < 0) and not short[6:].any()
    hard = worker.inpaint_weights(12, 2, soft=False)
    assert np.allclose(worker.inpaint_weights(12, 2, soft=True, soft_steps=0), hard)  # soft_steps 0: held steps only

    torch.manual_seed(0)
    x0, noise, known = torch.randn(1, 32, 80), torch.randn(1, 32, 80), torch.randn(1, 32, 80)

    def denoise(scheduler, dtype):
        timesteps, deltas = scheduler.build_inference_schedule(num_inference_steps=5, device="cpu", dtype=dtype)
        x = noise.to(dtype)
        for _t, delta in zip(timesteps, deltas, strict=True):
            x = scheduler.step((noise - x0).to(dtype), delta, x)  # the exact velocity of x0's straight path
        return x.float()

    inner = upstream.WanContinuousFlowMatchScheduler(shift=5.0)
    plain = denoise(inner, torch.float32)
    held = denoise(worker.InpaintScheduler(inner, known, torch.from_numpy(w).view(1, -1, 1)), torch.float32)
    assert torch.allclose(plain, x0, atol=1e-5)
    assert torch.allclose(held[:, :4], known[:, :4], atol=1e-5)
    assert torch.equal(held[:, 4:], plain[:, 4:])
    bf16 = denoise(worker.InpaintScheduler(inner, known, torch.from_numpy(w).view(1, -1, 1)), torch.bfloat16)
    assert torch.equal(bf16[:, 4:], denoise(inner, torch.bfloat16)[:, 4:])  # the model's own dtype too
    assert (bf16[:, :4] - known[:, :4]).abs().max() < 0.05
