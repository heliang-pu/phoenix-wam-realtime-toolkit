#!/usr/bin/env python
"""Phoenix-MWA Training RTC inference worker: phoenix_worker.py's protocol for a checkpoint trained with upstream's
prometheus branch (phoenix.train with training_time_rtc), so a chunk can continue the steps the previous one planned.

Runs under the Phoenix venv with the prometheus code (commit 823f009: $PHOENIX_PROMETHEUS_ROOT,
policy/phoenix/Phoenix-MWA-prometheus, or ../Phoenix-MWA-prometheus next to the repository) first on sys.path. Model
build, the strict restore, the checks and one inference follow upstream's phoenix/model.py (load_interface,
Model.predict: the views squashed to 256 x 256 bicubic as its training read them, z-score normalization.json, the
prefix held fixed through denoising and checked unchanged after) with five differences: the prompt's T5 context comes
from the run's text cache (the rows training read) for whichever task is asked, the denoising steps and the noise seed
are settable, a view can be encoded as missing, a run trained on centred squares (EXPORT.json image_crop
center_square: build_rtc_set.py --crop-square) gets each view cut the same way before the squash, and the replies are
phoenix_worker.py's.

    {"cmd": "info"}                                   -> {"ok": true, "info": {...}}
    {"cmd": "act", "obs_path": npz, "prompt": str}    -> {"ok": true, "actions": [[14 floats] x replan_steps]}
        + "full": true                                -> all 32 steps of the chunk
        + "prefix": [[14 floats] x d]                 -> the chunk continues these d planned steps (d <= the trained
                                                         maximum, 10): they come back unchanged as its first d steps
        + "seed": n                                   -> this chunk's noise seed instead of --seed (negative: fresh)
        + "pad_cameras": ["cam_high"]                 -> those views encoded as missing (camera_is_pad)
    {"cmd": "reset"}                                  -> {"ok": true}
    {"cmd": "set", "denoise_steps": n}                -> {"ok": true, "info": {...}}  (from the next chunk on)
    {"cmd": "set", "replan_steps": n}                 -> likewise: how many steps of each chunk "act" returns
    {"cmd": "set", "seed": n}                         -> likewise: every chunk's noise seed (negative: fresh noise)

The run dir (export_rtc.py puts it together from a training run):
    model.yaml           the training model definition (rtc_init.py's phoenix_model_rtc.yaml)
    stvae.pt             the frozen encoder as prometheus reads it (rtc_init.py: csvae.pt, renamed)
    normalization.json   prepared/normalization.json of the training data (its sha256 is in the weights)
    text_cache/          text_index.json + the T5 rows of the trained prompts
    checkpoints/step_NNNNNN/weights.pt
    EXPORT.json          image_crop: none (upstream: whole frames) or center_square
The npz and the units are phoenix_worker.py's: cam_high / cam_left_wrist / cam_right_wrist (uint8 HWC RGB), state
(14 joints + grippers as opening fractions); actions are absolute joint targets with opening fractions.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

POLICY_ROOT = Path(__file__).resolve().parent


def prometheus_root() -> Path:
    """$PHOENIX_PROMETHEUS_ROOT, else policy/phoenix/Phoenix-MWA-prometheus, else a checkout next to the repository
    (../Phoenix-MWA-prometheus)."""
    if os.environ.get("PHOENIX_PROMETHEUS_ROOT"):
        return Path(os.environ["PHOENIX_PROMETHEUS_ROOT"])
    for root in (POLICY_ROOT / "Phoenix-MWA-prometheus", POLICY_ROOT.parents[2] / "Phoenix-MWA-prometheus"):
        if (root / "phoenix" / "model.py").is_file():
            return root
    return POLICY_ROOT / "Phoenix-MWA-prometheus"


UPSTREAM = prometheus_root()
sys.path[:0] = [str(UPSTREAM), str(UPSTREAM / "src")]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402, N812

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")  # upstream's base, left wrist, right wrist (prepare order)
EMBODIMENT_ID = 18
EMBODIMENT = "fmc3_cobotmagic_piper_joint14"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True, help="model.yaml, stvae.pt, normalization.json, text_cache/")
    p.add_argument("--step", type=int, default=0, help="checkpoints/step_<step> (0: the latest)")
    p.add_argument("--encoders", type=Path, help="dir with dino110k.pt (default: <run-dir>/dino110k.pt)")
    p.add_argument("--wan-base", help="DIFFSYNTH_MODEL_BASE_PATH: Wan2.2 VAE + tokenizer (the DiT is not loaded)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--replan-steps", type=int, default=10)
    p.add_argument(
        "--denoise-steps", type=int, default=10, help="of both the future video and the actions (upstream 10)"
    )
    p.add_argument("--seed", type=int, default=0, help="noise seed of every chunk; negative: fresh noise")
    p.add_argument("--text-encoder", choices=("cache",), default="cache", help="only the trained prompts")
    return p.parse_args(argv)


def center_square(image: np.ndarray) -> np.ndarray:
    """The centred square of an HWC image (build_rtc_set.py --crop-square)."""
    height, width = image.shape[:2]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    return np.ascontiguousarray(image[top : top + side, left : left + side])


def weights_file(run_dir: Path, step: int) -> tuple[Path, int]:
    saved = {
        int(path.parent.name.removeprefix("step_")): path for path in run_dir.glob("checkpoints/step_*/weights.pt")
    }
    if not saved:
        raise FileNotFoundError(f"no checkpoints/step_*/weights.pt in {run_dir}")
    step = step or max(saved)
    if step not in saved:
        raise ValueError(f"no weights of step {step} here; saved steps: {sorted(saved)}")
    return saved[step], step


class Runtime:
    def __init__(self, args):
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        import wam.model.factory as factory
        from wam.model.modules.codecs.dino_stvae.provenance import bundled_config
        from wam.utils.config_resolvers import register_default_resolvers
        from wam.utils.torch24_compat import install_safe_serialization_compat

        from phoenix.checkpoint import _load_joint96_checkpoint
        from phoenix.config import sha256
        from phoenix.model import Normalizer
        from phoenix.training import validate_training_contract

        register_default_resolvers()
        install_safe_serialization_compat()
        torch.set_num_threads(1)
        torch.manual_seed(max(args.seed, 0))  # the generator a negative --seed (fresh noise every chunk) draws from
        self.args = args
        self.device = torch.device(args.device)
        run_dir = args.run_dir.resolve()
        self.checkpoint, self.step = weights_file(run_dir, args.step)
        self.normalization = run_dir / "normalization.json"
        self.text_index = run_dir / "text_cache" / "text_index.json"
        export = json.loads((run_dir / "EXPORT.json").read_text()) if (run_dir / "EXPORT.json").is_file() else {}
        self.image_crop = export.get("image_crop", "none")
        if self.image_crop not in ("none", "center_square"):
            raise ValueError(f"unknown image_crop {self.image_crop!r} in {run_dir / 'EXPORT.json'}")
        encoders = args.encoders or run_dir
        if args.wan_base:
            os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = args.wan_base
        os.environ["PHOENIX_STVAE_CHECKPOINT"] = str(run_dir / "stvae.pt")  # model.yaml's ${oc.env:...}
        os.environ["PHOENIX_ENCODER_CHECKPOINT"] = str(encoders / "dino110k.pt")

        # upstream load_interface's checks: the normalization training used, a completed Training RTC checkpoint, its
        # prefix contract, the run's prompt slot
        stats_digest = sha256(self.normalization)
        self.normalizer = Normalizer(json.loads(self.normalization.read_text()), "normalized")
        print(f"[phoenix-rtc] loading {self.checkpoint}", file=sys.stderr, flush=True)
        payload = torch.load(self.checkpoint, map_location="cpu", mmap=True, weights_only=True)
        if payload.get("training_inputs", {}).get("normalization", stats_digest) != stats_digest:
            raise ValueError(f"{self.normalization} is not the normalization these weights were trained with")
        if type(payload.get("step")) is not int or payload["step"] <= 0:
            raise ValueError("not a completed Training RTC checkpoint (no optimizer step)")
        payload = dict(payload)
        payload["joint96_metadata"] = copy.deepcopy(payload["joint96_metadata"])
        metadata = payload["joint96_metadata"]
        contract = metadata.get("soft_prompt", {}).get("training_contract", {})
        self.max_prefix = int(contract.get("rtc", {}).get("uniform_prefix", [0, 0])[1])
        validate_training_contract(contract, {"rtc": {"max_prefix_steps": self.max_prefix}})
        registry = metadata["soft_prompt"]["registry"]
        names, status = (
            registry["embodiments"],
            registry.get("slot_status", ["assigned"] * len(registry["embodiments"])),
        )
        if names[EMBODIMENT_ID] != EMBODIMENT or status[EMBODIMENT_ID] != "assigned":
            raise ValueError(f"prompt slot {EMBODIMENT_ID} is {names[EMBODIMENT_ID]!r} ({status[EMBODIMENT_ID]})")

        definition = OmegaConf.load(run_dir / "model.yaml")
        model_cfg = definition.model if "model" in definition else definition
        model_cfg = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
        rtc = model_cfg.action_dit_config.get("training_time_rtc", {})
        if rtc.get("enabled") is not True or rtc.get("max_delay_steps") != self.max_prefix:
            raise ValueError("model.yaml does not match the trained RTC contract")
        if not model_cfg.get("joint96_three_view"):
            raise ValueError("expected the Phoenix Joint96 model")
        model_cfg.joint_visual_codec = bundled_config(
            OmegaConf.to_container(model_cfg.joint_visual_codec, resolve=True)
        )
        model_cfg.load_text_encoder = False
        model_cfg.mot_attention_backend = "sdpa"
        model_cfg.mot_checkpoint_mixed_attn = False
        # the training definition pairs flex with masked batches (refused with sdpa) and gradient checkpointing
        model_cfg.mot_flex_masked_batches = False
        model_cfg.video_dit_config.use_gradient_checkpointing = False
        model_cfg.action_dit_config.use_gradient_checkpointing = False

        original = factory.load_wan22_ti2v_5b_components

        def checkpoint_components(**kwargs):
            kwargs["skip_dit_load_from_pretrain"] = True
            return original(**kwargs)

        with tempfile.TemporaryDirectory(prefix="phoenix-rtc-") as folder:
            registry_path = Path(folder) / "phoenix_embodiments.json"
            registry_path.write_text(json.dumps(registry))
            model_cfg.action_dit_config.embodiment_registry_path = str(registry_path)
            factory.load_wan22_ti2v_5b_components = checkpoint_components
            try:
                model = instantiate(model_cfg, model_dtype=torch.bfloat16, device="cpu")
            finally:
                factory.load_wan22_ti2v_5b_components = original
            _load_joint96_checkpoint(model, payload)  # strict: every policy tensor
        del payload
        model.to(self.device).eval().requires_grad_(requires_grad=False)
        model.set_runtime_device(self.device)
        self.model = model
        self.text = json.loads(self.text_index.read_text())
        self.contexts = {}
        self.calls = 0

    def info(self):
        return {"checkpoint": str(self.checkpoint), "step": self.step, "stats": str(self.normalization),
                "registry": "in the weights", "embodiment_id": EMBODIMENT_ID, "action_horizon": 32,
                "replan_steps": self.args.replan_steps, "denoise_steps": self.args.denoise_steps,
                "seed": self.args.seed, "text_encoder": "cache", "decode_video": False,
                "rtc_prefix": True, "max_prefix_steps": self.max_prefix, "image_crop": self.image_crop,
                "relative_dims": []}  # fmt: skip

    def context(self, instruction):
        """The prompt's T5 rows exactly as training read them (phoenix/dataset.py: PROMPT, text_index.json)."""
        from phoenix.dataset import PROMPT

        prompt = PROMPT.format(task=instruction)
        if prompt not in self.contexts:
            descriptor = self.text["prompts"].get(hashlib.sha256(prompt.encode()).hexdigest())
            if descriptor is None:
                raise KeyError(f"prompt {instruction!r} is not in {self.text_index} (only the trained prompts)")
            path = (self.text_index.parent / descriptor["path"]).resolve()
            if hashlib.sha256(path.read_bytes()).hexdigest() != descriptor["sha256"]:
                raise ValueError(f"T5 cache drift: {path}")
            pack = torch.load(path, map_location="cpu", weights_only=True)
            context, mask = pack["context"], pack["context_mask"]
            if (
                pack["prompt"] != prompt
                or context.shape != (128, 4096)
                or mask.shape != (128,)
                or mask.dtype != torch.bool
            ):
                raise ValueError(f"invalid T5 cache row {path}")
            self.contexts[prompt] = (context[None].to(self.device, self.model.torch_dtype), mask[None].to(self.device))
        return self.contexts[prompt]

    @torch.no_grad()
    def act(self, obs, instruction, *, full=False, prefix=None, seed=None, pad_cameras=()):
        from wam.inference.frame_joint import infer_frame_joint_window
        from wam.inference.online_action_policy import RTCActionPrefixCondition

        from phoenix.config import JOINT_SLOTS

        m = self.model
        state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError(f"expected finite 14-D state, got {state.shape}")
        prefix = np.zeros((0, 14), np.float32) if prefix is None else np.asarray(prefix, np.float32).reshape(-1, 14)
        if len(prefix) > self.max_prefix or not np.isfinite(prefix).all():
            raise ValueError(f"prefix: at most {self.max_prefix} finite steps (trained maximum), got {len(prefix)}")
        if set(pad_cameras or ()) - set(CAMERAS) or set(CAMERAS) <= set(pad_cameras or ()):
            raise ValueError(f"pad_cameras must name some of {CAMERAS}, not all, got {pad_cameras}")
        context, mask = self.context(instruction)
        seed = self.args.seed if seed is None else int(seed)

        views = []
        for key in CAMERAS:  # upstream Model.predict
            image = np.asarray(obs[key])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"expected uint8 HWC RGB for {key}")
            if self.image_crop == "center_square":
                image = center_square(image)
            image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None].float() / 255
            views.append(
                F.interpolate(image, size=(256, 256), mode="bicubic", align_corners=False).clamp(0, 1)[0] * 2 - 1
            )
        rgb = torch.stack(views)[None].to(device=m.device, dtype=m.torch_dtype)
        proprio = torch.zeros((1, 80), device=m.device, dtype=torch.float32)
        proprio[:, JOINT_SLOTS] = torch.as_tensor(self.normalizer.normalize(state, "state"), device=m.device)
        pad = torch.ones((1, 80), dtype=torch.bool, device=m.device)
        pad[:, JOINT_SLOTS] = False
        condition = normalized_prefix = None
        if len(prefix):
            target = torch.zeros((1, 32, 80), device=m.device, dtype=m.torch_dtype)
            target[:, : len(prefix), JOINT_SLOTS] = torch.as_tensor(
                self.normalizer.normalize(prefix, "action"), device=m.device, dtype=m.torch_dtype
            )
            normalized_prefix = target[:, : len(prefix)].float().cpu()
            condition = RTCActionPrefixCondition(target, len(prefix))
        camera_is_pad = torch.tensor([[key in (pad_cameras or ()) for key in CAMERAS]], device=m.device)
        result = infer_frame_joint_window(
            m, prompt=None, input_image=rgb, num_frames=9, action_horizon=32,
            proprio=proprio, context=context, context_mask=mask, camera_is_pad=camera_is_pad,
            action_dim_is_pad=pad, proprio_dim_is_pad=pad,
            embodiment_ids=torch.tensor([EMBODIMENT_ID], dtype=torch.long, device=m.device),
            num_inference_steps=self.args.denoise_steps, seed=None if seed < 0 else seed, decode_video=False,
            rtc_prefix_condition=condition,
        )  # fmt: skip
        if normalized_prefix is not None and not torch.equal(result["action"][: len(prefix)], normalized_prefix[0]):
            raise ValueError("the denoiser changed the normalized prefix")
        actions = result["action"].float().cpu().numpy()[:, JOINT_SLOTS]
        physical = np.ascontiguousarray(self.normalizer.decode(actions), dtype=np.float32)
        physical[: len(prefix)] = prefix  # the exact float32 targets, after the check above (as upstream)
        if physical.shape != (32, 14) or not np.isfinite(physical).all():
            raise ValueError("invalid action chunk")
        self.calls += 1
        return physical if full else physical[: self.args.replan_steps]


def main():
    args = parse_args()
    # everything the model libraries print goes to stderr; stdout carries only the JSON replies
    stdout = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    runtime = Runtime(args)
    print(f"[phoenix-rtc] ready: {json.dumps(runtime.info())}", file=sys.stderr, flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            if request["cmd"] == "info":
                reply = {"ok": True, "info": runtime.info()}
            elif request["cmd"] == "act":
                with np.load(request["obs_path"]) as data:
                    obs = {k: data[k] for k in data.files}
                started = time.perf_counter()
                actions = runtime.act(obs, request["prompt"], full=bool(request.get("full")),
                                      prefix=request.get("prefix"), seed=request.get("seed"),
                                      pad_cameras=request.get("pad_cameras") or ())  # fmt: skip
                reply = {"ok": True, "actions": actions.tolist(), "infer_s": time.perf_counter() - started}
            elif request["cmd"] == "reset":
                reply = {"ok": True}
            elif request["cmd"] == "set":
                if "denoise_steps" in request:
                    steps = int(request["denoise_steps"])
                    if not 1 <= steps <= 50:
                        raise ValueError(f"denoise_steps must be 1..50, got {steps}")
                    runtime.args.denoise_steps = steps
                    print(f"[phoenix-rtc] denoising steps: {steps}", file=sys.stderr, flush=True)
                if "replan_steps" in request:
                    steps = int(request["replan_steps"])
                    if not 1 <= steps <= 32:
                        raise ValueError(f"replan_steps must be 1..32 (the chunk), got {steps}")
                    runtime.args.replan_steps = steps
                    print(f"[phoenix-rtc] steps per chunk: {steps}", file=sys.stderr, flush=True)
                if "seed" in request:  # every chunk's noise from the next chunk on
                    runtime.args.seed = int(request["seed"])  # negative: fresh noise every chunk
                    print(f"[phoenix-rtc] noise seed: {runtime.args.seed}", file=sys.stderr, flush=True)
                reply = {"ok": True, "info": runtime.info()}
            else:
                raise ValueError(f"unknown cmd {request['cmd']!r}")
        except Exception as exc:  # report and keep serving; the adapter raises it
            import traceback

            traceback.print_exc()
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        stdout.write(json.dumps(reply) + "\n")
    with contextlib.suppress(Exception):
        stdout.close()


if __name__ == "__main__":
    main()
