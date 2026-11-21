#!/usr/bin/env python
"""Phoenix-MWA inference worker for policy/phoenix/deploy_policy.py: one JSON request per line on stdin.

Runs under the Phoenix venv (WS_ROOT/venvs/phoenix) with the submodule policy/phoenix/Phoenix-MWA on sys.path.
Model build, strict restore and one inference follow upstream's eval/robotwin/joint96_robot37_policy.py
(build_runtime, Robot37Joint96Session._infer_action_chunk) with three differences: the run's prompt slot (18, from the
run's registry), the run's normalization stats, and the prompt's T5 features read from the run's text cache (the same
tools/text_cache.py rows training read; --text-encoder gpu encodes uncached prompts online instead).

    {"cmd": "info"}                                   -> {"ok": true, "info": {...}}
    {"cmd": "act", "obs_path": npz, "prompt": str}    -> {"ok": true, "actions": [[14 floats] x replan_steps]}
        + "full": true                                -> all 32 steps of the chunk (deploy_policy's phoenix_async)
        + "prefix": [[14 floats] x d]                 -> Training RTC: the chunk continues these committed steps
                                                         (refused here: post-train checkpoints never learnt it)
        + "inpaint": [[14 floats] x k]                -> inpainting at inference (any checkpoint): the chunk's first k
                                                         steps held to these (the previous chunk's steps not executed),
          "inpaint_steps": h, "inpaint_soft": bool       the first h exactly, the rest by RTC's soft mask or not at all,
          "inpaint_soft_steps": n                        that over the next n only (absent / null: all k)
        + "rtc": {"prev": [[14 floats] x k],          -> RTC at inference (any checkpoint; RtcGuidance): the chunk is
                  "delay": d, "beta": 5.0}               guided to the running chunk's k targets from this observation
                                                         on, the first d (executed while this one is computed) at
                                                         weight 1, the rest by RTC's soft mask (not with "inpaint")
        + "seed": n                                   -> this chunk's noise seed instead of --seed (negative: fresh)
        + "pad_cameras": ["cam_high"]                 -> those views encoded as missing (camera_is_pad)
    {"cmd": "reset"}                                  -> {"ok": true}
    {"cmd": "set", "denoise_steps": n}                -> {"ok": true, "info": {...}}  (from the next chunk on)
    {"cmd": "set", "replan_steps": n}                 -> likewise: how many steps of each chunk "act" returns
    {"cmd": "set", "seed": n}                         -> likewise: every chunk's noise seed (negative: fresh noise)

The npz holds cam_high / cam_left_wrist / cam_right_wrist (uint8 HWC RGB) and state (14 joints + grippers, the
dataset's observation.state units). Actions are absolute joint targets with gripper opening fractions (0 closed,
1 open), as trained; the adapter scales the grippers. A checkpoint trained with relative joints
(configs/task/fmc3_posttrain_relative.yaml) predicts a[t+k] - s[t] on the 12 arm joints; this worker adds s[t] back,
so its replies are absolute as well ("relative_dims" in info).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time

POLICY_ROOT = Path(__file__).resolve().parent
UPSTREAM = Path(os.environ.get("PHOENIX_ROOT", POLICY_ROOT / "Phoenix-MWA"))
# this directory: robosyn_transforms, which the configs of relative-joint checkpoints name
sys.path[:0] = [str(UPSTREAM / "src"), str(UPSTREAM), str(POLICY_ROOT)]

import numpy as np  # noqa: E402
import torch  # noqa: E402

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
VALID_DIMS = [*range(6), 16, *range(40, 46), 56]
EMBODIMENT_ID = 18


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="training output (config.yaml, dataset_stats.json, checkpoints/weights/) or a copy of it")  # fmt: skip
    p.add_argument("--step", type=int, default=0, help="weights slot of this step (0: the latest)")
    p.add_argument("--registry", type=Path, help="default: <run-dir>/embodiment_registry.json, else <run-dir>/..")
    p.add_argument("--text-cache", type=Path, help="default: <run-dir>/text_cache, else <run-dir>/../text_cache")
    p.add_argument("--encoders", type=Path, help="dir with csvae.pt and dino110k.pt (default: the paths in config)")
    p.add_argument("--wan-base", help="DIFFSYNTH_MODEL_BASE_PATH: Wan2.2 VAE + tokenizer (the DiT is not loaded)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--replan-steps", type=int, default=10)
    p.add_argument("--denoise-steps", type=int, default=5, help="of both the future video and the actions")
    p.add_argument("--seed", type=int, default=0, help="noise seed of every chunk (upstream: 0); negative: fresh noise")
    p.add_argument("--text-encoder", choices=("cache", "gpu"), default="cache")
    p.add_argument("--frame-cache", action="store_true",
                   help="the current frame's trunk tokens computed once per inference, not per denoising step (FrameCache)")  # fmt: skip
    p.add_argument("--compile", choices=("off", "default"), default="off",
                   help="torch.compile around each MoT block's two halves and the DINO backbone, repeated frames "
                        "encoded once (compile_trunk, compile_encoder): fewer, fused small operators")  # fmt: skip
    p.add_argument(
        "--decode-video",
        action="store_true",
        help="also decode the predicted future frames (upstream's default; deployment never reads them)",
    )
    return p.parse_args(argv)


def beside(run_dir: Path, name: str, given: Path | None) -> Path:
    if given is not None:
        return given
    for root in (run_dir, run_dir.parent):
        if (root / name).exists():
            return root / name
    raise FileNotFoundError(f"no {name} in {run_dir} or its parent")


def relative_dims(processor) -> list[int]:
    """Raw action dims the checkpoint predicts relative to s[t] (robosyn_transforms.RelativeJointTransform)."""
    transforms = getattr(processor, "action_state_transforms", None) or []
    return [d for t in transforms if type(t).__name__ == "RelativeJointTransform" for d in t.dims]


def inpaint_weights(
    rows: int, hard: int, *, soft: bool, soft_steps: int | None = None, horizon: int = 32
) -> np.ndarray:
    """Per chunk index, how strongly inpainting holds it to a known step: 1 for the first ``hard`` of the ``rows`` known
    ones; with ``soft`` the next ``soft_steps`` (None: the rest of the known ones) decay as RTC's soft mask (Black et
    al. 2025, eq. 5: c (e^c - 1) / (e - 1), c = (end - i) / (end - hard + 1)), else 0; 0 past them."""
    rows = min(int(rows), horizon)
    hard = min(int(hard), rows)
    end = rows if soft_steps is None else min(rows, hard + max(0, int(soft_steps)))
    weights = np.zeros(horizon, dtype=np.float32)
    weights[:hard] = 1.0
    if soft and end > hard:
        c = (end - np.arange(hard, end)) / (end - hard + 1)
        weights[hard:end] = c * np.expm1(c) / np.expm1(1.0)
    return weights


class InpaintScheduler:
    """One inference's action scheduler, with the chunk's first steps held to known ones (training-free inpainting).

    An Euler step x + v * (s' - s) of upstream's flow (x_s = (1 - s) x0 + s noise, v = noise - x0) equals
    (1 - s') x0_hat + s' noise_hat, with x0_hat = x - s v and noise_hat = x + (1 - s) v. The known steps replace x0_hat
    by their weight w (1: exactly), so the next model call sees them at noise level s' and the last step (s' = 0)
    returns them; w = 0 leaves upstream's step bit for bit. Everything else is delegated to the wrapped scheduler.
    """

    def __init__(self, inner, known: torch.Tensor, weights: torch.Tensor):
        self.inner = inner
        self.known = known  # [1, horizon, 80] normalized, float32
        self.weights = weights  # [1, horizon, 1], float32
        self.sigmas = self.next_sigmas = None
        self.index = 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def build_inference_schedule(self, num_inference_steps, device, dtype, shift_override=None):
        timesteps, deltas = self.inner.build_inference_schedule(
            num_inference_steps=num_inference_steps, device=device, dtype=torch.float32, shift_override=shift_override
        )
        self.sigmas = timesteps / float(self.inner.num_train_timesteps)
        self.next_sigmas = self.sigmas + deltas
        self.index = 0
        return self.inner.build_inference_schedule(
            num_inference_steps=num_inference_steps, device=device, dtype=dtype, shift_override=shift_override
        )

    def step(self, model_output, delta, sample):
        stepped = self.inner.step(model_output, delta, sample)
        sigma, sigma_next = self.sigmas[self.index], self.next_sigmas[self.index]
        self.index += 1
        x0 = sample.float() - sigma * model_output.float()
        correction = (1.0 - sigma_next) * self.weights * (self.known - x0)
        return (stepped.float() + correction).to(sample.dtype)


def compile_trunk(model) -> int:
    """torch.compile around the two per-layer halves of upstream's MoT, for inference (upstream's own
    mot_compile_mode only applies while training: evaluation "uses the eager attention path").

    One inference is about 64 000 operator calls, and the trunk's matrix products are only about half of its time (on
    an RTX PRO 6000: 5 denoising steps x 30 layers at 1.5 ms a layer against 0.75 ms of products); the rest is small
    elementwise kernels (float32 RMS norms, modulation, gates, casts) and their launches. Inductor fuses those. Each
    block gets its own compiled frame around upstream's functions, which are traced unchanged
    (_build_expert_attention_io: modulation, norm, q / k / v, RoPE; _apply_expert_post_block: attention output,
    cross-attention, feed-forward), by a code object per block as upstream's training compile makes one per layer;
    the mixed FlexAttention between them stays as it is. The fused kernels round differently from the eager ones, so a
    chunk differs from the eager one at the level of bf16 rounding carried through the denoising (measured on two tasks'
    real frames: arm joints by 0.006-0.007 rad at the median chunk, 0.02 at most). The first inference compiles (14 s).
    Returns the number of blocks; restore_trunk puts the eager functions back."""
    mot = model.mot
    build = type(mot)._build_expert_attention_io  # noqa: SLF001
    post_block = type(mot)._apply_expert_post_block  # noqa: SLF001
    post_block = getattr(post_block, "__wrapped__", post_block)  # without upstream's timing decorator
    compiled_pre, compiled_post = {}, {}
    for name in mot.expert_order:
        expert = mot.mixtures[name]
        for index, block in enumerate(expert.blocks):

            def pre_fn(x, freqs, t_mod, _expert=expert, _block=block):
                return build(mot, _expert, _block, x, freqs, t_mod)

            def post_fn(residual_x, mixed_attn_out, gate_msa, shift_mlp, scale_mlp, gate_mlp, context, mask,
                        _block=block):  # fmt: skip
                payload = None if context is None else {"context": context, "mask": mask}
                return post_block(_block, residual_x, mixed_attn_out, gate_msa, shift_mlp, scale_mlp, gate_mlp, payload)

            pre_fn.__code__ = pre_fn.__code__.replace(co_name=f"phoenix_pre_{name}_{index}")
            post_fn.__code__ = post_fn.__code__.replace(co_name=f"phoenix_post_{name}_{index}")
            compiled_pre[id(block)] = torch.compile(pre_fn, dynamic=False, fullgraph=False)
            compiled_post[id(block)] = torch.compile(post_fn, dynamic=False, fullgraph=False)

    def build_dispatch(expert, block, x, freqs, t_mod):
        return compiled_pre[id(block)](x, freqs, t_mod)

    def post_dispatch(block, residual_x, mixed_attn_out, gate_msa, shift_mlp, scale_mlp, gate_mlp, context_payload):
        context = None if context_payload is None else context_payload.get("context")
        mask = None if context_payload is None else context_payload.get("mask")
        return compiled_post[id(block)](residual_x, mixed_attn_out, gate_msa, shift_mlp, scale_mlp, gate_mlp, context, mask)

    mot._build_expert_attention_io, mot._apply_expert_post_block = build_dispatch, post_dispatch  # noqa: SLF001
    return len(compiled_pre)


def compile_encoder(model) -> str:
    """The current frame's DINO backbone compiled, and run once per view instead of nine times.

    Upstream's online path repeats the observed frame nine times before DINO + CSVAE (current_visual_adapter's
    repeat9, what the checkpoint was trained with) and the student's backbone then treats the nine copies as nine
    images: 27 ViT forwards an inference for 3 different images. The copies give the same features, so they are
    computed once and repeated; anything else (frames that differ, a patch mask) goes through as before. Like
    compile_trunk, same computation, different rounding (the chunk's arm joints 0.006-0.007 rad at the median).
    RTX PRO 6000: the encoder 63 -> about 35 ms an inference."""
    try:
        student = model.joint_visual_encoder.csvae_encoder.dino.student
        features, backbone = student._dino_features, student.dino  # noqa: SLF001
    except AttributeError as exc:
        return f"left as it is ({exc})"
    backbone.forward_features = torch.compile(backbone.forward_features, dynamic=False, fullgraph=False)

    def once(frames, patch_mask, require_grad=True):
        count = int(frames.shape[1])
        if patch_mask is not None or count < 2 or not bool((frames[:, 1:] == frames[:, :1]).all()):
            return features(frames, patch_mask, require_grad)
        patches, cls = features(frames[:, :1], None, require_grad)
        return (patches.expand(-1, count, *patches.shape[2:]).contiguous(),
                cls.expand(-1, count, *cls.shape[2:]).contiguous())  # fmt: skip

    student._dino_features = once  # noqa: SLF001
    return "DINO backbone compiled, repeated frames encoded once"


def restore_trunk(model) -> dict:
    """The eager functions back in place of compile_trunk's; returns what it took off (for put_trunk)."""
    names = ("_build_expert_attention_io", "_apply_expert_post_block")
    taken = {name: vars(model.mot)[name] for name in names if name in vars(model.mot)}
    for name in taken:
        delattr(model.mot, name)
    return taken


class FrameCache:
    """The current frame's part of the trunk computed once per inference instead of once per denoising step.

    In joint96 the video sequence is [current frame, noisy future] and upstream's mask (build_joint96_attention_mask)
    lets the current frame's tokens attend only to themselves; their timestep is 0 at every step. So their hidden
    states, and the keys and values the other tokens read from them, are the same in all denoising steps of an
    inference, a third of the video tokens. Installed, the first MoT call of an inference runs as upstream's and keeps
    each layer's keys and values of those tokens; a later call whose current-frame inputs are bit for bit the kept
    ones (tokens, time modulation, RoPE, text context) runs upstream's own forward on the future and action tokens
    only, and the mixed attention is given the kept keys and values in front of theirs, under the same rows of
    upstream's mask. Any other call -- a new frame, a padded camera, other shapes -- runs in full and renews the
    cache. The kernels see other shapes, so the result differs from the full forward by rounding, as with
    compile_trunk. Upstream has this idea for a whole static video (kv_cache.py), not for part of the video expert."""

    def __init__(self, model):
        self.model, self.mot = model, model.mot
        self.ref = self.keys = self.values = self.current_out = None
        self.layer, self.capturing, self.cached_step, self.n = 0, False, False, 0
        self.full_mask = self.block = self.block_shape = None
        self.counts = {"full": 0, "cached": 0}

    def install(self) -> "FrameCache":
        mot = self.mot
        self.forward, self.build, self.mixed = mot.forward, mot._build_expert_attention_io, mot._mixed_attention  # noqa: SLF001
        mot.forward, mot._build_expert_attention_io, mot._mixed_attention = self._forward, self._build, self._mixed  # noqa: SLF001
        mot._phoenix_frame_cache = self  # noqa: SLF001
        return self

    def uninstall(self) -> None:
        mot = self.mot
        if getattr(mot, "_phoenix_frame_cache", None) is self:
            mot.forward, mot._build_expert_attention_io, mot._mixed_attention = self.forward, self.build, self.mixed  # noqa: SLF001
            for name in ("forward", "_build_expert_attention_io", "_mixed_attention"):
                if getattr(type(mot), name, None) is getattr(getattr(mot, name), "__func__", None):
                    delattr(mot, name)  # the class's own method again
            del mot._phoenix_frame_cache  # noqa: SLF001
        self.ref = self.keys = self.values = self.current_out = None

    @staticmethod
    def current_tokens(mask: torch.Tensor, video_len: int) -> int:
        """How many leading video tokens attend only to themselves and are attended in full (0: no such block)."""
        if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
            return 0
        row = mask[0]
        n = int(row.long().sum().item())
        if not 0 < n < video_len or not bool(row[:n].all().item()):
            return 0
        if not bool(mask[:n, :n].all().item()) or bool(mask[:n, n:].any().item()):
            return 0
        return n

    @staticmethod
    def _split(tensor, length: int, n: int, first: bool):
        """``tensor`` cut along its axis of the video sequence's ``length``: the first ``n`` or the rest. That axis is
        axis 1 ([B, S, ...]) when it has the length, else the only axis that has. A tensor without such an axis is the
        same for every token and comes back whole; several candidates elsewhere: ValueError."""
        if tensor is None:
            return None
        axes = [axis for axis, size in enumerate(tensor.shape) if size == length]
        if not axes:
            return tensor
        if 1 in axes:
            axes = [1]
        if len(axes) > 1:
            raise ValueError(f"which axis of {tuple(tensor.shape)} is the video sequence of length {length}?")
        return tensor.narrow(axes[0], 0, n) if first else tensor.narrow(axes[0], n, length - n)

    def _parts(self, kwargs, n, first: bool):
        """The video expert's inputs for its first ``n`` tokens, or for the others: tokens, time modulation, RoPE,
        text context, context mask."""
        length = int(kwargs["embeds_all"]["video"].shape[1])
        context = kwargs["context_all"].get("video") or {}
        return (self._split(kwargs["embeds_all"]["video"], length, n, first),
                self._split(kwargs["t_mod_all"]["video"], length, n, first),
                self._split(kwargs["freqs_all"]["video"], length, n, first),
                context.get("context"), self._split(context.get("mask"), length, n, first))  # fmt: skip

    def _same(self, inputs, mask, valid) -> bool:
        """The kept inputs and mask are these, bit for bit, and no token is masked out (one GPU round trip)."""
        if self.ref is None or self.full_mask.shape != mask.shape:
            return False
        checks = [(self.full_mask == mask).all(), *(m.all() for m in valid)]
        for kept, now in zip(self.ref, inputs, strict=True):
            if (kept is None) != (now is None):
                return False
            if kept is not None:
                if kept.shape != now.shape or kept.dtype != now.dtype:
                    return False
                checks.append((kept == now).all())
        return bool(torch.stack(checks).all().item())

    def _forward(self, *args, **kwargs):
        if args or set(kwargs) - {"embeds_all", "attention_mask", "freqs_all", "context_all", "t_mod_all",
                                  "key_masks_all", "query_masks_all"} or torch.is_grad_enabled():  # fmt: skip
            return self.forward(*args, **kwargs)  # captures, a profiler, a backward pass: upstream's path as it is
        mask, video = kwargs["attention_mask"], kwargs["embeds_all"]["video"]
        valid = [m for masks in (kwargs.get("key_masks_all"), kwargs.get("query_masks_all")) if masks
                 for m in masks.values() if m is not None]  # fmt: skip
        if mask is None or mask.ndim != 2 or list(self.mot.expert_order) != ["video", "action"]:
            self.ref = None
            return self.forward(**kwargs)
        try:
            if self.ref is not None and self._same(self._parts(kwargs, self.n, first=True), mask, valid):
                return self._cached(kwargs, self.n)
            n = self.current_tokens(mask, int(video.shape[1]))
            if n == 0 or not all(bool(m.all().item()) for m in valid):
                raise ValueError("no current-frame block, or masked tokens")
            inputs = self._parts(kwargs, n, first=True)
        except ValueError:
            self.ref = None
            return self.forward(**kwargs)
        total, batch = int(mask.shape[0]), int(video.shape[0])
        if self.block is None or self.block_shape != (batch, total, n) or not torch.equal(self.full_mask, mask):
            # the live queries' rows over every key, the kept ones first: one block mask for all layers and steps
            self.block = self.mot._build_flex_block_mask(  # noqa: SLF001
                attention_mask=mask[n:], batch_size=batch, query_len=total - n, key_len=total, device=video.device)  # fmt: skip
            self.block_shape = (batch, total, n)
        self.ref, self.n, self.full_mask = tuple(None if t is None else t.clone() for t in inputs), n, mask.clone()
        self.keys, self.values, self.capturing = [], [], True
        try:
            out = self.forward(**kwargs)
        finally:
            self.capturing = False
        if len(self.keys) != self.mot.num_layers:
            self.ref = None  # upstream's forward did not go through the hooks as expected: no cache
        else:
            self.current_out = out["video"][:, :n].clone()
        self.counts["full"] += 1
        return out

    def _cached(self, kwargs, n):
        tokens, t_mod, freqs, _context, context_mask = self._parts(kwargs, n, first=False)
        context = kwargs["context_all"].get("video")
        live = dict(kwargs)
        live["embeds_all"] = {**kwargs["embeds_all"], "video": tokens}
        live["freqs_all"] = {**kwargs["freqs_all"], "video": freqs}
        live["t_mod_all"] = {**kwargs["t_mod_all"], "video": t_mod}
        if context is not None:
            live["context_all"] = {**kwargs["context_all"], "video": {**context, "mask": context_mask}}
        for name in ("key_masks_all", "query_masks_all"):
            if kwargs.get(name) and kwargs[name].get("video") is not None:
                live[name] = {**kwargs[name], "video": kwargs[name]["video"][:, n:]}
        live["attention_mask"] = self.full_mask[n:, n:]  # square, for upstream's own checks; _mixed uses the full rows
        self.layer, self.cached_step = 0, True
        try:
            out = self.forward(**live)
        finally:
            self.cached_step = False
        self.counts["cached"] += 1
        return {**out, "video": torch.cat([self.current_out.to(out["video"].dtype), out["video"]], dim=1)}

    def _build(self, expert, block, x, freqs, t_mod):
        result = self.build(expert=expert, block=block, x=x, freqs=freqs, t_mod=t_mod)
        if self.capturing and expert is self.mot.mixtures["video"]:
            self.keys.append(result[1][:, : self.n].clone())
            self.values.append(result[2][:, : self.n].clone())
        return result

    def _mixed(self, q_cat, k_cat, v_cat, attention_mask, **kwargs):
        if not self.cached_step:
            return self.mixed(q_cat=q_cat, k_cat=k_cat, v_cat=v_cat, attention_mask=attention_mask, **kwargs)
        layer = self.layer
        self.layer += 1
        keys = torch.cat([self.keys[layer], k_cat], dim=1)
        values = torch.cat([self.values[layer], v_cat], dim=1)
        return self.mixed(q_cat=q_cat, k_cat=keys, v_cat=values, attention_mask=self.full_mask[self.n :],
                          flex_block_mask=self.block, flex_key_mask=None, query_mask=None, force_sdpa=False)  # fmt: skip


class RtcGuidance:
    """Inference-time RTC (Black et al. 2025, arXiv 2506.07339) around upstream's joint denoising loop, one inference.

    Upstream has RTC's guidance (online_action_policy._apply_rtc_guidance_to_action_velocity: PI's clipped
    pseudoinverse VJP correction in Phoenix's noise-minus-clean convention, with get_rtc_prefix_weights' soft mask),
    but only inside its static-video-cache policy, which refuses joint96 checkpoints; infer_frame_joint_window, the
    loop joint96 checkpoints need, has no hook for it. Entered, this puts upstream's function, unchanged, around that loop's own
    model call: at every denoising step the joint forward runs once, with gradients, as its ``predict_velocity``
    (the loop's action pre_dit with the noisy actions as the differentiated input -> MoT on the video tokens the
    loop built -> action post_dit); the loop then gets the same video tokens and, from post_dit, the corrected action
    velocity. One forward and one backward pass per step instead of one forward. beta 0 leaves the velocity as it is.

    The MoT's FlexAttention runs with the checkpoint's own training kernel options meanwhile
    (mot_flex_training_kernel_options, which upstream selects by ``self.training``): the eval options have no backward
    block sizes, and the backward kernel does not fit the GPU's shared memory with the defaults. The first guided
    inference of a process compiles that kernel (the adapter warms it up at start).
    """

    def __init__(self, model, prev: torch.Tensor, delay: int, rows: int, action_pad: torch.Tensor, beta: float = 5.0,
                 schedule: str = "exp"):  # fmt: skip
        from wam.inference.online_action_policy import RTCActionGuidance
        from wam.inference.online_action_policy import _prepare_rtc_action_guidance

        self.model, self.action_pad = model, action_pad
        guidance = RTCActionGuidance(
            prev_action_chunk=prev, inference_delay=min(int(delay), int(rows)), prefix_attention_horizon=int(rows),
            max_guidance_weight=float(beta), schedule=schedule,
        )  # fmt: skip
        self.prepared = _prepare_rtc_action_guidance(
            guidance, action_horizon=int(prev.shape[1]), action_dim=int(model.action_expert.action_dim),
            action_dim_is_pad=action_pad, device=model.device, dtype=model.torch_dtype,
        )  # fmt: skip
        self.pre_kwargs = self.velocity = None
        self.steps = 0

    def __enter__(self):
        expert, mot = self.model.action_expert, self.model.mot
        self.pre_dit, self.post_dit, self.mot_forward = expert.pre_dit, expert.post_dit, mot.forward
        # what was patched over them already (FrameCache's forward) comes back as it was
        self.patched = [(owner, name, vars(owner).get(name))
                        for owner, name in ((expert, "pre_dit"), (expert, "post_dit"), (mot, "forward"))]  # fmt: skip
        expert.pre_dit, expert.post_dit, mot.forward = self._pre_dit, self._post_dit, self._mot_forward
        self.compiled = restore_trunk(self.model)  # --compile: the guided forward and its backward run eagerly
        self.flex_attention, options = mot._flex_attention, getattr(mot, "flex_training_kernel_options", None)  # noqa: SLF001
        if options:

            def flex_attention(q, k, v, block_mask, _eval_options, *rest):
                return self.flex_attention(q, k, v, block_mask, options, *rest)

            mot._flex_attention = flex_attention  # noqa: SLF001
        return self

    def __exit__(self, *exc):
        mot = self.model.mot
        for owner, name, before in self.patched:
            if before is None:
                delattr(owner, name)  # the instance attribute: the class method is back
            else:
                setattr(owner, name, before)
        mot._flex_attention = self.flex_attention  # noqa: SLF001
        for name, function in self.compiled.items():
            setattr(mot, name, function)
        return False

    def _pre_dit(self, **kwargs):
        self.pre_kwargs = kwargs
        return self.pre_dit(**kwargs)

    def _mot_forward(self, *args, **kwargs):
        from wam.inference.online_action_policy import _apply_rtc_guidance_to_action_velocity

        if args or self.pre_kwargs is None:
            raise RuntimeError("upstream's joint loop no longer calls action pre_dit, then mot(**kwargs)")
        out, pre_kwargs = {}, self.pre_kwargs
        self.pre_kwargs = None

        def predict_velocity(action_input):
            pre = self.pre_dit(**{**pre_kwargs, "action_tokens": action_input})
            out.update(self.mot_forward(**{**kwargs, "embeds_all": {**kwargs["embeds_all"], "action": pre["tokens"]}}))
            return self.post_dit(out["action"], pre)

        self.velocity = _apply_rtc_guidance_to_action_velocity(
            latents_action=pre_kwargs["action_tokens"], timestep_action=pre_kwargs["timestep"],
            num_train_timesteps=int(self.model.infer_action_scheduler.num_train_timesteps),
            predict_velocity=predict_velocity, guidance=self.prepared, action_dim_is_pad=self.action_pad,
        )  # fmt: skip
        self.steps += 1
        return {name: tokens.detach() for name, tokens in out.items()}

    def _post_dit(self, tokens, pre_state):
        if self.velocity is None:
            raise RuntimeError("upstream's joint loop no longer calls action post_dit once per mot call")
        velocity, self.velocity = self.velocity, None
        return velocity.to(tokens.dtype)


def weights_slot(run_dir: Path, step: int) -> tuple[Path, int]:
    manifest = json.loads((run_dir / "checkpoints/weights/weights_manifest.json").read_text())
    slots: dict[int, list[Path]] = {}
    for slot in manifest["slots"]:  # the final step can be saved twice (last interval + train end)
        slots.setdefault(int(slot["step"]), []).append(run_dir / "checkpoints/weights" / Path(slot["path"]).name)
    step = step or max(slots)
    present = [path for path in slots.get(step, []) if path.is_file()]
    if not present:
        raise ValueError(f"no weights file of step {step} here; saved steps: {sorted(slots)}")
    return present[0], step


class Runtime:
    def __init__(self, args):
        from eval.robotwin.current_visual_adapter import configure_current_visual
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from wam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
        from wam.model.modules.codecs.dino_csvae.provenance import bundled_config
        from wam.posttrain.guard import install_import_guard
        from wam.training.checkpoint import _load_joint96_checkpoint
        from wam.utils.config_resolvers import register_default_resolvers

        install_import_guard()
        register_default_resolvers()
        torch.set_num_threads(1)
        torch.manual_seed(max(args.seed, 0))  # the generator a negative --seed (fresh noise every chunk) draws from
        self.args = args
        self.device = torch.device(args.device)
        run_dir = args.run_dir.resolve()
        self.checkpoint, self.step = weights_slot(run_dir, args.step)
        self.stats = run_dir / "dataset_stats.json"
        self.registry = beside(run_dir, "embodiment_registry.json", args.registry)
        self.text_cache = beside(run_dir, "text_cache", args.text_cache)
        if args.wan_base:
            os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = args.wan_base
        os.environ["EMBODIMENT_REGISTRY"] = str(self.registry)

        cfg = OmegaConf.load(run_dir / "config.yaml")
        processor_cfg = cfg.data.train.processor
        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=False))
        codec = model_cfg.joint_visual_codec
        if args.encoders:
            codec.csvae_checkpoint_path = str(args.encoders / "csvae.pt")
            codec.dino_kwargs.checkpoint_path = str(args.encoders / "dino110k.pt")
        model_cfg.joint_visual_codec = bundled_config(OmegaConf.to_container(codec, resolve=True))
        model_cfg.action_dit_config.embodiment_registry_path = str(self.registry)
        # as upstream build_runtime: inference-only switches, never the architecture or noise schedules
        model_cfg.load_text_encoder = args.text_encoder == "gpu"
        model_cfg.mot_checkpoint_mixed_attn = False
        model_cfg.mot_compile_gradient_checkpointing = False
        model_cfg.mot_compile_mode = "off"
        model_cfg.video_dit_config.use_gradient_checkpointing = False
        model_cfg.action_dit_config.use_gradient_checkpointing = False

        self.processor = instantiate(processor_cfg).eval()
        self.processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(self.stats)))
        self.relative_dims = relative_dims(self.processor)
        for field in ("action", "state"):
            if (~self.static_pad(field)).nonzero().flatten().tolist() != VALID_DIMS:
                raise ValueError(f"canonical80 mapping mismatch: {field}")

        print(f"[phoenix] loading {self.checkpoint}", file=sys.stderr, flush=True)
        payload = torch.load(self.checkpoint, map_location="cpu", mmap=True, weights_only=True)
        contract = payload["joint96_metadata"].get("soft_prompt", {}).get("training_contract", {})
        if contract.get("embodiment_id") != EMBODIMENT_ID:
            raise ValueError(f"not a slot-{EMBODIMENT_ID} checkpoint: {contract}")
        self.model = self.build(instantiate, model_cfg)
        _load_joint96_checkpoint(self.model, payload)  # strict: every policy tensor
        self.model.eval().requires_grad_(requires_grad=False)
        self.visual = configure_current_visual(self.model, "auto")
        if args.compile != "off":
            print(f"[phoenix] --compile: {compile_trunk(self.model)} MoT blocks; encoder: "
                  f"{compile_encoder(self.model)}; compiled at the first inference", file=sys.stderr, flush=True)  # fmt: skip
        self.frame_cache = FrameCache(self.model).install() if args.frame_cache else None
        if self.frame_cache is not None:
            print("[phoenix] --frame-cache: the current frame's trunk tokens once per inference", file=sys.stderr, flush=True)
        self.action_pad = self.static_pad("action").unsqueeze(0).to(self.device)
        self.state_pad = self.static_pad("state").unsqueeze(0).to(self.device)
        self.contexts = {}
        self.calls = 0

    def build(self, instantiate, model_cfg):
        """upstream instantiate_checkpoint_model: skip only the Wan DiT initialization weights (restored strictly)."""
        import wam.model.factory as factory

        original = factory.load_wan22_ti2v_5b_components

        def checkpoint_only_components(**kwargs):
            kwargs["skip_dit_load_from_pretrain"] = True
            return original(**kwargs)

        factory.load_wan22_ti2v_5b_components = checkpoint_only_components
        try:
            return instantiate(model_cfg, model_dtype=torch.bfloat16, device=str(self.device))
        finally:
            factory.load_wan22_ti2v_5b_components = original

    def static_pad(self, field):
        merger = self.processor.action_state_merger
        mask = torch.ones(int(getattr(merger, f"{field}_target_dim")), dtype=torch.bool)
        for entry in getattr(merger, f"{field}_target_slices"):
            start, end = (int(v) for v in entry["target_slice"])
            mask[start:end] = False
        return mask

    def info(self):
        return {"checkpoint": str(self.checkpoint), "step": self.step, "stats": str(self.stats),
                "registry": str(self.registry), "embodiment_id": EMBODIMENT_ID, "action_horizon": 32,
                "replan_steps": self.args.replan_steps, "denoise_steps": self.args.denoise_steps,
                "seed": self.args.seed, "text_encoder": self.args.text_encoder, "current_visual": self.visual["mode"],
                "decode_video": self.args.decode_video, "rtc_prefix": False, "inpaint": True, "rtc_guidance": True,
                "compile": self.args.compile, "frame_cache": bool(self.args.frame_cache),
                "relative_dims": self.relative_dims}  # fmt: skip

    def rgb(self, obs):
        """upstream current_rgb: training processor transform, then resize/crop to 256 and [-1, 1] per view."""
        from wam.datasets.dataset_utils import CenterCrop
        from wam.datasets.dataset_utils import Normalize
        from wam.datasets.dataset_utils import ResizeSmallestSideAspectPreserving

        images = {}
        for meta in self.processor.shape_meta["images"]:
            image = np.asarray(obs[meta["key"]])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"expected uint8 HWC RGB for {meta['key']}")
            images[meta["key"]] = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0)
        views = self.processor._transform_images({"images": images})  # noqa: SLF001
        resize = ResizeSmallestSideAspectPreserving({"img_h": 256, "img_w": 256})
        crop = CenterCrop({"img_h": 256, "img_w": 256})
        norm = Normalize({"mean": 0.5, "std": 0.5})
        result = torch.stack([norm(crop(resize(v))).clamp(-1, 1)[0] for v in views]).unsqueeze(0)
        if result.shape != (1, 3, 3, 256, 256) or not torch.isfinite(result).all():
            raise ValueError("invalid current RGB geometry or values")
        return result.to(device=self.device, dtype=torch.bfloat16)

    def proprio(self, state):
        batch = {"state": {self.processor.shape_meta["state"][0]["key"]: torch.as_tensor(state).unsqueeze(0)}}
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        return self.processor.action_state_merger.forward(batch)["state"].to(device=self.device, dtype=torch.bfloat16)

    def to_model(self, actions, state) -> torch.Tensor:
        """Native absolute targets (k x 14, grippers as opening fractions) -> the normalized [1, 32, 80] chunk the model
        denoises, the inverse of act()'s decode (relative dims against this state); rows past k repeat the last one,
        which only a normalizer fitted per chunk step would read."""
        rows = np.array(actions, dtype=np.float32).reshape(-1, 14)[:32]
        if self.relative_dims:
            rows[:, self.relative_dims] -= state[self.relative_dims]
        rows = np.concatenate([rows, np.repeat(rows[-1:], 32 - len(rows), axis=0)])
        batch = {"action": {self.processor.shape_meta["action"][0]["key"]: torch.from_numpy(rows)},
                 "state": {self.processor.shape_meta["state"][0]["key"]: torch.as_tensor(state).unsqueeze(0)}}  # fmt: skip
        batch = self.processor.normalizer.forward(batch)  # per-sample shapes, as the dataset hands them
        normalized = self.processor.action_state_merger.forward(batch)["action"]
        if normalized.shape != (32, 80) or not torch.isfinite(normalized).all():
            raise ValueError("invalid normalized inpainting rows")
        return normalized.unsqueeze(0).to(device=self.device, dtype=torch.float32)

    def context(self, instruction):
        from wam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

        prompt = DEFAULT_PROMPT.format(task=instruction)
        if prompt not in self.contexts:
            path = self.text_cache / f"{hashlib.sha256(prompt.encode()).hexdigest()}.t5_len128.wan22ti2v5b.pt"
            if path.is_file():  # the training rows; zero padding, all keys visible (upstream official_text_padding)
                row = torch.load(path, map_location="cpu")
                context = row["context"].clone()
                context[~row["mask"].bool()] = 0.0
                context, mask = context.unsqueeze(0), torch.ones((1, 128), dtype=torch.bool)
            elif self.args.text_encoder == "gpu":
                context, mask = self.model.encode_prompt(prompt)
            else:
                raise KeyError(f"prompt {instruction!r} is not in {self.text_cache} (or start with --text-encoder gpu)")
            if context.shape != (1, 128, 4096) or not mask.bool().all():
                raise ValueError("T5 context differs from the training padding semantics")
            self.contexts[prompt] = (context.to(self.device, torch.bfloat16), mask.to(self.device))
        return self.contexts[prompt]

    @torch.no_grad()
    def act(self, obs, instruction, *, full=False, prefix=None, seed=None, pad_cameras=(), inpaint=None,
            inpaint_steps=4, inpaint_soft=False, inpaint_soft_steps=None, rtc=None):  # fmt: skip
        """``seed``: this chunk's noise seed instead of --seed (the adapter's stall reseed); negative = fresh noise.
        ``pad_cameras``: views encoded as missing (camera_is_pad, zero latents), e.g. ["cam_high"] after a stall.
        ``inpaint``: native targets the chunk's first steps are held to (InpaintScheduler, inpaint_weights).
        ``rtc``: {"prev": native targets of the running chunk from this observation on, "delay": d, "beta": 5.0}:
        the chunk is guided to them (RtcGuidance), the first d at weight 1, the rest by RTC's soft mask."""
        from wam.inference.frame_joint import infer_frame_joint_window

        if prefix is not None and len(prefix):
            # the post-train branch has no prefix condition at inference, and its joint96 training refuses RTC
            # prefixes: only a Training RTC checkpoint (upstream's prometheus branch) continues a committed prefix
            raise ValueError("this checkpoint was not trained with Training RTC: run it without phoenix_rtc_prefix")

        state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError(f"expected finite 14-D state, got {state.shape}")
        context, mask = self.context(instruction)
        seed = self.args.seed if seed is None else int(seed)
        views = [meta["key"] for meta in self.processor.shape_meta["images"]]
        if set(pad_cameras or ()) - set(views) or set(views) <= set(pad_cameras or ()):
            raise ValueError(f"pad_cameras must name some of {views}, not all, got {pad_cameras}")
        camera_is_pad = torch.tensor([[view in (pad_cameras or ()) for view in views]], device=self.device)
        scheduler = self.model.infer_action_scheduler
        guided = contextlib.nullcontext()
        if rtc is not None and len(rtc.get("prev") or ()):
            if inpaint is not None and len(inpaint):
                raise ValueError("one of rtc (guidance) and inpaint per chunk, not both")
            rows = min(len(rtc["prev"]), 32)
            guided = RtcGuidance(self.model, self.to_model(rtc["prev"], state), int(rtc.get("delay", 0)), rows,
                                 self.action_pad, beta=float(rtc.get("beta", 5.0)))  # fmt: skip
        if inpaint is not None and len(inpaint):
            weights = inpaint_weights(
                len(inpaint), inpaint_steps, soft=bool(inpaint_soft), soft_steps=inpaint_soft_steps
            )
            self.model.infer_action_scheduler = InpaintScheduler(
                scheduler, self.to_model(inpaint, state), torch.from_numpy(weights).to(self.device).view(1, -1, 1)
            )
        try:
            with guided:
                result = infer_frame_joint_window(
                    self.model, None, self.rgb(obs), num_frames=9, action_horizon=32,
                    proprio=self.proprio(state), context=context, context_mask=mask,
                    # upstream's eval seeds every chunk alike (0); None draws fresh noise from the global generator
                    num_inference_steps=self.args.denoise_steps, seed=None if seed < 0 else seed, rand_device="cpu", tiled=False,
                    video_layout="latent_horizontal", video_view_names="main|left|right",
                    camera_is_pad=camera_is_pad,
                    action_dim_is_pad=self.action_pad, proprio_dim_is_pad=self.state_pad,
                    embodiment_ids=torch.tensor([EMBODIMENT_ID], dtype=torch.long, device=self.device),
                    decode_video=self.args.decode_video,
                )  # fmt: skip
        finally:
            self.model.infer_action_scheduler = scheduler
        normalized = result["action"]
        if normalized.shape != (32, 80) or not torch.isfinite(normalized).all():
            raise ValueError("invalid normalized action chunk")
        batch = {"action": normalized.float().cpu().unsqueeze(0)}
        batch["state"] = batch["action"].new_zeros(batch["action"].shape)
        batch = self.processor.action_state_merger.backward(batch)
        batch = self.processor.normalizer.backward(batch)
        action = batch["action"][self.processor.shape_meta["action"][0]["key"]].numpy()[0].astype(np.float32)
        if action.shape != (32, 14) or not np.isfinite(action).all():
            raise ValueError("invalid native action chunk")
        if self.relative_dims:  # the decode above denormalizes but does not undo action_state_transforms
            action[:, self.relative_dims] += state[self.relative_dims]
        self.calls += 1
        return action if full else action[: self.args.replan_steps]


def main():
    args = parse_args()
    # everything the model libraries print goes to stderr; stdout carries only the JSON replies
    stdout = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    runtime = Runtime(args)
    print(f"[phoenix] ready: {json.dumps(runtime.info())}", file=sys.stderr, flush=True)
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
                                      pad_cameras=request.get("pad_cameras") or (), inpaint=request.get("inpaint"),
                                      inpaint_steps=int(request.get("inpaint_steps", 4)),
                                      inpaint_soft=bool(request.get("inpaint_soft")),
                                      inpaint_soft_steps=request.get("inpaint_soft_steps"),
                                      rtc=request.get("rtc"))  # fmt: skip
                reply = {"ok": True, "actions": actions.tolist(), "infer_s": time.perf_counter() - started}
            elif request["cmd"] == "reset":
                reply = {"ok": True}
            elif request["cmd"] == "set":
                if "denoise_steps" in request:
                    steps = int(request["denoise_steps"])
                    if not 1 <= steps <= 50:
                        raise ValueError(f"denoise_steps must be 1..50, got {steps}")
                    runtime.args.denoise_steps = steps  # read by every act: the next chunk on
                    print(f"[phoenix] denoising steps: {steps}", file=sys.stderr, flush=True)
                if "replan_steps" in request:
                    steps = int(request["replan_steps"])
                    if not 1 <= steps <= 32:
                        raise ValueError(f"replan_steps must be 1..32 (the chunk), got {steps}")
                    runtime.args.replan_steps = steps
                    print(f"[phoenix] steps per chunk: {steps}", file=sys.stderr, flush=True)
                if "seed" in request:  # every chunk's noise from the next chunk on
                    runtime.args.seed = int(request["seed"])  # negative: fresh noise every chunk
                    print(f"[phoenix] noise seed: {runtime.args.seed}", file=sys.stderr, flush=True)
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
