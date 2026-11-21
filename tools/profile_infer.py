"""Where one Phoenix inference spends its time (the worker's Runtime.act on a real frame; cuda-synchronized timers
around the calls of upstream's joint loop). No change to what is computed.

    PHOENIX_ROOT=... python tools/profile_infer.py <policy/phoenix dir> <run dir> <step> <frames.npz> <prompt> <encoders> <wan base>
"""
import collections
import sys
import time

import numpy as np

worker_dir, run_dir, step, frames, prompt, encoders, wan = sys.argv[1:8]
sys.path.insert(0, worker_dir)
import phoenix_worker  # noqa: E402
import torch  # noqa: E402
import wam.inference.frame_joint as fj  # noqa: E402

runtime = phoenix_worker.Runtime(phoenix_worker.parse_args(
    ["--run-dir", run_dir, "--step", step, "--denoise-steps", "5", "--seed", "0", "--encoders", encoders, "--wan-base", wan]))
model = runtime.model
data = np.load(frames)
obs = lambda i: {"state": data["state"][i], **{c: data[c][i] for c in phoenix_worker.CAMERAS}}
spent, shapes = collections.defaultdict(list), {}


def timed(name, fn, note=None):
    def wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn(*args, **kwargs)
        torch.cuda.synchronize()
        spent[name].append(time.perf_counter() - t)
        if note is not None:
            note(args, kwargs, out)
        return out
    return wrapper


def note_mot(args, kwargs, out):
    shapes["video tokens"] = tuple(kwargs["embeds_all"]["video"].shape)
    shapes["action tokens"] = tuple(kwargs["embeds_all"]["action"].shape)
    shapes["video context"] = tuple(kwargs["context_all"]["video"]["context"].shape)
    shapes["action context"] = tuple(kwargs["context_all"]["action"]["context"].shape)
    mask = kwargs["attention_mask"]
    shapes["attention mask"] = (type(mask).__name__, tuple(getattr(mask, "shape", ())), str(getattr(mask, "dtype", "")))


observations = [obs(i) for i in range(16)]  # read before timing: the npz is compressed
obs = lambda i: observations[i]
if len(sys.argv) > 8 and sys.argv[8] == "compile":
    print("PROFILE compiled blocks", phoenix_worker.compile_trunk(model))
for i in range(3):
    runtime.act(obs(i), prompt, full=True, seed=0)  # warm-up
runtime.rgb = timed("worker: images to 256 tensors (CPU)", runtime.rgb)
runtime.proprio = timed("worker: state normalize", runtime.proprio)
enc = getattr(model, "joint_visual_encoder", None) or getattr(model, "current_visual_encoder", None)
print("PROFILE visual modules:", [n for n, m in model.named_children()])
fj.video_codec.encode_input_image_latents_tensor = timed("encode current frame (DINO + CSVAE)", fj.video_codec.encode_input_image_latents_tensor)
model.video_expert.pre_dit = timed("video pre_dit", model.video_expert.pre_dit)
model.action_expert.pre_dit = timed("action pre_dit", model.action_expert.pre_dit)
model.mot.forward = timed("MoT forward", model.mot.forward, note_mot)
model.video_expert.post_dit = timed("video post_dit", model.video_expert.post_dit)
model.action_expert.post_dit = timed("action post_dit", model.action_expert.post_dit)
for name in ("build_joint96_attention_mask", "joint96_camera_valid_mask"):
    if hasattr(fj, name):
        setattr(fj, name, timed(name, getattr(fj, name)))
fj.infer_frame_joint_window = timed("infer_frame_joint_window (all of upstream's loop)", fj.infer_frame_joint_window)
totals = []
N = 10
for i in range(N):
    torch.cuda.synchronize()
    t = time.perf_counter()
    runtime.act(obs(3 + i), prompt, full=True, seed=0)
    torch.cuda.synchronize()
    totals.append(time.perf_counter() - t)
print("PROFILE gpu", torch.cuda.get_device_name(0), "torch", torch.__version__, "calls", N)
print("PROFILE total act: median %.1f ms (min %.1f max %.1f)" % (1e3 * np.median(totals), 1e3 * min(totals), 1e3 * max(totals)))
for name, values in sorted(spent.items(), key=lambda kv: -sum(kv[1])):
    print("PROFILE %-52s %7.1f ms per inference (%d calls, %.2f ms each)" % (name, 1e3 * sum(values) / N, len(values) // N, 1e3 * np.mean(values)))
print("PROFILE shapes", shapes)
mot = model.mot
print("PROFILE mot layers", mot.num_layers, "heads", mot.num_heads, "head dim", mot.attn_head_dim, "backend", mot.attention_backend,
      "compile", mot.compile_mode, "flex compiled", mot._flex_attention_compiled, "flex_masked_batches", mot.flex_masked_batches)
params = lambda m: sum(p.numel() for p in m.parameters()) / 1e9
print("PROFILE params (1e9): video expert %.2f action expert %.2f total %.2f" % (params(model.video_expert), params(model.action_expert), params(model)))
print("PROFILE flags: cudnn.benchmark", torch.backends.cudnn.benchmark, "tf32 matmul", torch.backends.cuda.matmul.allow_tf32,
      "dtype", model.torch_dtype)
