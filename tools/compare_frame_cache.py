"""The frame cache against the full forward on recorded frames, in one process: eager, eager + cache, compiled,
compiled + cache. Time per inference, how far the chunks differ, and what the cache did.

    PHOENIX_ROOT=<Phoenix-MWA> python tools/compare_frame_cache.py <policy/phoenix dir> <run dir> <step> <frames.npz> <prompt> \
        <encoders> <wan base>
"""
import sys
import time

import numpy as np

worker_dir, run_dir, step, frames, prompt, encoders, wan = sys.argv[1:8]
sys.path.insert(0, worker_dir)
import phoenix_worker  # noqa: E402
import torch  # noqa: E402

runtime = phoenix_worker.Runtime(phoenix_worker.parse_args(
    ["--run-dir", run_dir, "--step", step, "--denoise-steps", "5", "--seed", "0", "--encoders", encoders, "--wan-base", wan]))
model = runtime.model
data = np.load(frames)
N = 12
observations = [{"state": data["state"][i], **{c: data[c][i] for c in phoenix_worker.CAMERAS}} for i in range(N)]
ARM, GRIP = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], [6, 13]


def run(tag, seed=0):
    chunks, times = [], []
    for o in observations:
        torch.cuda.synchronize()
        t = time.perf_counter()
        chunks.append(runtime.act(o, prompt, full=True, seed=seed))
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    print("FC %-26s median %.1f ms  min %.1f  max %.1f" % (tag, 1e3 * np.median(times), 1e3 * min(times), 1e3 * max(times)), flush=True)
    return np.stack(chunks)


def diff(tag, a, b):
    d = np.abs(a - b)
    per = d[:, :, ARM].max((1, 2))
    print("FC   %-34s arm mean %.5f, per-chunk max median %.5f max %.5f; grippers mean %.5f max %.5f" % (
        tag, d[:, :, ARM].mean(), np.median(per), per.max(), d[:, :, GRIP].mean(), d[:, :, GRIP].max()), flush=True)


for o in observations[:3]:
    runtime.act(o, prompt, full=True, seed=0)
eager = run("eager")
seed1 = run("eager, another seed", seed=1)
cache = phoenix_worker.FrameCache(model).install()
runtime.act(observations[0], prompt, full=True, seed=0)
runtime.act(observations[1], prompt, full=True, seed=0)
cache.counts.update(full=0, cached=0)
cached = run("eager + frame cache")
print("FC   kept inputs:", [None if r is None else tuple(r.shape) for r in cache.ref], flush=True)
print("FC   cache:", cache.counts, "per inference:", {k: v / N for k, v in cache.counts.items()}, "current tokens", cache.n,
      "kept layers", len(cache.keys), "key shape", tuple(cache.keys[0].shape), flush=True)
diff("eager + cache vs eager", cached, eager)
diff("another seed vs eager (for scale)", seed1, eager)
cache.uninstall()
again = run("eager, cache taken off")
diff("taken off vs eager", again, eager)
print("FC compile:", phoenix_worker.compile_trunk(model), phoenix_worker.compile_encoder(model), flush=True)
t = time.perf_counter()
runtime.act(observations[0], prompt, full=True, seed=0)
print("FC   first compiled inference %.1f s" % (time.perf_counter() - t), flush=True)
runtime.act(observations[1], prompt, full=True, seed=0)
compiled = run("compiled")
diff("compiled vs eager", compiled, eager)
cache = phoenix_worker.FrameCache(model).install()
t = time.perf_counter()
runtime.act(observations[0], prompt, full=True, seed=0)
print("FC   first compiled + cache inference %.1f s" % (time.perf_counter() - t), flush=True)
runtime.act(observations[1], prompt, full=True, seed=0)
cache.counts.update(full=0, cached=0)
both = run("compiled + frame cache")
print("FC   cache per inference:", {k: v / N for k, v in cache.counts.items()}, flush=True)
diff("compiled + cache vs eager", both, eager)
diff("compiled + cache vs compiled", both, compiled)
print("FC peak alloc GiB %.1f" % (torch.cuda.max_memory_allocated() / 2**30))
print("FC_DONE")
