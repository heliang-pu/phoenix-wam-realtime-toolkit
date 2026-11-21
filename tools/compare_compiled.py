"""Eager against compiled (phoenix_worker.compile_trunk + compile_encoder) on recorded frames, in one process: time
per inference and how far the chunks differ.

    PHOENIX_ROOT=<Phoenix-MWA> python tools/compare_compiled.py <policy/phoenix dir> <run dir> <step> <frames.npz> \
        <prompt> <encoders> <wan base>
"""
import json
import sys
import time

import numpy as np

worker_dir, run_dir, step, frames, prompt, encoders, wan = sys.argv[1:8]
sys.path.insert(0, worker_dir)
import phoenix_worker  # noqa: E402
import torch  # noqa: E402

runtime = phoenix_worker.Runtime(phoenix_worker.parse_args(
    ["--run-dir", run_dir, "--step", step, "--denoise-steps", "5", "--seed", "0", "--encoders", encoders, "--wan-base", wan]))
data = np.load(frames)
N = 12
observations = [{"state": data["state"][i], **{c: data[c][i] for c in phoenix_worker.CAMERAS}} for i in range(N)]
ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


def run(tag):
    chunks, times = [], []
    for o in observations:
        torch.cuda.synchronize()
        t = time.perf_counter()
        chunks.append(runtime.act(o, prompt, full=True, seed=0))
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    print("SPEED %-28s median %.1f ms  min %.1f  max %.1f  (n %d)" % (tag, 1e3 * np.median(times), 1e3 * min(times), 1e3 * max(times), len(times)), flush=True)
    return np.stack(chunks), times


for o in observations[:3]:
    runtime.act(o, prompt, full=True, seed=0)
eager, eager_times = run("eager")
eager2, _ = run("eager again")
print("SPEED eager vs eager again: max |diff| arm %.6f rad" % np.abs(eager - eager2)[:, :, ARM].max(), flush=True)
t = time.perf_counter()
print("SPEED blocks", phoenix_worker.compile_trunk(runtime.model), "| encoder:", phoenix_worker.compile_encoder(runtime.model), flush=True)
runtime.act(observations[0], prompt, full=True, seed=0)
torch.cuda.synchronize()
print("SPEED first compiled inference (compilation): %.1f s" % (time.perf_counter() - t), flush=True)
runtime.act(observations[1], prompt, full=True, seed=0)
compiled, compiled_times = run("compiled")
diff = np.abs(compiled - eager)
print("SPEED compiled vs eager: arm max %.5f rad, median of per-chunk max %.5f; grippers max %.5f" % (
    diff[:, :, ARM].max(), np.median(diff[:, :, ARM].max((1, 2))), diff[:, :, [6, 13]].max()), flush=True)
print("SPEED peak alloc GiB %.1f" % (torch.cuda.max_memory_allocated() / 2**30), flush=True)
phoenix_worker.restore_trunk(runtime.model)
back, _ = run("trunk eager again, encoder still compiled")
print("SPEED restored vs eager: max |diff| arm %.6f" % np.abs(back - eager)[:, :, ARM].max(), flush=True)
print("SPEED_DONE", json.dumps({"eager_ms": 1e3 * float(np.median(eager_times)), "compiled_ms": 1e3 * float(np.median(compiled_times))}))
