"""torch.profiler over two inferences: per-operator CUDA time against wall time (is the trunk kernel- or launch-bound?)."""
import sys
import time

import numpy as np

worker_dir, run_dir, step, frames, prompt, encoders, wan = sys.argv[1:8]
sys.path.insert(0, worker_dir)
import phoenix_worker  # noqa: E402
import torch  # noqa: E402
from torch.profiler import ProfilerActivity  # noqa: E402
from torch.profiler import profile  # noqa: E402

runtime = phoenix_worker.Runtime(phoenix_worker.parse_args(
    ["--run-dir", run_dir, "--step", step, "--denoise-steps", "5", "--seed", "0", "--encoders", encoders, "--wan-base", wan]))
data = np.load(frames)
observations = [{"state": data["state"][i], **{c: data[c][i] for c in phoenix_worker.CAMERAS}} for i in range(6)]
for o in observations[:3]:
    runtime.act(o, prompt, full=True, seed=0)
torch.cuda.synchronize()
t = time.perf_counter()
for o in observations[3:5]:
    runtime.act(o, prompt, full=True, seed=0)
torch.cuda.synchronize()
print("OPS wall per inference without profiler: %.1f ms" % (1e3 * (time.perf_counter() - t) / 2))
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for o in observations[3:5]:
        runtime.act(o, prompt, full=True, seed=0)
    torch.cuda.synchronize()
events = prof.key_averages()
cuda_total = sum(e.self_device_time_total for e in events) / 2e3
print("OPS summed self CUDA time per inference: %.1f ms; operator calls per inference: %d" % (
    cuda_total, sum(e.count for e in events if e.self_device_time_total > 0) / 2))
print(events.table(sort_by="self_cuda_time_total", row_limit=28, max_name_column_width=46))
