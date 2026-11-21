"""Offline check of inference-time RTC on real frames (no robot): timing, memory, the beta-0 identity and the joins.

    PHOENIX_ROOT=<Phoenix-MWA> python tools/rtc_offline.py <policy/phoenix dir> <run dir> <step> <frames.npz> <prompt> <out.json>
                                                    [--encoders D --wan-base D --denoise 5]

For each launch row r of the episode: chunk A from the observation at row r (plain). The next chunk B is drawn from
the observation at row r + l, l = the index of A at which it is launched, for (l, d) in SCENARIOS, in four ways with
the same noise: plain, inpaint (first d of A's remaining steps exactly, the rest by RTC's soft mask), guidance (RTC,
beta 5) and guidance beta 0 (must equal plain). The arm would execute A up to index l + d - 1 and then B from d.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("worker_dir")
p.add_argument("run_dir")
p.add_argument("step", type=int)
p.add_argument("frames")
p.add_argument("prompt")
p.add_argument("out")
p.add_argument("--encoders")
p.add_argument("--wan-base")
p.add_argument("--denoise", type=int, default=5)
p.add_argument("--beta", type=float, default=5.0)
a = p.parse_args()
sys.path.insert(0, a.worker_dir)
import phoenix_worker  # noqa: E402
import torch  # noqa: E402

argv = ["--run-dir", a.run_dir, "--step", str(a.step), "--denoise-steps", str(a.denoise), "--seed", "0"]
for flag, value in (("--encoders", a.encoders), ("--wan-base", a.wan_base)):
    if value:
        argv += [flag, value]
runtime = phoenix_worker.Runtime(phoenix_worker.parse_args(argv))
data = np.load(a.frames)
rows = data["rows"].tolist()
ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
SCENARIOS = [(20, 6), (26, 6), (14, 6), (20, 10), (20, 12)]


# read once, before any timing: the npz is compressed and every access unpacks a whole array
ARRAYS = {name: data[name] for name in ("state", *phoenix_worker.CAMERAS)}


def obs(row):
    i = rows.index(row)
    return {name: array[i] for name, array in ARRAYS.items()}


def act(row, seed, **extra):
    torch.cuda.synchronize()
    t = time.perf_counter()
    chunk = runtime.act(obs(row), a.prompt, full=True, seed=seed, **extra)
    torch.cuda.synchronize()
    return chunk, time.perf_counter() - t


act(rows[0], 0)  # warm-up
act(rows[0], 0, rtc={"prev": act(rows[0], 0)[0][:12].tolist(), "delay": 6})
torch.cuda.reset_peak_memory_stats()
base = torch.cuda.memory_allocated()
times = {"plain": [], "inpaint": [], "guidance": [], "guidance0": []}
peak = {}
records = []
launch_rows = [r for r in rows if all(r + o in rows for o in (14, 20, 26))]
for r in launch_rows:
    A, t = act(r, 0)
    times["plain"].append(t)
    peak["plain"] = max(peak.get("plain", 0), torch.cuda.max_memory_allocated())
    for launch, d in SCENARIOS:
        rest = A[launch:]
        k = len(rest)
        variants = {}
        variants["plain"], t = act(r + launch, 1)
        times["plain"].append(t)
        variants["inpaint"], t = act(r + launch, 1, inpaint=rest.tolist(), inpaint_steps=min(d, k), inpaint_soft=True)
        times["inpaint"].append(t)
        torch.cuda.reset_peak_memory_stats()
        variants["guidance"], t = act(r + launch, 1, rtc={"prev": rest.tolist(), "delay": d, "beta": a.beta})
        times["guidance"].append(t)
        peak["guidance"] = max(peak.get("guidance", 0), torch.cuda.max_memory_allocated())
        variants["guidance0"], t = act(r + launch, 1, rtc={"prev": rest.tolist(), "delay": d, "beta": 0.0})
        times["guidance0"].append(t)
        record = {"row": r, "launch": launch, "delay": d, "left": k}
        take = min(d, 31)  # B's index the arm takes over at; before it the arm executes A (its last target past the end)
        last = A[min(launch + take - 1, 31)]
        record["natural_step"] = float(np.abs(A[min(launch + take, 31)] - last)[ARM].max())
        for name, B in variants.items():
            n = min(d, k)
            record[name] = {
                "join_step": float(np.abs(B[take] - last)[ARM].max()),
                "prefix_err": float(np.abs(B[:n] - rest[:n])[:, ARM].max()),
                "overlap_err": float(np.abs(B[:k] - rest)[:, ARM].max()),
                "next_step": float(np.abs(B[take + 1] - B[take])[ARM].max()) if take + 1 < 32 else None,
                "max_inner_step": float(np.abs(np.diff(B[take:], axis=0))[:, ARM].max()) if take < 31 else None,
                "from_plain_free": float(np.abs(B[k:] - variants["plain"][k:])[:, ARM].max()) if k < 32 else None,
                "from_plain_all": float(np.abs(B - variants["plain"])[:, ARM].max()),
                "gripper_from_plain": float(np.abs(B - variants["plain"])[:, [6, 13]].max()),
            }
        records.append(record)
        print(json.dumps(record), flush=True)
summary = {
    "run_dir": a.run_dir, "step": runtime.step, "frames": a.frames, "denoise": a.denoise, "beta": a.beta,
    "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
    "time_s": {k: {"n": len(v), "median": float(np.median(v)), "min": float(min(v)), "max": float(max(v))}
               for k, v in times.items()},
    "peak_alloc_gib": {k: v / 2**30 for k, v in peak.items()}, "loaded_alloc_gib": base / 2**30,
    "records": records,
}
Path(a.out).write_text(json.dumps(summary, indent=1))
print("RTC_OFFLINE_DONE", json.dumps(summary["time_s"]), json.dumps(summary["peak_alloc_gib"]), flush=True)
