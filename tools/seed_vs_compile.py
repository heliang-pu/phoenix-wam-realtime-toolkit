"""How large the compiled-vs-eager difference is next to what a new noise draw does (with phoenix_seed -1 every
chunk gets fresh noise). Same frames: eager seed 0, eager seeds 1-3, compiled seed 0."""
import sys

import numpy as np

worker_dir, run_dir, step, frames, prompt, encoders, wan = sys.argv[1:8]
sys.path.insert(0, worker_dir)
import phoenix_worker  # noqa: E402

runtime = phoenix_worker.Runtime(phoenix_worker.parse_args(
    ["--run-dir", run_dir, "--step", step, "--denoise-steps", "5", "--seed", "0", "--encoders", encoders, "--wan-base", wan]))
data = np.load(frames)
index = np.linspace(0, len(data["rows"]) - 1, 20).astype(int)
observations = [{"state": data["state"][i], **{c: data[c][i] for c in phoenix_worker.CAMERAS}} for i in index]
ARM, GRIP = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], [6, 13]
draw = lambda seed: np.stack([runtime.act(o, prompt, full=True, seed=seed) for o in observations])
eager = {seed: draw(seed) for seed in range(4)}
phoenix_worker.compile_trunk(runtime.model), phoenix_worker.compile_encoder(runtime.model)
draw(0)
compiled = {seed: draw(seed) for seed in range(2)}


def stats(a, b, dims, steps=slice(None)):
    d = np.abs(a - b)[:, steps][:, :, dims]
    per_chunk = d.max((1, 2))
    return "mean %.4f | per-chunk max: median %.4f, p90 %.4f, max %.4f" % (d.mean(), np.median(per_chunk), np.quantile(per_chunk, 0.9), per_chunk.max())


print("CMP frames", len(observations))
for name, dims in (("arm joints (rad)", ARM), ("grippers (opening fraction)", GRIP)):
    print("CMP", name)
    print("CMP   compiled vs eager, same noise (seed 0):      ", stats(compiled[0], eager[0], dims))
    print("CMP   compiled vs eager, same noise (seed 1):      ", stats(compiled[1], eager[1], dims))
    for seed in (1, 2, 3):
        print("CMP   eager seed %d vs eager seed 0 (new noise):    " % seed, stats(eager[seed], eager[0], dims))
    print("CMP   first 20 steps only: compiled vs eager        ", stats(compiled[0], eager[0], dims, slice(0, 20)))
    print("CMP   first 20 steps only: seed 1 vs seed 0          ", stats(eager[1], eager[0], dims, slice(0, 20)))
# does the gripper command cross the adapter's close thresholds differently? (opening fraction below 0.3 = closing)
for threshold in (0.3, 0.5, 0.7):
    flips = lambda a, b: int(((a[:, :, GRIP] < threshold) != (b[:, :, GRIP] < threshold)).sum())
    print("CMP gripper steps on the other side of %.1f: compiled vs eager %d, seed 1 vs seed 0 %d (of %d)" % (
        threshold, flips(compiled[0], eager[0]), flips(eager[1], eager[0]), eager[0][:, :, GRIP].size))
print("CMP_DONE")
