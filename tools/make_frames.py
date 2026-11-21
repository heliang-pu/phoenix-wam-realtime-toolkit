"""Frames of one episode of a LeRobot v2.1 recording, for the offline tools here.

    python tools/make_frames.py <recording dir> <episode index> <out.npz> [every]

Writes, for the sampled rows, cam_high / cam_left_wrist / cam_right_wrist (uint8 HWC RGB), state (14-D,
observation.state units) and rows (their indices). Rows: every ``every``-th row r (default 20) from row 10, each with
r + 14, r + 20 and r + 26, the launch offsets rtc_offline.py compares. Needs av and pandas.
"""
import sys
from pathlib import Path

import av
import numpy as np
import pandas as pd

rec, episode, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
every = int(sys.argv[4]) if len(sys.argv) > 4 else 20
frame = pd.read_parquet(rec / f"data/chunk-000/episode_{episode:06d}.parquet")
states = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
rows = sorted({r + o for r in range(10, len(states) - 40, every) for o in (0, 14, 20, 26)})
data = {"rows": np.array(rows), "state": states[rows]}
for cam in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
    with av.open(str(rec / f"videos/chunk-000/observation.images.{cam}/episode_{episode:06d}.mp4")) as container:
        images = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    if len(images) != len(states):
        raise SystemExit(f"{cam}: {len(images)} frames, {len(states)} rows")
    data[cam] = np.stack([images[r] for r in rows])
np.savez_compressed(out, **data)
print(out, len(states), "rows ->", len(rows), data["cam_high"].shape)
