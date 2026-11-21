#!/usr/bin/env python
"""Normalization stats for relative-joint training (configs/task/fmc3_posttrain_relative.yaml).

    python relative_stats.py <dataset_dir> <upstream_stats.json> <out.json>

<upstream_stats.json> is a normalization-stats file in upstream's format computed with the same relative transform
(robosyn_transforms.RelativeJointTransform) applied. It is kept as is except the action q01/q99 of the 12 arm joints, which become the exact 1 % / 99 % quantiles of
a[t+k] - s[t] over every training window of the Phoenix reader: start t with a real future frame (t < n - 4), k < 32
inside the episode (padded steps carry no loss). The upstream tool takes the minimum / maximum of the per-episode
quantiles, about the extreme of the deltas: 2 rad per normalized unit against 0.5 for the absolute joints' z-score.
With the exact quantiles hardly any window touches the normalizer's +-5 clamp (with z-score about a tenth do). State stats and the grippers are untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

JOINT_DIMS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]  # robosyn_transforms.JOINT_DIMS
HORIZON = 32  # Joint96PosttrainDataset: action_offsets range(32)
FUTURE = 4  # action_video_freq_ratio: the reader's windows need frame t + 4


def episode_deltas(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """[windows * steps, 12] deltas a[t+k] - s[t] of one episode, padded steps left out."""
    n = len(state)
    rows = []
    for k in range(HORIZON):
        t = np.arange(max(0, min(n - FUTURE, n - k)))
        rows.append(action[t + k][:, JOINT_DIMS] - state[t][:, JOINT_DIMS])
    return np.concatenate(rows).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", type=Path)
    p.add_argument("upstream_stats", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()

    files = sorted((args.dataset / "data").glob("*/episode_*.parquet"))
    if not files:
        raise SystemExit(f"no episode parquet under {args.dataset}/data")
    parts = []
    for path in files:
        table = pq.read_table(path, columns=["observation.state", "action"])
        state = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float64)
        action = np.asarray(table.column("action").to_pylist(), dtype=np.float64)
        parts.append(episode_deltas(state, action))
    deltas = np.concatenate(parts)
    q01, q99 = np.quantile(deltas, [0.01, 0.99], axis=0)

    stats = json.loads(args.upstream_stats.read_text())
    field = stats["action"]["default"]
    if stats.get("num_episodes") != len(files):
        raise SystemExit(f"{args.upstream_stats} covers {stats.get('num_episodes')} episodes, the dataset {len(files)}")
    if abs(np.asarray(field["global_mean"])[JOINT_DIMS]).max() > 0.5:
        raise SystemExit(f"{args.upstream_stats} looks absolute: compute it with task=robosyn_delta")
    for name, value in (("global_q01", q01), ("global_q99", q99)):
        row = list(field[name])
        for dim, v in zip(JOINT_DIMS, value, strict=True):
            row[dim] = float(v)
        field[name] = row
    stats["fmc3_relative_quantiles"] = (
        f"action q01/q99 of dims {JOINT_DIMS}: exact quantiles of a[t+k]-s[t] over {len(deltas)} window steps "
        f"of {len(files)} episodes (relative_stats.py)"
    )
    args.out.write_text(json.dumps(stats, indent=2) + "\n")
    width = (q99 - q01) / 2
    print(f"{len(files)} episodes, {len(deltas)} window steps; rad per normalized unit {width.mean():.3f} "
          f"(min {width.min():.3f}, max {width.max():.3f}) -> {args.out}")  # fmt: skip


if __name__ == "__main__":
    main()
