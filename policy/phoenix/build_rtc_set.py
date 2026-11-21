#!/usr/bin/env python
"""A LeRobot v2.1 training set as native LeRobot v3 at a 30 Hz clock: the input of upstream prometheus's Training RTC.

    python build_rtc_set.py <v2.1 set> <out v3 dir> [--workers 24] [--limit N] [--crop-square]

Upstream's prometheus branch (Training RTC: chunks conditioned on an executed prefix) reads native LeRobot v3 only:
contiguous rows, an episode-relative 30 Hz clock, three H.264 views whose frames sit at the rows' times
(phoenix/data/prepare.py, phoenix/dataset.py). The post-train sets are v2.1 with 10 Hz frames, possibly with rows
dropped (pauses, static tails) and episodes repeated as copies (a per-task sampling weight). Here:

- every row of the source becomes one row, frame k of an episode stamped k/30: one model step per real 10 Hz step,
  as now (a 32-step chunk is still 3.2 s on the robot, deployed at 10 Hz);
- action = the next row's state (upstream's materialized_next_observation; dropped rows make the source's
  "next state" column skip the dropped rows), the last row keeping its own;
- each view is re-encoded from the source frames of the kept rows (libx264, 30 fps), once per source episode:
  the repeats (the per-task weight) are separate rows pointing at the same video files;
- grippers stay opening fractions (upstream's gripper_unit "normalized");
- --crop-square: each view cut to its centred square (640 x 480 -> the middle 480 x 480) before encoding. Upstream's
  reader squashes a whole frame to 256 x 256; post-train runs see the views resized and centre-cropped to 256
  (phoenix_worker.py rgb: the same middle square), so an RTC run continued from them keeps that input. rtc_worker.py
  crops the same way when the run says so (export_rtc.py --image-crop center_square).

Then, with upstream's own stages (see the README, Training RTC): data prepare (frozen manifest, train/val
split, normalization fitted on the train split), data cache-text, train.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FPS = 30
VIDEO_KEYS = ("observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist")
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
EPISODES_PER_FILE = 200


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def center_square(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    return np.ascontiguousarray(image[top : top + side, left : left + side])


def encode_view(src_video: str, times: list[float], dst: str, crf: int, crop: bool = False) -> int:
    """The source frames at ``times`` (the kept rows' stamps), encoded at 30 fps with pts k; ``crop``: the centred
    square of each."""
    import av

    wanted, frames = list(times), []
    with av.open(src_video) as container:
        stream = container.streams.video[0]
        tolerance = 0.4 / 25  # well under one source frame (0.04 s)
        for frame in container.decode(stream):
            stamp = float(frame.pts * stream.time_base)
            if wanted and stamp > wanted[0] + tolerance:
                raise ValueError(f"{src_video}: no frame at {wanted[0]:.3f} s")
            if wanted and abs(stamp - wanted[0]) <= tolerance:
                image = frame.to_ndarray(format="rgb24")
                frames.append(center_square(image) if crop else image)
                wanted.pop(0)
                if not wanted:
                    break
    if wanted:
        raise ValueError(f"{src_video}: {len(wanted)} rows past its last frame")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with av.open(dst, "w") as out:
        stream = out.add_stream("libx264", rate=FPS)
        stream.height, stream.width = frames[0].shape[:2]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": "fast", "g": "15"}
        for k, image in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = k
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return len(frames)


def encode_episode(job: dict) -> int:
    for key in VIDEO_KEYS:
        if encode_view(job["src"][key], job["times"], job["dst"][key], job["crf"], job["crop"]) != len(job["times"]):
            raise ValueError(f"{job['dst'][key]}: frame count")
    return len(job["times"])


def video_shape(shape: list[int], crop: bool) -> list[int]:
    """A v2.1 video feature's [height, width, channels] after --crop-square."""
    if not crop:
        return shape
    side = min(shape[0], shape[1])
    return [side, side, shape[2]]


def check_video(path: Path, length: int) -> None:
    """Upstream's reader matches frames by pts: k/30 for k < length."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stamps = [float(f.pts * stream.time_base) for f in container.decode(stream)]
    if len(stamps) != length or not np.allclose(stamps, np.arange(length) / FPS, atol=1e-4):
        raise ValueError(f"{path}: {len(stamps)} frames, expected {length} at 1/{FPS} s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--limit", type=int, default=0, help="only N episodes (a test build)")
    parser.add_argument("--start", type=int, default=0, help="with --limit: from this episode on")
    parser.add_argument("--crop-square", action="store_true", help="each view cut to its centred square")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists")
    info = json.loads((args.src / "meta/info.json").read_text())
    if info["codebase_version"] != "v2.1" or [k for k in VIDEO_KEYS if k not in info["features"]]:
        raise SystemExit(f"{args.src}: expected a v2.1 set with {VIDEO_KEYS}")
    episodes = jsonl(args.src / "meta/episodes.jsonl")
    sources = {r["episode_index"]: r for r in jsonl(args.src / "meta/episode_sources.jsonl")}
    tasks = [t["task"] for t in sorted(jsonl(args.src / "meta/tasks.jsonl"), key=lambda t: t["task_index"])]
    if args.limit:
        episodes = episodes[args.start : args.start + args.limit]
    tmp = args.out.with_name(f".{args.out.name}.building")
    shutil.rmtree(tmp, ignore_errors=True)

    def src_path(template: str, episode: int, **fmt) -> Path:
        return args.src / template.format(episode_chunk=episode // info["chunks_size"], episode_index=episode, **fmt)

    # one video set per source episode: repeated episodes are identical copies of it
    videos, jobs, rows_of = {}, [], {}
    for e in episodes:
        i = e["episode_index"]
        s = sources[i]
        key = (s["source"], s["source_episode"])
        table = pq.read_table(src_path(info["data_path"], i), columns=["timestamp", "observation.state", "action",
                                                                         "task_index"])  # fmt: skip
        rows_of[i] = table
        if key in videos:
            continue
        uid = len(videos)
        videos[key] = uid
        jobs.append({"src": {k: str(src_path(info["video_path"], i, video_key=k)) for k in VIDEO_KEYS},
                     "dst": {k: str(tmp / VIDEO_PATH.format(video_key=k, chunk_index=uid // 1000,
                                                             file_index=uid % 1000)) for k in VIDEO_KEYS},
                     "times": table.column("timestamp").to_pylist(), "crf": args.crf,
                     "crop": args.crop_square})  # fmt: skip
    with ProcessPoolExecutor(args.workers) as pool:
        for done, _ in enumerate(pool.map(encode_episode, jobs, chunksize=1), 1):
            if done % 100 == 0:
                print(f"encoded {done}/{len(jobs)} source episodes", flush=True)

    schema = pa.schema([("index", pa.int64()), ("frame_index", pa.int64()), ("episode_index", pa.int64()),
                        ("timestamp", pa.float32()), ("task_index", pa.int64()),
                        ("observation.state", pa.list_(pa.float32(), 14)), ("action", pa.list_(pa.float32(), 14))])  # fmt: skip
    meta_rows, index, writer, file_no = [], 0, None, -1
    for n, e in enumerate(episodes):
        if n % EPISODES_PER_FILE == 0:
            if writer is not None:
                writer.close()
            file_no += 1
            path = tmp / DATA_PATH.format(chunk_index=0, file_index=file_no)
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(path, schema)
        table, i = rows_of[e["episode_index"]], e["episode_index"]
        state = np.stack(table.column("observation.state").to_numpy(zero_copy_only=False)).astype(np.float32)
        action = np.vstack([state[1:], np.stack(table.column("action").to_numpy(zero_copy_only=False))[-1:]])
        length = len(state)
        task_index = set(table.column("task_index").to_pylist())
        if len(task_index) != 1:
            raise ValueError(f"episode {i}: several tasks")
        writer.write_table(pa.table({
            "index": np.arange(index, index + length), "frame_index": np.arange(length),
            "episode_index": np.full(length, n), "timestamp": (np.arange(length) / FPS).astype(np.float32),
            "task_index": np.full(length, task_index.pop()),
            "observation.state": pa.array(list(state), pa.list_(pa.float32(), 14)),
            "action": pa.array(list(action.astype(np.float32)), pa.list_(pa.float32(), 14)),
        }, schema=schema))  # fmt: skip  # one row group per episode
        uid = videos[(sources[i]["source"], sources[i]["source_episode"])]
        row = {"episode_index": n, "tasks": [tasks[int(table.column("task_index")[0].as_py())]], "length": length,
               "dataset_from_index": index, "dataset_to_index": index + length,
               "data/chunk_index": 0, "data/file_index": file_no, "source_episode_index": i}  # fmt: skip
        for key in VIDEO_KEYS:
            row.update({f"videos/{key}/chunk_index": uid // 1000, f"videos/{key}/file_index": uid % 1000,
                        f"videos/{key}/from_timestamp": 0.0, f"videos/{key}/to_timestamp": length / FPS})  # fmt: skip
        meta_rows.append(row)
        index += length
    writer.close()

    features = {
        "observation.state": {"dtype": "float32", "shape": [14], "names": info["features"]["observation.state"].get("names")},
        "action": {"dtype": "float32", "shape": [14], "names": info["features"]["action"].get("names")},
        **{k: {"dtype": "video", "shape": video_shape(info["features"][k]["shape"], args.crop_square),
               "names": info["features"][k].get("names"),
               "info": {"video.fps": FPS, "video.codec": "h264", "video.pix_fmt": "yuv420p"}} for k in VIDEO_KEYS},
        **{k: {"dtype": dt, "shape": [1], "names": None} for k, dt in (("timestamp", "float32"), ("frame_index", "int64"),
            ("episode_index", "int64"), ("index", "int64"), ("task_index", "int64"))},
    }  # fmt: skip
    meta = tmp / "meta"
    (meta / "episodes/chunk-000").mkdir(parents=True)
    out_info = {"codebase_version": "v3.0", "robot_type": info.get("robot_type", "cobotmagic"),
                "total_episodes": len(meta_rows), "total_frames": index, "total_tasks": len(tasks),
                "chunks_size": 1000, "data_files_size_in_mb": 100, "video_files_size_in_mb": 500, "fps": FPS,
                "splits": {"train": f"0:{len(meta_rows)}"}, "data_path": DATA_PATH, "video_path": VIDEO_PATH,
                "features": features}  # fmt: skip
    (meta / "info.json").write_text(json.dumps(out_info, indent=4) + "\n")
    pq.write_table(pa.table({"task_index": list(range(len(tasks))), "task": tasks}), meta / "tasks.parquet")
    pq.write_table(pa.Table.from_pylist(meta_rows), meta / "episodes/chunk-000/file-000.parquet")
    (tmp / "CONVERTED.json").write_text(json.dumps({
        "schema": "fmc3.rtc_set.v1", "source": str(args.src.resolve()), "action_semantics": "materialized_next_observation",
        "gripper_unit": "normalized", "joint_unit": "radian", "clock": f"one row per real 10 Hz step, stamped k/{FPS}",
        "episodes": len(meta_rows), "source_videos": len(videos), "frames": index,
        "image_crop": "center_square" if args.crop_square else "none",
    }, indent=1) + "\n")  # fmt: skip
    for row in meta_rows[:: max(1, len(meta_rows) // 20)]:  # spot check: frames at k/30, one per row
        for key in VIDEO_KEYS:
            check_video(tmp / VIDEO_PATH.format(video_key=key, chunk_index=row[f"videos/{key}/chunk_index"],
                                                file_index=row[f"videos/{key}/file_index"]), row["length"])  # fmt: skip
    tmp.rename(args.out)
    print(f"{args.out}: {len(meta_rows)} episodes ({len(videos)} source videos), {index} rows at {FPS} Hz")


if __name__ == "__main__":
    main()
