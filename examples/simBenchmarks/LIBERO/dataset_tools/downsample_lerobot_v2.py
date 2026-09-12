#!/usr/bin/env python3
"""Create an independent, episode-safe 10 Hz overlay of LeRobot v2 datasets.

The source dataset is read-only.  Parquet rows are sampled at a fixed stride,
timestamps/frame indices are rebuilt, statistics and step caches are regenerated,
and metadata is copied into a sibling output root.  By default videos are
hard-linked and remain at their source FPS; the loader still selects the correct
frames because it uses the rebuilt timestamps.  ``--video-mode reencode`` can be
used when a physically 10 Hz video stream is required, at the cost of extra time
and storage.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _array_stats(values: list[np.ndarray]) -> dict[str, list[float]]:
    data = np.concatenate(values, axis=0).astype(np.float32)
    return {
        "mean": np.mean(data, axis=0).tolist(),
        "std": np.std(data, axis=0).tolist(),
        "min": np.min(data, axis=0).tolist(),
        "max": np.max(data, axis=0).tolist(),
        "q01": np.quantile(data, 0.01, axis=0).tolist(),
        "q99": np.quantile(data, 0.99, axis=0).tolist(),
    }


def _downsample_table(table: pd.DataFrame, stride: int, fps: int, index_start: int) -> pd.DataFrame:
    sampled = table.iloc[::stride].copy().reset_index(drop=True)
    sampled["frame_index"] = np.arange(len(sampled), dtype=np.int64)
    sampled["index"] = np.arange(index_start, index_start + len(sampled), dtype=np.int64)
    # Rebuild timestamps instead of inheriting floating-point source artifacts.
    sampled["timestamp"] = np.arange(len(sampled), dtype=np.float32) / float(fps)
    return sampled


def _reencode_video(source: Path, target: Path, stride: int, fps: int) -> None:
    import av

    target.parent.mkdir(parents=True, exist_ok=True)
    source_container = av.open(str(source))
    target_container = av.open(str(target), mode="w")
    try:
        source_stream = source_container.streams.video[0]
        target_stream = target_container.add_stream("libx264", rate=fps)
        target_stream.width = source_stream.width
        target_stream.height = source_stream.height
        target_stream.pix_fmt = "yuv420p"
        output_index = 0
        for input_index, frame in enumerate(source_container.decode(video=0)):
            if input_index % stride:
                continue
            frame.pts = output_index
            frame.time_base = target_stream.time_base
            for packet in target_stream.encode(frame):
                target_container.mux(packet)
            output_index += 1
        for packet in target_stream.encode():
            target_container.mux(packet)
    finally:
        target_container.close()
        source_container.close()


def _copy_videos(source_root: Path, target_root: Path, stride: int, fps: int, mode: str) -> None:
    for source in sorted((source_root / "videos").rglob("*.mp4")):
        target = target_root / source.relative_to(source_root)
        if mode == "timestamp":
            _link_or_copy(source, target)
        else:
            _reencode_video(source, target, stride, fps)


def convert_dataset(source_root: Path, target_root: Path, stride: int, fps: int, video_mode: str) -> None:
    if target_root.exists() and any(target_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {target_root}")
    target_root.mkdir(parents=True, exist_ok=True)

    source_info = json.loads((source_root / "meta/info.json").read_text())
    source_episodes = [
        json.loads(line)
        for line in (source_root / "meta/episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]

    for relative in (".gitattributes", "meta/modality.json", "meta/tasks.jsonl"):
        source = source_root / relative
        if source.exists():
            _link_or_copy(source, target_root / relative)

    all_tables: dict[str, list[np.ndarray]] = {}
    episode_lengths: dict[int, int] = {}
    episode_statistics: dict[int, dict[str, dict[str, list[float]]]] = {}
    steps: list[tuple[int, int]] = []
    global_index = 0
    for source_parquet in sorted((source_root / "data").glob("*/episode_*.parquet")):
        table = pd.read_parquet(source_parquet)
        sampled = _downsample_table(table, stride, fps, global_index)
        global_index += len(sampled)
        episode_ids = sampled["episode_index"].to_numpy()
        if len(set(int(value) for value in episode_ids)) != 1:
            raise ValueError(f"Expected one episode per parquet file: {source_parquet}")
        episode_index = int(episode_ids[0])
        episode_lengths[episode_index] = len(sampled)
        steps.extend((episode_index, index) for index in range(len(sampled)))
        for column in ("observation.state", "action"):
            if column in sampled:
                values = np.stack(sampled[column].to_numpy())
                all_tables.setdefault(column, []).append(values)
                episode_statistics.setdefault(episode_index, {})[column] = _array_stats([values])
        target_parquet = target_root / source_parquet.relative_to(source_root)
        target_parquet.parent.mkdir(parents=True, exist_ok=True)
        sampled.to_parquet(target_parquet, index=False)

    if not episode_lengths:
        raise FileNotFoundError(f"No episode parquet files found under {source_root / 'data'}")

    episodes_out = []
    for episode in source_episodes:
        episode = dict(episode)
        episode_index = int(episode["episode_index"])
        if episode_index not in episode_lengths:
            raise ValueError(f"Metadata episode {episode_index} has no parquet file")
        episode["length"] = episode_lengths[episode_index]
        episodes_out.append(episode)
    with (target_root / "meta/episodes.jsonl").open("w", encoding="utf-8") as handle:
        for episode in episodes_out:
            handle.write(json.dumps(episode, ensure_ascii=False) + "\n")

    info = dict(source_info)
    info["fps"] = fps
    info["total_frames"] = int(sum(episode_lengths.values()))
    info["total_episodes"] = len(episode_lengths)
    info["source_fps"] = source_info.get("fps")
    if video_mode == "reencode":
        for feature in info.get("features", {}).values():
            if feature.get("dtype") == "video":
                feature.setdefault("info", {})["video.fps"] = fps
    _json_dump(target_root / "meta/info.json", info)

    statistics = {column: _array_stats(values) for column, values in all_tables.items()}
    _json_dump(
        target_root / "meta/stats_gr00t.json",
        {"__format_version": 2, "__cache_config": {"mode": "abs"}, "statistics": statistics},
    )
    with (target_root / "meta/episodes_stats.jsonl").open("w", encoding="utf-8") as handle:
        for episode_index in sorted(episode_statistics):
            handle.write(
                json.dumps(
                    {"episode_index": episode_index, "stats": episode_statistics[episode_index]},
                    ensure_ascii=False,
                )
                + "\n"
            )

    with (target_root / "meta/steps_data_index.pkl").open("wb") as handle:
        pickle.dump(
            {
                "config_key": "downsampled-10hz",
                "steps": steps,
                "num_trajectories": len(episode_lengths),
                "total_steps": len(steps),
                "delete_pause_frame": False,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    _copy_videos(source_root, target_root, stride, fps, video_mode)
    _json_dump(
        target_root / "meta/downsample.json",
        {
            "source": str(source_root),
            "source_fps": source_info.get("fps"),
            "target_fps": fps,
            "stride": stride,
            "video_mode": video_mode,
            "video_note": (
                "Videos are source-FPS hard links; timestamp-based decoding selects sampled frames."
                if video_mode == "timestamp"
                else "Videos were physically re-encoded at target FPS."
            ),
        },
    )
    print(f"created {target_root}: {len(episode_lengths)} episodes, {sum(episode_lengths.values())} frames")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--video-mode", choices=("timestamp", "reencode"), default="timestamp")
    args = parser.parse_args()
    if args.stride <= 0 or args.fps <= 0:
        parser.error("--stride and --fps must be positive")
    if not args.input_root.is_dir():
        parser.error(f"input root does not exist: {args.input_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    datasets = sorted(path for path in args.input_root.iterdir() if path.is_dir())
    if not datasets:
        parser.error(f"no dataset directories found under {args.input_root}")
    for source_root in datasets:
        convert_dataset(
            source_root,
            args.output_root / source_root.name,
            args.stride,
            args.fps,
            args.video_mode,
        )


if __name__ == "__main__":
    main()
