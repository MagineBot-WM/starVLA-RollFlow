#!/usr/bin/env python3
"""Create a canonical overlay for the Apple G1 export.

The source stores qpos as left arm(7), left gripper, right arm(7), right
gripper, head(2), waist(2). The StarVLA contract is arms(14), waist(2),
head(2), grippers(2), stationary base velocity(2).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# Raw export: left arm(7), left gripper, right arm(7), right gripper,
# head(2), waist(2).  The model-facing contract is arms14, waist2, head2,
# grippers2, base_velocity2.  Keep this permutation in one place so the
# parquet conversion and its audit cannot silently disagree.
ORDER = np.array([
    0, 1, 2, 3, 4, 5, 6,
    8, 9, 10, 11, 12, 13, 14,
    18, 19,  # waist
    16, 17,  # head
    7, 15,    # left/right grippers
])
CAMERAS = {
    "head": "observation.images.cam_high",
    "hand_left": "observation.images.cam_left_wrist",
    "hand_right": "observation.images.cam_right_wrist",
}


def _modality() -> dict:
    fields = {"arms": (0, 14), "waist": (14, 16), "head": (16, 18), "grippers": (18, 20), "base_velocity": (20, 22)}
    out = {}
    for family, original in (("state", "observation.state"), ("action", "action")):
        out[family] = {
            key: {"start": start, "end": end, "original_key": original, "absolute": True, "dtype": "float32"}
            for key, (start, end) in fields.items()
        }
    out["video"] = {slot: {"original_key": key} for slot, key in CAMERAS.items()}
    out["annotation"] = {"human.action.task_description": {"original_key": "task_index"}}
    return out


def prepare(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination = destination.resolve()
    if source == destination or source in destination.parents:
        raise ValueError("destination must be outside source")
    if not (source / "meta/info.json").is_file():
        raise FileNotFoundError(f"not a LeRobot v2 dataset: {source}")
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing path: {destination}")

    destination.mkdir(parents=True)
    (destination / "meta").mkdir()
    info = json.loads((source / "meta/info.json").read_text())
    info["robot_type"] = "agibot-g1"
    info["features"]["observation.state"] = {"dtype": "float32", "shape": [22], "names": ["arms", "waist", "head", "grippers", "base_velocity"]}
    info["features"]["action"] = {"dtype": "float32", "shape": [22], "names": ["arms", "waist", "head", "grippers", "base_velocity"]}
    (destination / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    for name in ("episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl", "quality_split.json"):
        src = source / "meta" / name
        if src.exists():
            (destination / "meta" / name).symlink_to(src)
    (destination / "meta/modality.json").write_text(json.dumps(_modality(), indent=2) + "\n")
    (destination / "videos").symlink_to(source / "videos", target_is_directory=True)

    total = 0
    for src in sorted(source.glob("data/**/*.parquet")):
        table = pq.read_table(src)
        state = np.asarray(table["observations.state.qpos"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["action.qpos"].to_pylist(), dtype=np.float32)
        if state.ndim != 2 or state.shape[1] != 20 or action.shape != state.shape:
            raise ValueError(f"unexpected qpos shape in {src}: {state.shape}, {action.shape}")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"non-finite qpos in {src}")
        state = np.concatenate((state[:, ORDER], np.zeros((len(state), 2), np.float32)), axis=1)
        action = np.concatenate((action[:, ORDER], np.zeros((len(action), 2), np.float32)), axis=1)
        table = table.append_column("observation.state", pa.array(state.tolist(), type=pa.list_(pa.float32(), 22)))
        table = table.append_column("action", pa.array(action.tolist(), type=pa.list_(pa.float32(), 22)))
        out = destination / src.relative_to(source)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out)
        total += len(table)

    if total != int(info["total_frames"]):
        raise ValueError(f"frame count mismatch: {total} != {info['total_frames']}")
    (destination / "source.json").write_text(json.dumps({"source": str(source), "qpos_order": "left arm7,left gripper,right arm7,right gripper,head2,waist2", "canonical_order": "arms14,waist2,head2,grippers2,base_velocity2", "base_velocity": "stationary zeros", "action_semantics": "absolute qpos target"}, indent=2) + "\n")
    print(f"created {destination}: {total} frames from {source}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("/data/tzq/datasets/starVLA/Datasets/pick_up_the_apple"))
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("/data/tzq/datasets/starVLA/Datasets/agibot-g1-apple-corrected"),
    )
    args = parser.parse_args()
    prepare(args.source, args.destination)
