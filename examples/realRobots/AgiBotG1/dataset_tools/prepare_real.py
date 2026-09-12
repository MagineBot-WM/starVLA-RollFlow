#!/usr/bin/env python3
"""Prepare stationary native G1 data. Copy small parquet payloads; link videos.

Source files stay unchanged. Both flat and named base actions are zeroed before
statistics are computed. Existing destinations are never silently refreshed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def prepare(source: Path, destination: Path):
    source = source.resolve()
    destination = destination.resolve()
    if source == destination or source in destination.parents:
        raise ValueError("destination must be outside the source dataset")
    tasks = sorted(source.glob("*/meta/info.json"))
    if not tasks:
        raise ValueError(f"No datasets found under {source}")
    # Refuse replacement, including incomplete previous preparations.
    destination.mkdir(parents=True, exist_ok=False)
    for info_path in tasks:
        task = info_path.parent.parent
        target = destination / task.name
        meta = target / "meta"
        meta.mkdir(parents=True)
        info = json.loads(info_path.read_text())
        info["robot_type"] = "agibot-g1"
        info["features"]["observation.state"]["shape"] = [22]
        info["features"]["observation.state"]["names"] = None
        info["features"]["observation.state.base_velocity"] = {"dtype": "float32", "shape": [2], "names": ["vx", "wz"]}
        (meta / "info.json").write_text(json.dumps(info, indent=2) + "\n")
        for name in ("episodes.jsonl", "tasks.jsonl"):
            (meta / name).symlink_to(task / "meta" / name)
        (target / "videos").symlink_to(task / "videos", target_is_directory=True)
        fields = {"arms": (0, 14), "waist": (14, 16), "head": (16, 18), "grippers": (18, 20)}
        modality = {}
        for family, key in (("state", "observation.state"), ("action", "action")):
            slices = dict(fields)
            slices["base_velocity"] = (20, 22)
            modality[family] = {
                name: dict(start=start, end=end, original_key=key, absolute=True, dtype="float32")
                for name, (start, end) in slices.items()
            }
        modality["video"] = {
            slot: {"original_key": "observation.images." + camera}
            for slot, camera in zip(
                ("head", "hand_left", "hand_right"),
                ("cam_high_rgb", "cam_left_wrist_rgb", "cam_right_wrist_rgb"),
            )
        }
        modality["annotation"] = {"human.action.task_description": {"original_key": "task_index"}}
        (meta / "modality.json").write_text(json.dumps(modality, indent=2) + "\n")
        # Do not copy source statistics: base velocity has changed.
        rows = 0
        for path in sorted(task.glob("data/**/*.parquet")):
            table = pq.read_table(path)
            action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
            state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
            if action.shape != (len(table), 22) or state.shape != (len(table), 20):
                raise ValueError(f"Unexpected tensor shape: {path}")
            if not np.isfinite(action).all() or not np.isfinite(state).all():
                raise ValueError(f"Nonfinite values: {path}")
            if not np.isin(action[:, 18:20], [0, 1]).all():
                raise ValueError(f"Expected binary gripper commands: {path}")
            action[:, 20:22] = 0
            state = np.concatenate((state, np.zeros((len(table), 2), dtype=np.float32)), axis=1)
            index = table.schema.get_field_index("observation.state")
            # The old fixed-size list may be 20-wide; derive the 22-wide type.
            old_type = table.schema.field(index).type
            state_type = pa.list_(pa.float32(), 22) if pa.types.is_fixed_size_list(old_type) else pa.list_(pa.float32())
            table = table.set_column(index, "observation.state", pa.array(state.tolist(), type=state_type))
            table = table.append_column("observation.state.base_velocity", pa.array(state[:, 20:22].tolist(), type=pa.list_(pa.float32())))
            for key, values in (("action", action), ("action.robot_velocity", action[:, 20:22])):
                index = table.schema.get_field_index(key)
                if index < 0:
                    raise ValueError(f"Missing {key}: {path}")
                table = table.set_column(index, table.schema.field(index), pa.array(values.tolist(), type=table.schema.field(index).type))
            output = target / path.relative_to(task)
            output.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, output)
            rows += len(table)
        if rows != info["total_frames"]:
            raise ValueError(f"Frame count mismatch: {task}")
        print(f"{task.name}: {rows} frames, base actions zeroed")
    (destination / "source.json").write_text(json.dumps({
        "source": str(source), "stationary": True,
        "action_gripper": "0=open, 1=closed", "state_gripper": "encoder: 0=open, 120=closed",
        "note": "Raw source untouched; parquet copied, videos linked; recompute statistics here.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/data/tzq/datasets/starVLA/Datasets/AgiBot-G1-Real"))
    parser.add_argument("--destination", type=Path, default=Path("/data/tzq/datasets/starVLA/Datasets/agibot-g1"))
    args = parser.parse_args()
    prepare(args.source, args.destination)
