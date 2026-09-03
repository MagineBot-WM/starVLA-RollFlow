#!/usr/bin/env python3
"""Audit the two AgiBot G1 source families and their canonical overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from agibot_g1_catalog import ACTION_DIMS, CANONICAL_FIELDS, PUBLIC_TASKS, REAL_TASKS, STATE_DIMS

DEFAULT_DATASETS_ROOT = Path("/data/tzq/datasets/starVLA/Datasets")


def _vector(table, key: str) -> np.ndarray:
    return np.asarray(table[key].to_pylist(), dtype=np.float32)


def _sample_contract(dataset: Path, family: str) -> dict:
    modality = json.loads((dataset / "meta/modality.json").read_text())
    info = json.loads((dataset / "meta/info.json").read_text())
    parquet = sorted(dataset.glob("data/*/*.parquet"))[0]
    table = pq.read_table(parquet, columns=["observation.state", "action"])
    state = _vector(table, "observation.state")
    action = _vector(table, "action")

    expected_raw_action = 22 if family == "public" else 20
    errors = []
    if state.shape[1] != 20:
        errors.append(f"state dim is {state.shape[1]}, expected 20")
    if action.shape[1] != expected_raw_action:
        errors.append(f"raw action dim is {action.shape[1]}, expected {expected_raw_action}")
    for key in CANONICAL_FIELDS:
        if key not in modality["state"] or key not in modality["action"]:
            errors.append(f"missing canonical field {key}")
    if sum(STATE_DIMS.values()) != 20 or sum(ACTION_DIMS.values()) != 20:
        errors.append("canonical dimension table is invalid")

    gs = modality["state"]["grippers"]
    ga = modality["action"]["grippers"]
    measured = state[:, gs["start"] : gs["end"]]
    command = action[:, ga["start"] : ga["end"]]
    correlations = []
    lag = min(16, max(1, len(state) // 20))
    for index in range(2):
        x, y = command[:-lag, index], measured[lag:, index]
        correlations.append(None if x.std() == 0 or y.std() == 0 else float(np.corrcoef(x, y)[0, 1]))
    if command.min() < -1e-4 or command.max() > 1.0001:
        errors.append("gripper command falls outside normalized [0,1]")
    if not np.any((command > 0.01) & (command < 0.99)):
        errors.append("sample has no intermediate gripper opening commands")

    camera_shapes = {name: info["features"][spec["original_key"]]["shape"] for name, spec in modality["video"].items()}
    return {
        "dataset": str(dataset),
        "family": family,
        "fps": info.get("fps"),
        "episodes": info.get("total_episodes"),
        "frames": info.get("total_frames"),
        "raw_state_dim": int(state.shape[1]),
        "raw_action_dim": int(action.shape[1]),
        "model_action_dim": 20,
        "gripper_command_range": [float(command.min()), float(command.max())],
        "gripper_state_range_mm": [float(measured.min()), float(measured.max())],
        "gripper_future_correlations": correlations,
        "camera_shapes": camera_shapes,
        "errors": errors,
    }


def audit(datasets_root: Path) -> dict:
    collection = datasets_root / "AgiBot-G1-G2-StarVLA"
    public_root = collection / "g1"
    real_root = datasets_root / "my_real_g1_dataset-StarVLA"
    rows = []
    for legacy in PUBLIC_TASKS:
        rows.append(_sample_contract(public_root / legacy, "public"))
    for legacy in REAL_TASKS:
        rows.append(_sample_contract(real_root / legacy, "real"))

    canonical_root = collection / "g1/manipulation"
    expected_names = sorted([*PUBLIC_TASKS.values(), *REAL_TASKS.values()])
    missing = [name for name in expected_names if not (canonical_root / name).is_dir()]
    return {
        "summary": {
            "datasets": len(rows),
            "public_datasets": len(PUBLIC_TASKS),
            "real_datasets": len(REAL_TASKS),
            "all_fps_30": all(row["fps"] == 30 for row in rows),
            "errors": sum(len(row["errors"]) for row in rows),
            "missing_canonical_overlays": missing,
        },
        "inconsistencies": {
            "raw_action": "public=22D with base velocity; real=20D fixed base",
            "raw_state_order": "public=arms,grippers,head,waist; real=arms,waist,head,grippers",
            "gripper_state": "both measured in mm, but hardware opening ranges differ",
            "gripper_action": "both continuous normalized opening: 0=closed, 1=open",
            "camera_raw_keys": "different names, same semantic order: head,left wrist,right wrist",
        },
        "canonical_contract": {
            "state": "arms14 + waist2 + head2 + measured_grippers2",
            "action": "arms14 + waist2 + head2 + normalized_opening2",
            "excluded": "public mobile base velocity2",
        },
        "datasets": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.datasets_root.resolve())
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    if report["summary"]["errors"] or report["summary"]["missing_canonical_overlays"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
