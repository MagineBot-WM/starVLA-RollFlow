#!/usr/bin/env python3
"""Verify the Apple G1 raw-to-canonical state/action permutation.

This audit intentionally compares every parquet row instead of checking only
shapes or state/action equality.  The latter can pass even when two semantic
fields have been swapped in both arrays.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from prepare_apple import ORDER


def _read(path: Path, *columns: str) -> dict[str, np.ndarray]:
    table = pq.read_table(path, columns=list(columns))
    return {
        column: np.asarray(table[column].to_pylist(), dtype=np.float32)
        for column in columns
    }


def audit(source: Path, overlay: Path) -> dict[str, float | int]:
    source = source.resolve()
    overlay = overlay.resolve()
    source_files = sorted(source.glob("data/**/*.parquet"))
    if not source_files:
        raise FileNotFoundError(f"no parquet files under {source / 'data'}")

    frame_count = 0
    max_state_error = 0.0
    max_action_error = 0.0
    max_base_velocity = 0.0
    for source_file in source_files:
        overlay_file = overlay / source_file.relative_to(source)
        if not overlay_file.is_file():
            raise FileNotFoundError(overlay_file)

        raw = _read(source_file, "observations.state.qpos", "action.qpos")
        canonical = _read(overlay_file, "observation.state", "action")
        state = raw["observations.state.qpos"]
        action = raw["action.qpos"]
        expected_state = np.concatenate(
            (state[:, ORDER], np.zeros((len(state), 2), dtype=np.float32)), axis=1
        )
        expected_action = np.concatenate(
            (action[:, ORDER], np.zeros((len(action), 2), dtype=np.float32)), axis=1
        )
        if state.shape != action.shape or state.ndim != 2 or state.shape[1] != 20:
            raise AssertionError(f"unexpected raw shapes in {source_file}")
        if canonical["observation.state"].shape != expected_state.shape:
            raise AssertionError(f"unexpected state shape in {overlay_file}")
        if canonical["action"].shape != expected_action.shape:
            raise AssertionError(f"unexpected action shape in {overlay_file}")

        max_state_error = max(
            max_state_error,
            float(np.max(np.abs(canonical["observation.state"] - expected_state))),
        )
        max_action_error = max(
            max_action_error,
            float(np.max(np.abs(canonical["action"] - expected_action))),
        )
        max_base_velocity = max(
            max_base_velocity,
            float(np.max(np.abs(canonical["action"][:, 20:22]))),
            float(np.max(np.abs(canonical["observation.state"][:, 20:22]))),
        )
        frame_count += len(state)

    modality = overlay / "meta/modality.json"
    if not modality.is_file():
        raise FileNotFoundError(modality)
    import json

    fields = json.loads(modality.read_text())
    expected_slices = {
        "arms": (0, 14),
        "waist": (14, 16),
        "head": (16, 18),
        "grippers": (18, 20),
        "base_velocity": (20, 22),
    }
    for family in ("state", "action"):
        for key, (start, end) in expected_slices.items():
            actual = fields[family][key]
            if (actual["start"], actual["end"]) != (start, end):
                raise AssertionError(f"{family}.{key} has an incorrect slice")

    return {
        "parquet_files": len(source_files),
        "frames": frame_count,
        "max_state_error": max_state_error,
        "max_action_error": max_action_error,
        "max_base_velocity": max_base_velocity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source, args.overlay)
    print(result)
    if result["max_state_error"] or result["max_action_error"] or result["max_base_velocity"]:
        raise SystemExit("canonical mapping audit failed")


if __name__ == "__main__":
    main()
