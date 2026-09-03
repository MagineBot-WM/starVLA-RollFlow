#!/usr/bin/env python3
"""Create G2-style, named G1 overlays without copying parquet or video data."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from agibot_g1_catalog import PUBLIC_TASKS, REAL_TASKS

DEFAULT_DATASETS_ROOT = Path("/data/tzq/datasets/starVLA/Datasets")
META_LINKS = ("info.json", "episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl")


def _replace_symlink(path: Path, target: Path) -> None:
    if path.is_symlink():
        if Path(os.readlink(path)) == target:
            return
        path.unlink()
    elif path.exists():
        raise FileExistsError(f"refusing to replace non-symlink: {path}")
    path.symlink_to(target)


def _write_json(path: Path, payload: dict) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def _correct_modality(source: Path) -> dict:
    modality = json.loads((source / "meta/modality.json").read_text())
    modality["state"]["waist"]["unit"] = "mixed_rad_m"
    modality["action"]["waist"]["unit"] = "mixed_rad_m"
    modality["action"]["grippers"]["unit"] = "normalized_opening_0_closed_1_open"
    return modality


def _make_overlay(destination: Path, source: Path, family: str, legacy_name: str) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    destination.mkdir(parents=True, exist_ok=True)
    meta = destination / "meta"
    meta.mkdir(exist_ok=True)

    _replace_symlink(destination / "data", source / "data")
    _replace_symlink(destination / "videos", source / "videos")
    for name in META_LINKS:
        _replace_symlink(meta / name, source / "meta" / name)

    source_stats = source / "meta/stats_gr00t.json"
    destination_stats = meta / "stats_gr00t.json"
    if not destination_stats.exists() and source_stats.exists():
        shutil.copy2(source_stats, destination_stats)

    _write_json(meta / "modality.json", _correct_modality(source))
    source_record = json.loads((source / "source.json").read_text())
    _write_json(
        destination / "source.json",
        {
            "source_overlay": str(source),
            "raw_source": source_record.get("source"),
            "source_family": family,
            "category": "manipulation",
            "legacy_name": legacy_name,
            "canonical_name": destination.name,
            "control_contract": {
                "action": "arms14 + waist2 + head2 + grippers2",
                "grippers": "continuous normalized opening; 0=closed, 1=open",
                "excluded": "mobile base velocity2 (public family only)",
            },
        },
    )


def prepare(datasets_root: Path) -> Path:
    collection = datasets_root / "AgiBot-G1-G2-StarVLA"
    canonical_root = collection / "g1/manipulation"
    public_root = collection / "g1"
    real_root = datasets_root / "my_real_g1_dataset-StarVLA"
    catalog = []

    for legacy_name, canonical_name in PUBLIC_TASKS.items():
        source = public_root / legacy_name
        destination = canonical_root / canonical_name
        _make_overlay(destination, source, "AgiBotWorld-Alpha", legacy_name)
        catalog.append({"name": canonical_name, "family": "public", "source": str(source)})

    for legacy_name, canonical_name in REAL_TASKS.items():
        source = real_root / legacy_name
        destination = canonical_root / canonical_name
        _make_overlay(destination, source, "my_real_g1_dataset", legacy_name)
        catalog.append({"name": canonical_name, "family": "real", "source": str(source)})

    _write_json(
        collection / "g1/catalog.json",
        {
            "layout": "g1/manipulation/<semantic_task_name>",
            "legacy_public_paths_preserved": True,
            "datasets": catalog,
        },
    )
    return canonical_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    args = parser.parse_args()
    destination = prepare(args.datasets_root.resolve())
    print(f"Prepared {len(PUBLIC_TASKS) + len(REAL_TASKS)} no-copy overlays under {destination}")


if __name__ == "__main__":
    main()
