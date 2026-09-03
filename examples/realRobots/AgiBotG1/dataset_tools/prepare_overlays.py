#!/usr/bin/env python3
"""Create G2-style, named G1 overlays without copying parquet or video data."""

from __future__ import annotations

import argparse
import json
import os
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
    # Both source families already carry this native column. Mapping the real
    # field keeps stationary local episodes valid while allowing future mobile
    # episodes without another schema migration.
    modality["action"]["base_velocity"] = {
        "start": 0,
        "end": 2,
        "absolute": True,
        "dtype": "float32",
        "original_key": "action.robot_velocity",
        "unit": "m_s_and_rad_s",
    }
    modality["action"].pop("base", None)
    return modality


def _load_raw_statistics(source: Path) -> dict:
    """Return a v2 stats cache that includes the separately stored base velocity."""

    stats_path = source / "meta/stats_gr00t.json"
    payload = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    statistics = payload.get("statistics", payload)
    if "action.robot_velocity" in statistics:
        return payload

    source_record = json.loads((source / "source.json").read_text())
    raw_source = Path(source_record["source"])
    raw_stats_path = raw_source / "meta/stats.json"
    if not raw_stats_path.exists():
        raise FileNotFoundError(f"statistics for action.robot_velocity are unavailable: {raw_stats_path}")
    raw_velocity = json.loads(raw_stats_path.read_text())["action.robot_velocity"]
    velocity_stats = {key: raw_velocity[key] for key in ("mean", "std", "min", "max")}
    # The original LeRobot stats omit percentiles. Base velocities are bounded
    # controller commands, so min/max are a safe cache fallback.
    velocity_stats["q01"] = raw_velocity["min"]
    velocity_stats["q99"] = raw_velocity["max"]
    statistics["action.robot_velocity"] = velocity_stats
    return {
        "__format_version": 2,
        "__cache_config": {"mode": "abs"},
        "statistics": statistics,
    }


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

    destination_stats = meta / "stats_gr00t.json"
    stats_payload = _load_raw_statistics(source)
    # Refresh on every preparation so newly added mobile episodes cannot keep
    # using a stale all-zero normalization cache.
    _write_json(destination_stats, stats_payload)

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
                "action": "arms14 + waist2 + head2 + grippers2 + base_velocity2",
                "grippers": "continuous normalized opening; 0=closed, 1=open",
                "base_velocity": "linear_x(m/s) + angular_z(rad/s)",
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
        catalog.append(
            {
                "name": canonical_name,
                "family": "real",
                "logical_task": "local_pick_object_place_pink_plate",
                "source": str(source),
            }
        )

    catalog_payload = {
        "layout": "g1/manipulation/<semantic_task_name>",
        "legacy_public_paths_preserved": True,
        "logical_tasks": 17,
        "datasets": catalog,
    }
    _write_json(collection / "g1/catalog.json", catalog_payload)

    # Keep the collection-level index consistent with the canonical G1 tree
    # while preserving every existing G2 entry.
    manifest_path = collection / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    g2_names = [name for name in manifest.get("datasets", []) if name.startswith("g2/")]
    g2_specs = [spec for spec in manifest.get("dataset_specs", []) if spec.get("name", "").startswith("g2/")]
    g1_names = [f"g1/manipulation/{item['name']}" for item in catalog]
    manifest["counts"]["g1"] = len(catalog)
    manifest["logical_task_counts"] = {"g1": 17, "g2": manifest["counts"]["g2"]}
    manifest["control_contracts"].pop("agibot_g1_mobile", None)
    manifest["control_contracts"]["agibot_g1"] = {
        "state": "arms14 + waist2 + head2 + measured grippers2(mm)",
        "action": "arms14 + waist2 + head2 + grippers2 + base velocity2",
        "grippers": "continuous normalized opening; 0=closed, 1=open",
        "base_velocity": "linear_x(m/s) + angular_z(rad/s)",
    }
    manifest["datasets"] = g1_names + g2_names
    manifest["dataset_specs"] = [
        {
            "name": f"g1/manipulation/{item['name']}",
            "source": item["source"],
            "logical_task": item.get("logical_task", item["name"]),
        }
        for item in catalog
    ] + g2_specs
    _write_json(manifest_path, manifest)
    return canonical_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    args = parser.parse_args()
    destination = prepare(args.datasets_root.resolve())
    print(f"Prepared {len(PUBLIC_TASKS) + len(REAL_TASKS)} no-copy overlays under {destination}")


if __name__ == "__main__":
    main()
