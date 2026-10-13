#!/usr/bin/env python3
"""Safely unpack the captured G2 LeRobot dataset and add its modality map."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path, PurePosixPath


DEFAULT_ARCHIVE = Path("/data/tzq/datasets/starVLA/Datasets/lerobot_eva_agibot_g2.zip")
DEFAULT_DATA_ROOT = Path("/data/tzq/datasets/starVLA/Datasets/AgiBotG2")
DATASET_NAME = "white_carton_to_left_tan_carton_20261009"
ZIP_ROOT = "lerobot_eva"

SLICES = {
    "left_arm": (0, 7),
    "left_gripper": (7, 8),
    "right_arm": (8, 15),
    "right_gripper": (15, 16),
    "head": (16, 19),
    "waist": (19, 24),
}
CAMERAS = {
    "top_head": "observation.images.top_head",
    "hand_left": "observation.images.hand_left",
    "hand_right": "observation.images.hand_right",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def modality_metadata() -> dict:
    state = {}
    action = {}
    for name, (start, end) in SLICES.items():
        state[name] = {
            "start": start,
            "end": end,
            "absolute": True,
            "dtype": "float32",
            "original_key": "observation.state",
        }
        action[name] = {
            "start": start,
            "end": end,
            "absolute": True,
            "dtype": "float32",
            "original_key": "action",
        }
    return {
        "state": state,
        "action": action,
        "video": {name: {"original_key": key} for name, key in CAMERAS.items()},
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"}
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    args = parser.parse_args()

    archive = args.archive.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    destination = data_root / args.dataset_name
    if not archive.is_file():
        raise FileNotFoundError(archive)
    archive_hash = sha256(archive)

    if os.path.lexists(destination):
        manifest_path = destination / "meta" / "g2_dataset_manifest.json"
        if not manifest_path.is_file():
            raise FileExistsError(
                f"Refusing to overwrite existing dataset path without our manifest: {destination}"
            )
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("archive_sha256") != archive_hash:
            raise RuntimeError(f"Existing dataset was prepared from a different archive: {destination}")
        print(f"Dataset already prepared and archive hash matches: {destination}")
        return

    data_root.mkdir(parents=True, exist_ok=True)
    staging = data_root / f".{args.dataset_name}.staging-{os.getpid()}"
    if os.path.lexists(staging):
        raise FileExistsError(staging)
    staging.mkdir()
    try:
        with zipfile.ZipFile(archive) as zipped:
            names = [item.filename for item in zipped.infolist()]
            unsafe = [
                name
                for name in names
                if PurePosixPath(name).is_absolute()
                or ".." in PurePosixPath(name).parts
                or (PurePosixPath(name).parts and PurePosixPath(name).parts[0] != ZIP_ROOT)
            ]
            if unsafe:
                raise ValueError(f"Unsafe or unexpected paths in archive: {unsafe[:5]}")
            bad = zipped.testzip()
            if bad is not None:
                raise RuntimeError(f"Archive CRC check failed at {bad}")
            zipped.extractall(staging)

        extracted_root = staging / ZIP_ROOT
        if not extracted_root.is_dir():
            raise RuntimeError(f"Expected dataset directory not found in archive: {extracted_root}")
        meta_dir = extracted_root / "meta"
        (meta_dir / "modality.json").write_text(
            json.dumps(modality_metadata(), indent=2) + "\n"
        )
        manifest = {
            "archive": str(archive),
            "archive_sha256": archive_hash,
            "zip_root": ZIP_ROOT,
            "dataset_name": args.dataset_name,
            "robot_type": "agibot_g2",
            "task": "Pick up the white carton and place it inside the large tan carton on the left.",
            "modality_json": "meta/modality.json",
            "note": "Generated modality mapping; the source zip remains unchanged.",
        }
        (meta_dir / "g2_dataset_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(destination):
            raise FileExistsError(destination)
        os.rename(extracted_root, destination)
        print(f"Prepared G2 dataset: {destination}")
        print(f"Source archive SHA-256: {archive_hash}")
        print(f"Generated metadata: {destination / 'meta' / 'modality.json'}")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
