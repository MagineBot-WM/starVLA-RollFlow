"""Configurable contract shared by an EVA policy client and StarVLA server.

The contract makes camera order, state splitting, action semantics and rolling
chunk shape explicit. It deliberately does not assume that arbitrary robot
embodiments have interchangeable action spaces: each model/robot pairing needs
an explicit mapping here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_contract(path: str | Path) -> dict[str, Any]:
    """Load and validate a JSON bridge contract."""
    with Path(path).open(encoding="utf-8") as stream:
        contract = json.load(stream)
    required = {"name", "robot", "cameras", "image", "state", "action", "model"}
    missing = sorted(required - set(contract))
    if missing:
        raise ValueError(f"bridge contract missing sections: {missing}")
    cameras = contract["cameras"]
    if not isinstance(cameras, list) or not cameras:
        raise ValueError("bridge contract cameras must be a non-empty ordered list")
    eva_keys = [camera["eva_key"] for camera in cameras]
    training_keys = [camera["training_key"] for camera in cameras]
    if len(set(eva_keys)) != len(eva_keys) or len(set(training_keys)) != len(training_keys):
        raise ValueError("camera keys must be unique")
    image_hw = contract["image"].get("hw")
    if not isinstance(image_hw, list) or len(image_hw) != 2 or min(map(int, image_hw)) <= 0:
        raise ValueError("image.hw must be [height, width] with positive values")
    if contract["image"].get("color") != "rgb" or contract["image"].get("resize") != "direct":
        raise ValueError("this bridge requires explicit RGB images and direct resize")
    action = contract["action"]
    for name in ("dim", "chunk_size"):
        if int(action.get(name, 0)) <= 0:
            raise ValueError(f"action.{name} must be positive")
    if action.get("mode") not in {"joint", "qpos", "eef"}:
        raise ValueError("action.mode must be joint, qpos or eef")
    state = contract["state"]
    if state.get("include", False):
        raw_dim = int(state.get("raw_dim", 0))
        fields = state.get("fields", [])
        if raw_dim <= 0 or not fields:
            raise ValueError("included state needs raw_dim and explicit state.fields")
        seen: set[int] = set()
        for field in fields:
            start, stop = map(int, field["slice"])
            if not field.get("training_key") or start < 0 or stop <= start or stop > raw_dim:
                raise ValueError(f"invalid state field: {field}")
            indices = set(range(start, stop))
            if seen & indices:
                raise ValueError("state fields overlap")
            seen.update(indices)
        if seen != set(range(raw_dim)):
            raise ValueError("state fields must cover every raw state dimension exactly once")
    return contract


def contract_sha256(contract: dict[str, Any]) -> str:
    """Hash canonical JSON so client and server reject mismatched contracts."""
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def split_state(raw_state: Any, fields: list[dict[str, Any]], raw_dim: int) -> dict[str, Any]:
    """Split a flat raw state vector into the training-time named state keys."""
    import numpy as np

    state = np.asarray(raw_state, dtype=np.float32).reshape(-1)
    if state.shape != (int(raw_dim),) or not np.isfinite(state).all():
        raise ValueError(f"expected finite raw state ({raw_dim},), got {state.shape}")
    return {
        str(field["training_key"]): state[int(field["slice"][0]) : int(field["slice"][1])][None, :].copy()
        for field in fields
    }
