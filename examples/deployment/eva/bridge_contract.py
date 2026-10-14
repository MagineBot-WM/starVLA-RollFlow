"""CPU-only data contract shared by the robot client and the model server.

This describes meaning as well as shape. Normalization remains with the
checkpoint's training transforms on the server.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


PROTOCOLS = {"starvla_eva_contract_v1", "starvla_eva_contract_v2"}


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _fields(fields: Any, dim: int, name: str) -> None:
    if not isinstance(fields, list) or not fields:
        raise ValueError(f"{name} requires an ordered, non-empty fields list")
    seen: set[int] = set()
    keys: set[str] = set()
    for field in fields:
        if not isinstance(field, dict):
            raise ValueError(f"{name}: each field must be an object")
        key = field.get("training_key")
        bounds = field.get("slice")
        if not isinstance(key, str) or not key or key in keys:
            raise ValueError(f"{name}: missing or duplicate training_key {key!r}")
        if (not isinstance(bounds, list) or len(bounds) != 2
                or any(isinstance(v, bool) or not isinstance(v, int) for v in bounds)):
            raise ValueError(f"{name}.{key}.slice must be [start, stop] integers")
        start, stop = bounds
        if start < 0 or stop <= start or stop > dim:
            raise ValueError(f"{name}.{key}: invalid slice {bounds} for dimension {dim}")
        indices = set(range(start, stop))
        if seen & indices:
            raise ValueError(f"{name}: fields overlap at {key}")
        seen.update(indices)
        keys.add(key)
    if seen != set(range(dim)):
        raise ValueError(f"{name}: fields must cover all {dim} dimensions exactly once")


def execution_spec(contract: dict[str, Any]) -> dict[str, Any]:
    """V1 compatibility uses complete rolling chunks; V2 declares scheduling."""
    return contract.get("execution", {
        "kind": "rolling", "execute_steps": contract["action"]["chunk_size"],
    })


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(contract, dict):
        raise ValueError("contract must be a JSON object")
    required = {"name", "protocol", "robot", "cameras", "image", "state", "action", "model"}
    missing = sorted(required - set(contract))
    if missing:
        raise ValueError(f"bridge contract missing sections: {missing}")
    for name in ("name", "robot", "protocol"):
        if not isinstance(contract[name], str) or not contract[name]:
            raise ValueError(f"{name} must be a non-empty string")
    for name in ("image", "state", "action", "model"):
        if not isinstance(contract[name], dict):
            raise ValueError(f"{name} must be an object")
    if contract["protocol"] not in PROTOCOLS:
        raise ValueError(f"unsupported protocol {contract['protocol']!r}")
    cameras = contract["cameras"]
    if not isinstance(cameras, list) or not cameras:
        raise ValueError("bridge contract cameras must be a non-empty ordered list")
    if any(not isinstance(camera, dict) for camera in cameras):
        raise ValueError("each camera must be an object")
    for key in ("eva_key", "training_key"):
        values = [camera.get(key) for camera in cameras]
        if any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values):
            raise ValueError(f"camera {key} values must be non-empty and unique")
    image_hw = contract["image"].get("hw")
    if not isinstance(image_hw, list) or len(image_hw) != 2:
        raise ValueError("image.hw must be [height, width] with positive values")
    for value in image_hw:
        _positive_int(value, "image.hw")
    if contract["image"].get("color") != "rgb" or contract["image"].get("resize") != "direct":
        raise ValueError("this bridge requires explicit RGB images and direct resize")
    if contract["image"].get("interpolation", "bicubic") not in {"bicubic", "bilinear"}:
        raise ValueError("image.interpolation must be bicubic or bilinear")
    if not isinstance(contract["image"].get("rotate_180", False), bool):
        raise ValueError("image.rotate_180 must be boolean")
    action = contract["action"]
    for name in ("dim", "chunk_size"):
        _positive_int(action.get(name), f"action.{name}")
    if action.get("mode") not in {"joint", "qpos", "eef"}:
        raise ValueError("action.mode must be joint, qpos or eef")
    if not isinstance(action.get("unnorm_key"), str) or not action["unnorm_key"]:
        raise ValueError("action.unnorm_key is required")
    if "fields" in action:
        _fields(action["fields"], action["dim"], "action")
        if action.get("training_keys") != [f["training_key"] for f in action["fields"]]:
            raise ValueError("action.training_keys must equal the ordered action.fields keys")
    state = contract["state"]
    if not isinstance(state.get("include"), bool):
        raise ValueError("state.include must be boolean")
    if state["include"]:
        _fields(state.get("fields"), _positive_int(state.get("raw_dim"), "state.raw_dim"), "state")
    execution = execution_spec(contract)
    if not isinstance(execution, dict):
        raise ValueError("execution must be an object")
    steps = _positive_int(execution.get("execute_steps"), "execution.execute_steps")
    if execution.get("kind") not in {"rolling", "chunk"}:
        raise ValueError("execution.kind must be rolling or chunk")
    chunk = action["chunk_size"]
    if steps > chunk or (execution["kind"] == "rolling" and steps != chunk):
        raise ValueError("rolling must execute the complete returned chunk; chunk execution cannot exceed it")
    if "control_hz" in execution:
        hz = execution["control_hz"]
        if isinstance(hz, bool) or not isinstance(hz, (int, float)) or not math.isfinite(hz) or hz <= 0:
            raise ValueError("execution.control_hz must be finite and positive")
    if execution.get("timing", "native_arrival") not in {"fixed_rate", "native_arrival"}:
        raise ValueError("execution.timing must be fixed_rate or native_arrival")
    scheduling = execution.get("scheduling", "sync")
    if scheduling not in {"sync", "async_prefetch"}:
        raise ValueError("execution.scheduling must be sync or async_prefetch")
    if scheduling == "async_prefetch":
        trigger = _positive_int(execution.get("prefetch_after_steps"), "execution.prefetch_after_steps")
        if execution["kind"] != "rolling" or trigger >= steps:
            raise ValueError("async prefetch requires rolling and 0 < prefetch_after_steps < execute_steps")
        if execution.get("aggregation", "none") != "none":
            raise ValueError("Consecutive rolling chunks do not overlap; temporal aggregation is unsupported")
    model = contract["model"]
    if "horizon" in model and _positive_int(model["horizon"], "model.horizon") < chunk:
        raise ValueError("model.horizon cannot be smaller than the returned chunk")
    if "inference_steps" in model:
        _positive_int(model["inference_steps"], "model.inference_steps")
    kwargs = contract.get("inference_kwargs", {})
    if not isinstance(kwargs, dict) or {"examples", "unnorm_key", "trace"} & set(kwargs):
        raise ValueError("inference_kwargs cannot override examples, unnorm_key or trace")
    if contract["protocol"].endswith("v2"):
        if "execution" not in contract or not action.get("semantics") or "fields" not in action:
            raise ValueError("V2 requires execution, action.semantics and action.fields")
        for section in (action, state) if state["include"] else (action,):
            if any(not field.get("units") for field in section["fields"]):
                raise ValueError("V2 state/action fields require explicit units")
        binding = contract.get("eva", {})
        if not isinstance(binding, dict) or not binding.get("robot_type") or binding.get("source_color") not in {"rgb", "bgr"}:
            raise ValueError("V2 requires eva.robot_type and eva.source_color")
        names = binding.get("joint_names")
        pose = binding.get("initial_qpos")
        if pose is not None and (not isinstance(pose, list) or len(pose) != action["dim"]
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in pose)):
            raise ValueError("eva.initial_qpos must be a finite vector in the declared joint order")
        if names is not None and (not isinstance(names, list)
                or any(not isinstance(name, str) or not name for name in names)
                or len(names) != action["dim"] or len(set(names)) != action["dim"]):
            raise ValueError("eva.joint_names must name each output dimension exactly once")
        hardware_names = binding.get("hardware_joint_names")
        if binding.get("action_socket_role", "connect") not in {"bind", "connect"}:
            raise ValueError("eva.action_socket_role must be bind or connect")
        if binding.get("hardware_node") is not None:
            if (not isinstance(binding["hardware_node"], str) or not binding["hardware_node"]
                    or not isinstance(hardware_names, list) or len(hardware_names) != action["dim"]
                    or any(not isinstance(name, str) or not name for name in hardware_names)
                    or len(set(hardware_names)) != action["dim"]):
                raise ValueError("hardware_node requires a complete unique hardware_joint_names order")
        groups = binding.get("state_groups")
        if groups is not None:
            if not isinstance(groups, list):
                raise ValueError("eva.state_groups must be an ordered list")
            if not state["include"]:
                raise ValueError("eva.state_groups requires state.include")
            _fields([dict(training_key=group.get("name"), slice=group.get("slice"))
                     if isinstance(group, dict) else group for group in groups],
                    state["raw_dim"], "eva.state_groups")
            offset = 0
            for group in groups:
                if group["slice"][0] != offset:
                    raise ValueError("eva.state_groups must follow the raw vector order")
                offset = group["slice"][1]
        gripper = binding.get("gripper")
        if gripper is not None:
            if not isinstance(gripper, dict):
                raise ValueError("eva.gripper must be an object")
            for key in ("threshold", "open", "close"):
                value = gripper.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"eva.gripper.{key} must be finite")
    return contract


def load_contract(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open(encoding="utf-8") as stream:
        return validate_contract(json.load(stream))


def contract_sha256(contract: dict[str, Any]) -> str:
    """Hash canonical JSON so client and server reject mismatched contracts."""
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def split_state(raw_state: Any, fields: list[dict[str, Any]], raw_dim: int) -> dict[str, Any]:
    """Split a flat raw state vector into the training-time named state keys."""
    import numpy as np

    state = np.asarray(raw_state, dtype=np.float32)
    if state.shape != (int(raw_dim),) or not np.isfinite(state).all():
        raise ValueError(f"expected finite raw state ({raw_dim},), got {state.shape}")
    return {
        str(field["training_key"]): state[int(field["slice"][0]) : int(field["slice"][1])][None, :].copy()
        for field in fields
    }
