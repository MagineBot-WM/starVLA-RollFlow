"""AgiBot G1 observation and action helpers for GR00T-compatible serving."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

CAMERA_ORDER = ("head", "hand_left", "hand_right")
STATE_DIMS = {"arms": 14, "waist": 2, "head": 2, "grippers": 2}
ACTION_DIMS = dict(STATE_DIMS)


@dataclass(frozen=True)
class GripperCalibration:
    """Physical opening range for one installed gripper, in millimetres."""

    closed_mm: float
    open_mm: float

    def opening_mm(self, normalized_opening: np.ndarray) -> np.ndarray:
        value = np.clip(np.asarray(normalized_opening, dtype=np.float32), 0.0, 1.0)
        return self.closed_mm + value * (self.open_mm - self.closed_mm)


def build_observation(
    images: Mapping[str, np.ndarray],
    state: Mapping[str, np.ndarray],
    instruction: str,
) -> dict:
    """Build the exact training-time wire contract in stable camera/key order."""

    missing_images = set(CAMERA_ORDER).difference(images)
    missing_state = set(STATE_DIMS).difference(state)
    if missing_images:
        raise KeyError(f"missing cameras: {sorted(missing_images)}")
    if missing_state:
        raise KeyError(f"missing state fields: {sorted(missing_state)}")

    video = {}
    for key in CAMERA_ORDER:
        image = np.asarray(images[key])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"camera {key!r} must be HWC RGB, got {image.shape}")
        video[key] = image[np.newaxis, np.newaxis]

    state_wire = {}
    for key, dim in STATE_DIMS.items():
        value = np.asarray(state[key], dtype=np.float32).reshape(-1)
        if value.size != dim:
            raise ValueError(f"state {key!r} must have {dim} values, got {value.size}")
        state_wire[key] = value.reshape(1, 1, dim)

    return {
        "video": video,
        "state": state_wire,
        "language": {"annotation.human.task_description": [[instruction]]},
    }


def actions_to_robot_targets(
    action: Mapping[str, np.ndarray],
    grippers: Sequence[GripperCalibration],
) -> dict[str, np.ndarray]:
    """Validate one action chunk and convert normalized grippers to mm targets."""

    if len(grippers) != 2:
        raise ValueError("exactly two gripper calibrations are required")
    result = {}
    chunk_length = None
    for key, dim in ACTION_DIMS.items():
        if key not in action:
            raise KeyError(f"missing action field {key!r}")
        value = np.asarray(action[key], dtype=np.float32)
        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 2 or value.shape[1] != dim:
            raise ValueError(f"action {key!r} must be [T,{dim}], got {value.shape}")
        chunk_length = value.shape[0] if chunk_length is None else chunk_length
        if value.shape[0] != chunk_length:
            raise ValueError("all action fields must have the same chunk length")
        result[key] = value

    normalized = result.pop("grippers")
    result["grippers_mm"] = np.stack([grippers[index].opening_mm(normalized[:, index]) for index in range(2)], axis=-1)
    return result
