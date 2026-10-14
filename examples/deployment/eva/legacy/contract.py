"""Verified G2 observation and rolling-inference contract for EVA integration."""

from __future__ import annotations

import hashlib
import json
from typing import Any

CAMERA_ORDER = ("top_head", "hand_left", "hand_right")
TRAINING_VIDEO_KEYS = {
    "top_head": "video.top_head",
    "hand_left": "video.hand_left",
    "hand_right": "video.hand_right",
}
CAMERA_SOURCE_KEYS = {
    "top_head": "top_head",
    "hand_left": "hand_left",
    "hand_right": "hand_right",
}
IMAGE_HW = (224, 224)
IMAGE_COLOR = "rgb"
RAW_EVA_COLOR = "bgr"
STATE_DIM = 24
ACTION_DIM = 24
HORIZON = 64
ACTION_CHUNK_SIZE = 16
REFINEMENT_STEPS = 4
UNNORM_KEY = "agibot_g2"

# Source observation.state/action names from the G2 LeRobot metadata, in order.
JOINT_NAMES = (
    "idx21_arm_l_joint1", "idx22_arm_l_joint2", "idx23_arm_l_joint3",
    "idx24_arm_l_joint4", "idx25_arm_l_joint5", "idx26_arm_l_joint6",
    "idx27_arm_l_joint7", "idx31_gripper_l_inner_joint1",
    "idx61_arm_r_joint1", "idx62_arm_r_joint2", "idx63_arm_r_joint3",
    "idx64_arm_r_joint4", "idx65_arm_r_joint5", "idx66_arm_r_joint6",
    "idx67_arm_r_joint7", "idx71_gripper_r_inner_joint1",
    "idx11_head_joint1", "idx12_head_joint2", "idx13_head_joint3",
    "idx01_body_joint1", "idx02_body_joint2", "idx03_body_joint3",
    "idx04_body_joint4", "idx05_body_joint5",
)
STATE_ACTION_SLICES = {
    "left_arm": (0, 7),
    "left_gripper": (7, 8),
    "right_arm": (8, 15),
    "right_gripper": (15, 16),
    "head": (16, 19),
    "waist": (19, 24),
}


def contract_payload() -> dict[str, Any]:
    """Return the canonical contract shared by the EVA client and policy server."""
    return {
        "protocol": "eva_starvla_rollflow_v1",
        "robot": "agibot_g2",
        "camera_order": list(CAMERA_ORDER),
        "training_video_keys": [TRAINING_VIDEO_KEYS[key] for key in CAMERA_ORDER],
        "raw_eva_color": RAW_EVA_COLOR,
        "model_image_color": IMAGE_COLOR,
        "image_resize": "PIL.Image.fromarray(rgb).resize((224,224))",
        "image_hw": list(IMAGE_HW),
        "joint_names": list(JOINT_NAMES),
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "state_mode": "raw qpos; normalized by saved G2 training transform on server",
        "unnorm_key": UNNORM_KEY,
        "horizon": HORIZON,
        "action_chunk_size": ACTION_CHUNK_SIZE,
        "refinement_steps": REFINEMENT_STEPS,
        "rolling_cache": "preserve across chunks; reset at each episode",
    }


def contract_sha256() -> str:
    encoded = json.dumps(contract_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def server_metadata(checkpoint: str, available_unnorm_keys: list[str]) -> dict[str, Any]:
    return {
        "env": "eva_starvla_rollflow_server",
        "checkpoint": checkpoint,
        "contract": contract_payload(),
        "contract_sha256": contract_sha256(),
        "available_unnorm_keys": available_unnorm_keys,
        "default_unnorm_key": UNNORM_KEY,
        "action_chunk_size": ACTION_CHUNK_SIZE,
        "horizon": HORIZON,
        "refinement_steps": REFINEMENT_STEPS,
    }
