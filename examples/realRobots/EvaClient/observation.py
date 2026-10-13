"""Strict adapter from EVA's raw G2 frame to the trained StarVLA input contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from PIL import Image

from .contract import CAMERA_ORDER, IMAGE_HW, STATE_DIM


def _validate_image(name: str, value: Any) -> np.ndarray:
    image = np.asarray(value)
    if image.dtype != np.uint8:
        raise TypeError(f"camera {name!r} must be uint8, got {image.dtype}")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"camera {name!r} must have HWC/3 shape, got {image.shape}")
    if image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError(f"camera {name!r} has an empty image: {image.shape}")
    return image


def prepare_policy_example(
    images_bgr: Mapping[str, np.ndarray], state_qpos: np.ndarray, prompt: str
) -> dict[str, Any]:
    """Convert raw EVA BGR/HWC images into training-time RGB PIL images.

    G2 EVA and DatasetTransport both expose BGR frames. Training reads RGB video
    frames and applies ``PIL.Image.fromarray(image).resize((224, 224))`` without
    letterboxing. Keeping this operation here makes live and replay paths identical.
    State is kept raw here; the server applies the checkpoint's G2 transform.
    """
    missing = [name for name in CAMERA_ORDER if name not in images_bgr]
    extra = sorted(set(images_bgr) - set(CAMERA_ORDER))
    if missing or extra:
        raise ValueError(f"G2 camera keys mismatch: missing={missing}, unexpected={extra}")

    model_images = []
    for name in CAMERA_ORDER:
        bgr = _validate_image(name, images_bgr[name])
        rgb = np.ascontiguousarray(bgr[..., ::-1])
        resized = Image.fromarray(rgb).resize((IMAGE_HW[1], IMAGE_HW[0]))
        model_images.append(np.asarray(resized, dtype=np.uint8).copy())

    state = np.asarray(state_qpos, dtype=np.float32).reshape(-1)
    if state.shape != (STATE_DIM,):
        raise ValueError(f"G2 state must have {STATE_DIM} joints, got shape {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("G2 state contains NaN or Inf")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("task prompt must be a non-empty string")

    return {
        "images_rgb": model_images,
        "state_qpos": state,
        "prompt": prompt.strip(),
    }
