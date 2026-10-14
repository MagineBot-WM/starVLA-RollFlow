"""Parallel G2-aware WebSocket server; the upstream StarVLA/EVA files stay untouched."""

from __future__ import annotations

import argparse
import logging
import threading
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from deployment.model_server.tools import msgpack_numpy
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer

from .contract import (
    ACTION_CHUNK_SIZE,
    ACTION_DIM,
    CAMERA_ORDER,
    HORIZON,
    IMAGE_HW,
    REFINEMENT_STEPS,
    STATE_ACTION_SLICES,
    STATE_DIM,
    UNNORM_KEY,
    contract_sha256,
    contract_payload,
    server_metadata,
)

logger = logging.getLogger(__name__)


def normalize_g2_state(raw_state: np.ndarray, processor: Any) -> np.ndarray:
    """Apply the checkpoint's exact DataConfig state transform to one raw qpos."""
    state = np.asarray(raw_state, dtype=np.float32).reshape(-1)
    if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
        raise ValueError(f"expected finite G2 state ({STATE_DIM},), got {state.shape}")
    data = {
        f"state.{name}": state[start:end][None, :].copy()
        for name, (start, end) in STATE_ACTION_SLICES.items()
    }
    transformed = processor.transform.apply(data)
    pieces = [
        transformed[key].detach().cpu().numpy()
        if isinstance(transformed[key], torch.Tensor)
        else np.asarray(transformed[key])
        for key in processor.state_keys
    ]
    result = np.concatenate(pieces, axis=-1).astype(np.float16)
    if result.shape != (1, STATE_DIM) or not np.isfinite(result).all():
        raise ValueError(f"training state transform returned invalid shape/value: {result.shape}")
    return result


class EvaPolicyAdapter:
    """Strict EVA input adapter around the stock StarVLA PolicyServerWrapper."""

    def __init__(self, wrapper: PolicyServerWrapper, unnorm_key: str = UNNORM_KEY) -> None:
        if unnorm_key not in wrapper.metadata["available_unnorm_keys"]:
            raise KeyError(
                f"checkpoint lacks {unnorm_key!r}; available keys: "
                f"{wrapper.metadata['available_unnorm_keys']}"
            )
        self.wrapper = wrapper
        self.unnorm_key = unnorm_key
        self.processor = wrapper.get_norm_processor(unnorm_key)
        self._lock = threading.RLock()

    def reset(self) -> None:
        with self._lock:
            self.wrapper.reset()

    def infer_eva(self, payload: dict[str, Any]) -> dict[str, np.ndarray]:
        images = payload.get("images_rgb")
        if not isinstance(images, (list, tuple)) or len(images) != len(CAMERA_ORDER):
            raise ValueError(f"expected {len(CAMERA_ORDER)} ordered RGB images")
        pil_images = []
        for camera, value in zip(CAMERA_ORDER, images, strict=True):
            array = np.asarray(value)
            if array.dtype != np.uint8 or array.shape != (*IMAGE_HW, 3):
                raise ValueError(
                    f"camera {camera} must be RGB uint8 {(*IMAGE_HW, 3)}, "
                    f"got {array.dtype}/{array.shape}"
                )
            pil_images.append(Image.fromarray(np.ascontiguousarray(array), mode="RGB"))

        state = normalize_g2_state(payload.get("state_qpos"), self.processor)
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        example = {"image": pil_images, "lang": prompt, "state": state}
        with self._lock:
            result = self.wrapper.predict_action([example], unnorm_key=self.unnorm_key)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.shape != (1, ACTION_CHUNK_SIZE, ACTION_DIM):
            raise ValueError(
                "checkpoint/server output must be [1,16,24], "
                f"got {actions.shape}; check rolling head and execution_horizon"
            )
        if not np.isfinite(actions).all():
            raise ValueError("model returned NaN or Inf actions")
        return {"actions": actions[0]}

    def predict_action(self, **_kwargs):
        raise ValueError("this endpoint accepts infer_eva only; use typed G2 observations")


class EvaWebsocketPolicyServer(WebsocketPolicyServer):
    """Add typed G2 routes while retaining the project's standard MsgPack framing."""

    def _route_message(self, msg: dict) -> dict:
        request_id = msg.get("request_id", "default")
        request_type = msg.get("type", "")
        payload = msg.get("payload", {})
        if request_type == "ping":
            return {"status": "ok", "type": "ping", "request_id": request_id}
        if request_type == "reset":
            try:
                self._policy.reset()
                return {"status": "ok", "request_id": request_id, "data": {"reset": True}}
            except Exception as exc:
                logger.exception("RollFlow reset failed")
                return {"status": "error", "request_id": request_id, "error": str(exc)}
        if request_type != "infer_eva":
            return {
                "status": "error",
                "request_id": request_id,
                "error": f"unsupported request type {request_type!r}; expected infer_eva/reset/ping",
            }
        if not isinstance(payload, dict):
            return {"status": "error", "request_id": request_id, "error": "payload must be a dict"}
        try:
            data = self._policy.infer_eva(payload)
            return {"status": "ok", "request_id": request_id, "data": data}
        except Exception as exc:
            logger.exception("EVA RollFlow inference failed")
            return {"status": "error", "request_id": request_id, "error": str(exc)}


def build_server(checkpoint: str, device: str, host: str, port: int, use_bf16: bool):
    wrapper = PolicyServerWrapper(
        ckpt_path=checkpoint,
        device=device,
        use_bf16=use_bf16,
        unnorm_key=UNNORM_KEY,
    )
    policy = EvaPolicyAdapter(wrapper)
    cfg = wrapper._model_cfg["framework"]["action_model"]
    actual = {
        "horizon": int(cfg["action_horizon"]),
        "action_chunk_size": int(wrapper.metadata["action_chunk_size"]),
        "refinement_steps": int(cfg.get("inference_steps", cfg.get("num_inference_timesteps", -1))),
    }
    expected = {
        "horizon": HORIZON,
        "action_chunk_size": ACTION_CHUNK_SIZE,
        "refinement_steps": REFINEMENT_STEPS,
    }
    if actual != expected:
        raise ValueError(f"checkpoint rolling contract mismatch: got {actual}, expected {expected}")
    metadata = server_metadata(
        checkpoint=str(Path(checkpoint).resolve()),
        available_unnorm_keys=wrapper.metadata["available_unnorm_keys"],
    )
    metadata["contract_sha256"] = contract_sha256()
    return EvaWebsocketPolicyServer(
        policy,
        host=host,
        port=port,
        metadata=metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10094)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-bf16", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    server = build_server(
        args.checkpoint,
        args.device,
        args.host,
        args.port,
        use_bf16=not args.no_bf16,
    )
    logger.info("G2 EVA RollFlow endpoint listening on %s:%d", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
