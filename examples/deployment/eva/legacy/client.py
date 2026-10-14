"""Small synchronous MsgPack/WebSocket client for the additive EVA protocol."""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

import numpy as np
import websockets.sync.client

from .. import msgpack_numpy

from .contract import ACTION_CHUNK_SIZE, ACTION_DIM, contract_sha256
from .observation import prepare_policy_example


class EvaPolicyClient:
    """Connect to the parallel EVA RollFlow endpoint and return one qpos chunk."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 10094,
        *,
        connect_timeout_s: float = 30.0,
        expected_checkpoint: str | None = None,
    ) -> None:
        self._ws = websockets.sync.client.connect(
            f"ws://{host}:{int(port)}",
            compression=None,
            max_size=None,
            open_timeout=connect_timeout_s,
            ping_interval=None,
        )
        self._packer = msgpack_numpy.Packer()
        self._lock = threading.Lock()
        self._request_index = 0
        self.metadata = msgpack_numpy.unpackb(self._ws.recv())
        if self.metadata.get("contract_sha256") != contract_sha256():
            self.close()
            raise RuntimeError("EVA server/client G2 contract hashes do not match")
        if int(self.metadata.get("action_chunk_size", -1)) != ACTION_CHUNK_SIZE:
            self.close()
            raise RuntimeError(
                f"server emits {self.metadata.get('action_chunk_size')} actions; "
                f"G2 rolling client expects {ACTION_CHUNK_SIZE}"
            )
        if expected_checkpoint and self.metadata.get("checkpoint") != expected_checkpoint:
            self.close()
            raise RuntimeError(
                f"connected to checkpoint {self.metadata.get('checkpoint')!r}, "
                f"expected {expected_checkpoint!r}"
            )

    def _request(self, request_type: str, payload: dict[str, Any] | None = None) -> dict:
        with self._lock:
            request_id = f"eva-{self._request_index}-{uuid.uuid4().hex[:8]}"
            self._request_index += 1
            self._ws.send(
                self._packer.pack(
                    {"type": request_type, "request_id": request_id, "payload": payload or {}}
                )
            )
            response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"policy server returned a text error: {response}")
        decoded = msgpack_numpy.unpackb(response)
        if decoded.get("request_id") != request_id:
            raise RuntimeError("policy server response request_id mismatch")
        if decoded.get("status") != "ok":
            raise RuntimeError(f"policy server error: {decoded.get('error')}")
        return decoded

    def reset(self) -> None:
        """Clear rolling model state before beginning a new evaluation episode."""
        self._request("reset")

    def infer(self, images_bgr: dict[str, np.ndarray], state_qpos: np.ndarray, prompt: str) -> np.ndarray:
        """Return the next 16 unnormalized absolute-qpos commands, without publishing them."""
        example = prepare_policy_example(images_bgr, state_qpos, prompt)
        response = self._request(
            "infer_eva",
            {
                "images_rgb": example["images_rgb"],
                "state_qpos": example["state_qpos"],
                "prompt": example["prompt"],
            },
        )
        actions = np.asarray(response["data"]["actions"], dtype=np.float32)
        if actions.shape != (ACTION_CHUNK_SIZE, ACTION_DIM):
            raise ValueError(
                f"server action shape {actions.shape}, expected "
                f"({ACTION_CHUNK_SIZE}, {ACTION_DIM})"
            )
        if not np.isfinite(actions).all():
            raise ValueError("policy returned NaN or Inf actions")
        return actions

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass

    def __enter__(self) -> "EvaPolicyClient":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
