"""EVA registry plugin for ordered multi-camera StarVLA policy requests."""

from __future__ import annotations

import logging
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import websockets.sync.client

from core.config import ConfigDict
from core.registry import POLICY_REGISTRY
from policy_client.base import PolicyBuildContext, PolicyClient, PolicyConnectionError, PolicyRequestError

from examples.realRobots.EvaClient.bridge_contract import contract_sha256, load_contract

logger = logging.getLogger(__name__)


def _msgpack_numpy():
    try:
        from openpi_client import msgpack_numpy
    except ImportError:
        from deployment.model_server.tools import msgpack_numpy
    return msgpack_numpy


@POLICY_REGISTRY.register_client("starvla_eva")
class StarVlaEvaPolicyClient(PolicyClient):
    """A configurable EVA policy client for the stock StarVLA checkpoint wrapper.

    This backend only formats observations and returns action chunks. Robot motion
    and chunk scheduling remain in EVA's existing transport/strategy layers.
    """

    def __init__(self, host: str, port: int, contract_path: str, *, timeout_s: float = 30.0):
        self.contract_path = Path(contract_path).expanduser().resolve()
        self.contract = load_contract(self.contract_path)
        self.timeout_s = float(timeout_s)
        try:
            self._ws = websockets.sync.client.connect(
                f"ws://{host}:{int(port)}",
                compression=None,
                max_size=None,
                open_timeout=float(timeout_s),
            )
            self._codec = _msgpack_numpy()
            raw_metadata = self._ws.recv(timeout=self.timeout_s)
            self._metadata = self._codec.unpackb(raw_metadata)
        except Exception as error:
            raise PolicyConnectionError(f"Cannot connect to StarVLA server {host}:{port}") from error
        self._lock = threading.Lock()
        self._request_index = 0
        expected_hash = contract_sha256(self.contract)
        if self._metadata.get("contract_sha256") != expected_hash:
            self.close()
            raise PolicyConnectionError(
                "StarVLA server/client bridge-contract hash mismatch; "
                f"client={expected_hash}, server={self._metadata.get('contract_sha256')}"
            )
        expected_action = self.contract["action"]
        if int(self._metadata.get("action_chunk_size", -1)) != int(expected_action["chunk_size"]):
            self.close()
            raise PolicyConnectionError(
                f"server chunk size {self._metadata.get('action_chunk_size')} does not match "
                f"contract {expected_action['chunk_size']}"
            )
        available = self._metadata.get("available_unnorm_keys", [])
        if expected_action["unnorm_key"] not in available:
            self.close()
            raise PolicyConnectionError(
                f"server lacks unnorm key {expected_action['unnorm_key']!r}; available={available}"
            )

    @classmethod
    def from_config(cls, config: ConfigDict, ctx: PolicyBuildContext) -> "StarVlaEvaPolicyClient":
        options = config.backend_options
        contract_path = options.get("contract_path")
        if not contract_path:
            raise ValueError("policy.backend_options.contract_path is required")
        contract = load_contract(str(contract_path))
        if ctx.action_dim != int(contract["action"]["dim"]):
            raise ValueError(f"EVA robot action dim {ctx.action_dim} != contract {contract['action']['dim']}")
        return cls(
            config.host,
            config.port,
            str(contract_path),
            timeout_s=float(options.get("timeout_s", 30.0)),
        )

    @property
    def metadata(self) -> dict:
        return {
            **self._metadata,
            "server_name": "starvla-eva-contract",
            "action_mode": self.contract["action"]["mode"],
            "action_dim": int(self.contract["action"]["dim"]),
            "contract_name": self.contract["name"],
        }

    def _request(self, kind: str, payload: dict[str, Any] | None = None) -> dict:
        with self._lock:
            request_id = f"eva-{self._request_index}-{uuid.uuid4().hex[:8]}"
            self._request_index += 1
            try:
                self._ws.send(
                    self._codec.Packer().pack(
                        {"type": kind, "request_id": request_id, "payload": payload or {}}
                    )
                )
                response = self._codec.unpackb(self._ws.recv(timeout=self.timeout_s))
            except Exception as error:
                raise PolicyRequestError(f"StarVLA {kind} request failed") from error
        if response.get("request_id") != request_id:
            raise PolicyRequestError("StarVLA response request_id mismatch")
        if response.get("status") != "ok":
            message = response.get("error", {})
            raise PolicyRequestError(f"StarVLA server error: {message}")
        return response

    def _ordered_images(self, images: dict[str, Any]) -> list[np.ndarray]:
        expected_hw = tuple(map(int, self.contract["image"]["hw"]))
        ordered: list[np.ndarray] = []
        for camera in self.contract["cameras"]:
            key = camera["eva_key"]
            if key not in images:
                raise ValueError(f"EVA observation missing configured camera {key!r}")
            image = np.asarray(images[key])
            if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
                image = np.transpose(image, (1, 2, 0))
            if image.dtype != np.uint8 or image.shape != (*expected_hw, 3):
                raise ValueError(
                    f"camera {key} expected RGB uint8 {(*expected_hw, 3)}, "
                    f"got {image.dtype}/{image.shape}; check EVA image config"
                )
            if self.contract["image"].get("rotate_180", False):
                image = image[::-1, ::-1]
            ordered.append(np.ascontiguousarray(image))
        return ordered

    def infer(
        self,
        observation: dict,
        prev_action: np.ndarray | None = None,
        rtc_params: dict | None = None,
    ) -> dict:
        _ = prev_action, rtc_params
        cameras = self._ordered_images(observation["images"])
        example: dict[str, Any] = {"image": cameras, "lang": str(observation["prompt"])}
        state_contract = self.contract["state"]
        if state_contract.get("include", False):
            state = np.asarray(observation.get("state"), dtype=np.float32).reshape(-1)
            if state.shape != (int(state_contract["raw_dim"]),) or not np.isfinite(state).all():
                raise ValueError(
                    f"state expected finite ({state_contract['raw_dim']},), got {state.shape}"
                )
            example["state"] = state
        payload = {
            "examples": [example],
            "unnorm_key": self.contract["action"]["unnorm_key"],
            **self.contract.get("inference_kwargs", {}),
        }
        response = self._request("infer", payload)
        actions = np.asarray(response.get("data", {}).get("actions"), dtype=np.float32)
        expected = (int(self.contract["action"]["chunk_size"]), int(self.contract["action"]["dim"]))
        if actions.ndim == 3:
            if actions.shape[0] != 1:
                raise PolicyRequestError(f"expected one output batch, got {actions.shape}")
            actions = actions[0]
        if actions.shape != expected or not np.isfinite(actions).all():
            raise PolicyRequestError(f"invalid action chunk: got {actions.shape}, expected {expected}")
        return {"actions": actions, "action_mode": self.contract["action"]["mode"]}

    def reset(self, *, seed: int | None = None) -> None:
        self._request("reset", {"seed": seed} if seed is not None else {})

    def close(self) -> None:
        websocket = getattr(self, "_ws", None)
        if websocket is not None:
            try:
                websocket.close()
            except Exception:
                logger.debug("Error closing StarVLA websocket", exc_info=True)
