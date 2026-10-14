"""EVA registry plugin for ordered multi-camera StarVLA policy requests."""

from __future__ import annotations

import logging
import inspect
import threading
import json
import time
import uuid
import weakref
from pathlib import Path
from typing import Any

import numpy as np
import websockets.sync.client

from core.config import ConfigDict
from core.registry import POLICY_REGISTRY
from policy_client.base import PolicyBuildContext, PolicyClient, PolicyConnectionError, PolicyRequestError

from examples.deployment.eva.bridge_contract import contract_sha256, execution_spec, load_contract
from examples.deployment.eva import msgpack_numpy
from examples.deployment.eva.protocol import validate_handshake

logger = logging.getLogger(__name__)
_factory_clients = weakref.WeakValueDictionary()
_factory_lock = threading.Lock()


def _msgpack_numpy():
    return msgpack_numpy


@POLICY_REGISTRY.register_client("starvla_eva")
class StarVlaEvaPolicyClient(PolicyClient):
    """A configurable EVA policy client for the stock StarVLA checkpoint wrapper.

    This backend only formats observations and returns action chunks. Robot motion
    and chunk scheduling remain in EVA's existing transport/strategy layers.
    """

    def __init__(self, host: str, port: int, contract_path: str, *, timeout_s: float = 30.0,
                 trace_dir: str | None = None, trace_limit: int = 32,
                 max_inference_latency_s: float | None = None,
                 max_cold_start_latency_s: float | None = None):
        self.contract_path = Path(contract_path).expanduser().resolve()
        self.contract = load_contract(self.contract_path)
        self.timeout_s = float(timeout_s)
        if not np.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        self._lock = threading.RLock()
        self._request_index = 0
        self._sequence = 0
        self._episode = 0
        self._active_prompt = None
        self.last_chunk = None
        self._max_inference_latency_s = max_inference_latency_s
        if max_inference_latency_s is not None:
            self._max_inference_latency_s = float(max_inference_latency_s)
            if not np.isfinite(self._max_inference_latency_s) or self._max_inference_latency_s <= 0:
                raise ValueError("max_inference_latency_s must be finite and positive")
        self._max_cold_start_latency_s = self._max_inference_latency_s
        if max_cold_start_latency_s is not None:
            self._max_cold_start_latency_s = float(max_cold_start_latency_s)
            if not np.isfinite(self._max_cold_start_latency_s) or self._max_cold_start_latency_s <= 0:
                raise ValueError("max_cold_start_latency_s must be finite and positive")
        self._trace_dir = Path(trace_dir).expanduser() if trace_dir else None
        self._trace_limit = int(trace_limit)
        if self._trace_limit < 0:
            raise ValueError("trace_limit must be non-negative")
        self._trace_count = 0
        self._closed = False
        try:
            options = dict(compression=None, max_size=None, open_timeout=self.timeout_s)
            if "proxy" in inspect.signature(websockets.sync.client.connect).parameters:
                options["proxy"] = None
            self._ws = websockets.sync.client.connect(f"ws://{host}:{int(port)}", **options)
            self._codec = _msgpack_numpy()
            raw_metadata = self._ws.recv(timeout=self.timeout_s)
            self._metadata = self._codec.unpackb(raw_metadata)
        except Exception as error:
            self.close()
            raise PolicyConnectionError(f"Cannot connect to StarVLA server {host}:{port}") from error
        try:
            validate_handshake(self._metadata, self.contract)
        except ValueError as error:
            self.close()
            raise PolicyConnectionError(str(error)) from error
        self._session_id = self._metadata.get("session_id")

    @classmethod
    def from_config(cls, config: ConfigDict, ctx: PolicyBuildContext) -> "StarVlaEvaPolicyClient":
        options = config.backend_options
        contract_path = options.get("contract_path")
        if not contract_path:
            raise ValueError("policy.backend_options.contract_path is required")
        contract = load_contract(str(contract_path))
        if contract["action"]["mode"] != "eef" and ctx.action_dim != int(contract["action"]["dim"]):
            raise ValueError(f"EVA robot action dim {ctx.action_dim} != contract {contract['action']['dim']}")
        # Native force_reconnect builds the replacement BEFORE releasing the old
        # policy. Explicitly close this factory's prior connection to the same
        # endpoint so the single-stream server can accept the replacement.
        endpoint = (config.host, int(config.port))
        with _factory_lock:
            previous = _factory_clients.get(endpoint)
            if previous is not None:
                previous.close()
            client = cls(
                config.host, config.port, str(contract_path),
                timeout_s=float(options.get("timeout_s", 30.0)),
                trace_dir=options.get("trace_dir"),
                trace_limit=int(options.get("trace_limit", 32)),
                max_inference_latency_s=options.get("max_inference_latency_s"),
                max_cold_start_latency_s=options.get("max_cold_start_latency_s"),
            )
            _factory_clients[endpoint] = client
            return client

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
            if self._closed:
                raise PolicyRequestError("StarVLA connection is closed; reconnect and reset before continuing")
            request_id = f"eva-{self._request_index}-{uuid.uuid4().hex[:8]}"
            self._request_index += 1
            try:
                message = {"type": kind, "request_id": request_id, "payload": payload or {}}
                if self.contract["protocol"].endswith("v2"):
                    message["bridge"] = {
                        "session_id": self._session_id,
                        "contract_sha256": contract_sha256(self.contract),
                        "sequence": self._sequence,
                    }
                self._ws.send(
                    self._codec.Packer().pack(
                        message
                    )
                )
                response = self._codec.unpackb(self._ws.recv(timeout=self.timeout_s))
            except Exception as error:
                # A timeout can occur after the server advanced its rolling cache.
                # Never retry that uncertain request on the same stream.
                self.close()
                raise PolicyRequestError(f"StarVLA {kind} failed; stream closed to prevent rolling-cache drift") from error
        if not isinstance(response, dict) or response.get("request_id") != request_id:
            self.close()
            raise PolicyRequestError("StarVLA response request_id mismatch")
        if response.get("status") != "ok":
            self.close()
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
        if prev_action is not None or rtc_params is not None:
            raise ValueError("This deployment does not support RTC or implicit previous-action conditioning")
        cameras = self._ordered_images(observation["images"])
        prompt = observation.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Task instruction must be a non-empty string")
        example: dict[str, Any] = {"image": cameras, "lang": prompt}
        state_contract = self.contract["state"]
        if state_contract.get("include", False):
            state = np.asarray(observation.get("state"), dtype=np.float32)
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
        if self._trace_dir is not None:
            payload["trace"] = True
        # Serialize inference/reset across the full transaction, including index
        # validation and trace capture; sharing one stream cannot reorder chunks.
        with self._lock:
            if (execution_spec(self.contract)["kind"] == "rolling"
                    and self._active_prompt is not None and prompt != self._active_prompt):
                raise PolicyRequestError("Task changed within a rolling stream; reset before changing the prompt")
            sequence = self._sequence
            started = time.perf_counter()
            response = self._request("infer", payload)
            data = response.get("data", {})
            if not isinstance(data, dict):
                self.close()
                raise PolicyRequestError("StarVLA response data must be an object")
            if self.contract["protocol"].endswith("v2"):
                returned_sequence = data.get("sequence")
                if (isinstance(returned_sequence, bool) or not isinstance(returned_sequence, int)
                        or returned_sequence != sequence or data.get("session_id") != self._session_id):
                    self.close()
                    raise PolicyRequestError("StarVLA chunk sequence/session mismatch; stream closed")
            latency_ms = (time.perf_counter() - started) * 1000
            limit = self._max_cold_start_latency_s if sequence == 0 else self._max_inference_latency_s
            if limit is not None and latency_ms > 1000 * limit:
                self.close()
                raise PolicyRequestError(f"Inference took {latency_ms:.1f}ms; latency limit exceeded, reset/reconnect")
            try:
                actions = np.asarray(data.get("actions"), dtype=np.float32)
            except (ValueError, TypeError) as error:
                self.close()
                raise PolicyRequestError("StarVLA actions are not a numeric chunk; stream closed") from error
            expected = (int(self.contract["action"]["chunk_size"]), int(self.contract["action"]["dim"]))
            if actions.shape == (1, *expected):
                actions = actions[0]
            if actions.shape != expected or not np.isfinite(actions).all():
                self.close()
                raise PolicyRequestError(f"invalid action chunk: got {actions.shape}, expected {expected}")
            self._sequence += 1
            self._active_prompt = prompt
            self.last_chunk = dict(sequence=sequence, session_id=self._session_id,
                                   episode=self._episode, prompt=prompt, latency_ms=latency_ms)
            try:
                self._save_trace(sequence, example, actions, response, latency_ms)
            except Exception:
                logger.warning("Could not save deployment trace", exc_info=True)
            return {"actions": actions, "action_mode": self.contract["action"]["mode"]}

    def reset(self, *, seed: int | None = None) -> None:
        with self._lock:
            if self._closed:
                # EVA also resets during teardown after a connection failure.
                # No request can be sent; the next connection resets server cache.
                return
            response = self._request("reset", {"seed": seed} if seed is not None else {})
            data = response.get("data")
            if not isinstance(data, dict) or data.get("reset") is not True:
                self.close()
                raise PolicyRequestError("Server did not acknowledge cache reset")
            if self.contract["protocol"].endswith("v2") and (
                    data.get("session_id") != self._session_id or type(data.get("sequence")) is not int or data["sequence"] != 0):
                self.close()
                raise PolicyRequestError("Reset session/sequence mismatch")
            self._sequence = 0
            self._episode += 1
            self._active_prompt = None
            self.last_chunk = None

    def _save_trace(self, sequence, example, actions, response, latency_ms):
        if self._trace_dir is None or self._trace_count >= self._trace_limit:
            return
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{self._session_id or 'v1'}-{self._trace_count:04d}-{uuid.uuid4().hex[:8]}"
        arrays = {f"image_{i}": value for i, value in enumerate(example["image"])}
        if "state" in example:
            arrays["raw_state"] = example["state"]
        diagnostics = response.get("data", {}).get("diagnostics", {})
        if "normalized_state" in diagnostics:
            arrays["normalized_state"] = np.asarray(diagnostics["normalized_state"])
        arrays["raw_policy_actions"] = actions
        np.savez_compressed(self._trace_dir / f"{stem}.npz", **arrays)
        manifest = {
            "sequence": sequence, "session_id": self._session_id,
            "episode": self._episode,
            "contract_sha256": contract_sha256(self.contract), "contract": self.contract,
            "prompt": example["lang"], "latency_ms": latency_ms,
            "camera_order": [camera["eva_key"] for camera in self.contract["cameras"]],
            "actions_before_eva_processing": True,
        }
        (self._trace_dir / f"{stem}.json").write_text(json.dumps(manifest, indent=2) + "\n")
        self._trace_count += 1

    def close(self) -> None:
        self._closed = True
        websocket = getattr(self, "_ws", None)
        if websocket is not None:
            try:
                websocket.close()
            except Exception:
                logger.debug("Error closing StarVLA websocket", exc_info=True)

    def __del__(self):
        # EVA releases the policy reference on disconnect; it has no close hook.
        self.close()
