"""Generic StarVLA WebSocket endpoint for an explicit EVA bridge contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer

from .bridge_contract import contract_sha256, execution_spec, load_contract, split_state
from .adapters.starvla import CheckpointAdapter
from .protocol import StreamSession

logger = logging.getLogger(__name__)


def _model_value(config: dict[str, Any], field: str) -> int | None:
    action_model = config["framework"]["action_model"]
    value = action_model.get(field)
    return int(value) if value is not None else None


class ContractPolicy:
    """Validate a contract's inputs, normalize state and invoke the stock wrapper."""

    def __init__(self, wrapper: PolicyServerWrapper, contract: dict[str, Any]) -> None:
        self.wrapper = wrapper
        self.contract = contract
        self._lock = threading.RLock()
        action = contract["action"]
        self.unnorm_key = str(action["unnorm_key"])
        if self.unnorm_key not in wrapper.metadata["available_unnorm_keys"]:
            raise ValueError(
                f"checkpoint lacks unnorm_key={self.unnorm_key!r}; "
                f"available={wrapper.metadata['available_unnorm_keys']}"
            )
        self.adapter = CheckpointAdapter(wrapper, self.unnorm_key)
        processor = self.adapter.processor
        actual_cameras = self.adapter.camera_keys
        expected_cameras = [camera["training_key"] for camera in contract["cameras"]]
        if not actual_cameras:
            raise ValueError("Checkpoint does not expose training camera order; add an explicit training adapter")
        if actual_cameras != expected_cameras:
            raise ValueError(f"contract camera order {expected_cameras} != training {actual_cameras}")
        expected_keys = action.get("training_keys")
        if expected_keys is not None and list(expected_keys) != processor.action_keys:
            raise ValueError(
                f"contract action keys {expected_keys} do not match checkpoint "
                f"keys {processor.action_keys}"
            )
        action_dim = sum(processor.action_key_dims[key] for key in processor.action_keys)
        if action_dim != int(action["dim"]):
            raise ValueError(f"contract action dim {action['dim']} != checkpoint {action_dim}")
        if "fields" in action:
            offset = 0
            for field in action["fields"]:
                width = processor.action_key_dims[field["training_key"]]
                if field["slice"] != [offset, offset + width]:
                    raise ValueError(f"Action slice/order disagrees with checkpoint: {field}")
                offset += width
        actual_image_hw = wrapper.metadata.get("training_obs_image_size")
        if actual_image_hw is not None and list(actual_image_hw) != contract["image"]["hw"]:
            raise ValueError(f"contract image size {contract['image']['hw']} != training {actual_image_hw}")
        state_contract = contract["state"]
        configured_state_dim = _model_value(self.adapter.config, "state_dim")
        if configured_state_dim is not None:
            expected_state_dim = state_contract["raw_dim"] if state_contract["include"] else 0
            if configured_state_dim != expected_state_dim:
                raise ValueError(f"checkpoint state_dim={configured_state_dim}, contract expects {expected_state_dim}")
        if state_contract.get("include", False):
            expected_state_keys = [field["training_key"] for field in state_contract["fields"]]
            if set(expected_state_keys) != set(processor.state_keys):
                raise ValueError(
                    f"contract state keys {expected_state_keys} do not match checkpoint "
                    f"keys {processor.state_keys}"
                )
            for field in state_contract["fields"]:
                key = field["training_key"]
                start, stop = map(int, field["slice"])
                if stop - start != processor.state_key_dims[key]:
                    raise ValueError(
                        f"raw state slice for {key} has dim {stop-start}; checkpoint expects "
                        f"{processor.state_key_dims[key]}"
                    )
        expected_model = contract.get("model", {})
        actual_horizon = _model_value(self.adapter.config, "action_horizon")
        actual_chunk = int(wrapper.metadata["action_chunk_size"])
        actual_steps = _model_value(self.adapter.config, "inference_steps")
        expected_checks = {
            "horizon": actual_horizon,
            "chunk_size": actual_chunk,
            "inference_steps": actual_steps,
        }
        for field, actual in expected_checks.items():
            expected = action.get(field, expected_model.get(field))
            if expected is not None and actual is not None and int(expected) != int(actual):
                raise ValueError(f"checkpoint {field}={actual}, contract expects {expected}")
        if actual_chunk != int(action["chunk_size"]):
            raise ValueError(
                f"checkpoint action chunk {actual_chunk}, contract expects {action['chunk_size']}"
            )
        expected_framework = expected_model.get("framework")
        actual_framework = self.adapter.config.get("framework", {}).get("name")
        if expected_framework and expected_framework != actual_framework:
            raise ValueError(
                f"checkpoint framework {actual_framework!r}, contract expects {expected_framework!r}"
            )
        has_rolling_cache = self.adapter.is_rolling
        if has_rolling_cache != (execution_spec(contract)["kind"] == "rolling"):
            raise ValueError("Contract scheduling kind differs from the checkpoint's rolling capability")

    def reset(self) -> None:
        with self._lock:
            self.wrapper.reset()

    def predict_action(self, examples: list[dict[str, Any]], unnorm_key: str | None = None,
                       trace: bool = False, **kwargs):
        allowed_kwargs = self.contract.get("inference_kwargs", {})
        if self.contract["protocol"].endswith("v2") and kwargs != allowed_kwargs:
            raise ValueError("Inference overrides must match the shared contract exactly")
        if unnorm_key is not None and str(unnorm_key) != self.unnorm_key:
            raise ValueError(f"request unnorm_key={unnorm_key!r}, expected {self.unnorm_key!r}")
        if not isinstance(examples, list) or len(examples) != 1:
            raise ValueError("EVA bridge currently requires one observation per request")
        contract = self.contract
        image_hw = tuple(map(int, contract["image"]["hw"]))
        camera_count = len(contract["cameras"])
        example = examples[0]
        image_list = example.get("image")
        if not isinstance(image_list, (list, tuple)) or len(image_list) != camera_count:
            raise ValueError(f"expected {camera_count} ordered images, got {type(image_list).__name__}")
        pil_images = []
        for index, image in enumerate(image_list):
            array = np.asarray(image)
            if array.dtype != np.uint8 or array.shape != (*image_hw, 3):
                raise ValueError(
                    f"camera {contract['cameras'][index]['eva_key']} expected uint8 "
                    f"{(*image_hw, 3)}, got {array.dtype}/{array.shape}"
                )
            pil_images.append(Image.fromarray(np.ascontiguousarray(array)))
        prompt = example.get("lang")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("example.lang must be a non-empty task instruction")
        model_example: dict[str, Any] = {"image": pil_images, "lang": prompt}
        state_contract = contract["state"]
        if state_contract.get("include", False):
            state_fields = split_state(
                example.get("state"),
                state_contract["fields"],
                int(state_contract["raw_dim"]),
            )
            model_example["state"] = self.adapter.normalize_state(state_fields)
        actions = self.wrapper.predict_action(
            [model_example], unnorm_key=self.unnorm_key, **kwargs
        )["actions"]
        actions = np.asarray(actions, dtype=np.float32)
        expected = (1, int(contract["action"]["chunk_size"]), int(contract["action"]["dim"]))
        if actions.shape != expected or not np.isfinite(actions).all():
            raise ValueError(f"model returned invalid action shape/value {actions.shape}, expected {expected}")
        result = {"actions": actions}
        if trace:
            result["diagnostics"] = {}
            if "state" in model_example:
                result["diagnostics"]["normalized_state"] = model_example["state"]
        return result


class ContractWebsocketPolicyServer(WebsocketPolicyServer):
    """Stock infer route plus an explicit per-episode reset route."""

    async def _handler(self, websocket):
        # A rolling checkpoint owns one cache. Reject a second robot stream
        # instead of mixing its cache with the active connection.
        if getattr(self, "_active_client", False):
            await websocket.close(code=1013, reason="Policy already has an active robot stream")
            return
        self._active_client = True
        try:
            self._session = StreamSession(self._policy.contract)
            self._metadata = {**self._metadata, "session_id": self._session.session_id,
                              "protocol": self._policy.contract["protocol"]}
            self._policy.reset()
            await super()._handler(websocket)
        finally:
            self._policy.reset()
            self._active_client = False

    def _route_message(self, msg: dict) -> dict:
        return self._session.route(msg, super()._route_message, self._reset_session)

    def _reset_session(self, payload):
        self._policy.reset()
        seed = payload.get("seed")
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))


def build_server(
    checkpoint: str,
    contract_path: str,
    host: str = "0.0.0.0",
    port: int = 10093,
    device: str = "cuda:0",
    use_bf16: bool = True,
) -> ContractWebsocketPolicyServer:
    contract = load_contract(contract_path)
    wrapper = PolicyServerWrapper(
        ckpt_path=checkpoint,
        device=device,
        use_bf16=use_bf16,
        unnorm_key=contract["action"]["unnorm_key"],
    )
    policy = ContractPolicy(wrapper, contract)
    checkpoint_file = Path(checkpoint).expanduser().resolve()
    source_identity = [{"path": str(path.resolve()),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                       for path in policy.adapter.source_paths]
    metadata = {
        **wrapper.metadata,
        "env": "starvla_eva_contract_server",
        "contract_name": contract["name"],
        "contract": contract,
        "contract_sha256": contract_sha256(contract),
        "action_chunk_size": int(contract["action"]["chunk_size"]),
        "stream_policy": "single_connection_with_reset_on_connect_and_disconnect",
        "capabilities": {"reset": True, "chunk_sequence": True, "trace": True,
                         "execution": execution_spec(contract)},
        "runtime_identity": {
            "checkpoint_file": {"path": str(checkpoint_file),
                                "size_bytes": checkpoint_file.stat().st_size,
                                "mtime_ns": checkpoint_file.stat().st_mtime_ns},
            "resolved_training_config_sha256": hashlib.sha256(json.dumps(
                policy.adapter.config, sort_keys=True, default=str).encode()).hexdigest(),
            "sources": source_identity,
        },
    }
    return ContractWebsocketPolicyServer(policy, host=host, port=port, metadata=metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--gpu-memory-gib", type=float, default=0,
                        help="Optional allocator limit for this evaluation process only")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.gpu_memory_gib:
        device = torch.device(args.device)
        if device.type != "cuda":
            parser.error("--gpu-memory-gib requires a CUDA device")
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_gib * 1024**3 / total, device)
    server = build_server(
        args.checkpoint,
        args.contract,
        args.host,
        args.port,
        args.device,
        use_bf16=not args.no_bf16,
    )
    logger.info("Contract policy server listening on %s:%d", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
