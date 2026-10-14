"""Inspect deployment compatibility without inference or publishing robot actions."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

from .bridge_contract import contract_sha256, execution_spec, load_contract
from .runtime import DEFAULT_EVA_ROOT, load_config
from .protocol import validate_handshake


def source_identity(path: Path) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--eva-root", type=Path, default=DEFAULT_EVA_ROOT)
    parser.add_argument("--host", help="Optional server handshake check; needs an idle policy stream")
    parser.add_argument("--port", type=int, default=10094)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.config and not args.contract:
        parser.error("Pass --config or --contract")
    report = {"python": platform.python_version(), "action_publishing": False,
              "inference": False, "scope": "schema/API/handshake compatibility; no model-quality claim"}
    if args.config:
        config = load_config(args.config, args.eva_root)
        contract = load_contract(config.policy.backend_options.contract_path)
        from .adapters.eva import check_api
        report["eva_api"] = check_api()
        report["eva_sources"] = [source_identity(args.eva_root.resolve() / path) for path in (
            "src/core/config.py", "src/policy_client/base.py", "src/transport/zmq.py",
            "src/robots/base.py", "src/strategy/base_strategy.py",
        )]
        report["resolved_config"] = {
            "robot_type": config.robot.type, "transport": config.transport.type,
            "image_hw": [config.transport.image_height, config.transport.image_width],
            "source_color": "bgr" if config.transport.convert_bgr_to_rgb else "rgb",
            "execute_steps": execution_spec(contract)["execute_steps"],
            "control_hz": config.inference_cfg.publish_rate,
            "strategy": config.inference_strategies.rolling.type,
            "timing": execution_spec(contract).get("timing", "native_arrival"),
            "scheduling": execution_spec(contract).get("scheduling", "sync"),
            "prefetch_after_steps": execution_spec(contract).get("prefetch_after_steps"),
            "initial_qpos": contract.get("eva", {}).get("initial_qpos"),
            "max_frame_age_s": config.transport.get("max_frame_age_s"),
            "max_inference_latency_s": config.policy.backend_options.get("max_inference_latency_s"),
            "max_cold_start_latency_s": config.policy.backend_options.get("max_cold_start_latency_s"),
        }
        from .adapters.hardware import inspect_hardware_joint_order
        report["hardware_order"] = inspect_hardware_joint_order(args.eva_root, contract)
        if args.contract and contract_sha256(load_contract(args.contract)) != contract_sha256(contract):
            raise ValueError("--contract differs from the selected EVA config")
    else:
        contract = load_contract(args.contract)
    report["contract"] = contract
    report["contract_sha256"] = contract_sha256(contract)
    report["dependencies"] = {}
    for name in ("numpy", "Pillow", "websockets", "msgpack", "pyzmq"):
        try:
            report["dependencies"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["dependencies"][name] = None
    if args.host:
        import inspect
        import websockets.sync.client
        from . import msgpack_numpy
        options = {"compression": None, "open_timeout": 10, "max_size": None}
        if "proxy" in inspect.signature(websockets.sync.client.connect).parameters:
            options["proxy"] = None
        with websockets.sync.client.connect(f"ws://{args.host}:{args.port}", **options) as socket:
            metadata = msgpack_numpy.unpackb(socket.recv(timeout=10))
            validate_handshake(metadata, contract)
            report["server"] = metadata
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
