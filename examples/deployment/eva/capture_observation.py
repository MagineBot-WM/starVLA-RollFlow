"""Capture live model inputs without loading a model or creating action sockets."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .runtime import DEFAULT_EVA_ROOT, load_config
from .bridge_contract import contract_sha256, load_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eva-root", type=Path, default=DEFAULT_EVA_ROOT)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/agibot_g2_live.py")
    parser.add_argument("--obs-endpoint")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.count <= 0 or not np.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("count and timeout-s must be positive")
    config = load_config(args.config, args.eva_root)
    if args.obs_endpoint:
        config.transport.sub_endpoint = args.obs_endpoint
    from core.registry import ROBOT_REGISTRY
    from core.app.handlers.recording import build_policy_observation
    from core.app.state import RuntimeState
    from .eva_extension.image_transport import ReadOnlyContractObservation
    from .eva_extension.starvla_policy import StarVlaEvaPolicyClient
    from core.types import Observation
    contract = load_contract(config.policy.backend_options.contract_path)
    robot = ROBOT_REGISTRY.build(config.robot.type)
    source = ReadOnlyContractObservation(config, robot)
    runtime = RuntimeState(robot=robot, transport=source)
    # Only the pure image-ordering method is used, without a websocket client.
    order = StarVlaEvaPolicyClient.__new__(StarVlaEvaPolicyClient)
    order.contract = contract
    arrays, rows = {}, []
    deadline, previous = time.monotonic()+args.timeout_s, None
    try:
        while len(rows) < args.count and time.monotonic() < deadline:
            snapshot = source._checked_snapshot()
            if snapshot is None:
                time.sleep(.01)
                continue
            wire, state = snapshot
            if wire.t == previous:
                time.sleep(.01)
                continue
            previous = wire.t
            frame = source._prepare_frame(Observation(images=wire.images, state_qpos=state, timestamp=wire.t))
            observation = build_policy_observation(frame, "observation capture", config, runtime)
            images = order._ordered_images(observation["images"])
            index = len(rows)
            arrays[f"sample_{index}_raw_state"] = state
            for i, camera in enumerate(contract["cameras"]):
                arrays[f"sample_{index}_source_{camera['eva_key']}"] = wire.images[camera["eva_key"]]
                arrays[f"sample_{index}_model_image_{i}"] = images[i]
            rows.append(dict(source_timestamp=wire.t, client_monotonic_s=time.monotonic(),
                             raw_state=state.tolist()))
        if len(rows) != args.count:
            raise TimeoutError(f"Captured {len(rows)}/{args.count} complete advancing observations")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
        args.output.write_text(json.dumps(dict(contract=contract, contract_sha256=contract_sha256(contract),
            samples=rows, physical_robot_actions=False, model_inference=False,
            action_socket_created=False, numpy_artifact=str(args.output.with_suffix(".npz"))), indent=2)+"\n")
        print(f"Captured {len(rows)} observations; no model calls or action sockets")
    finally:
        source.close()


if __name__ == "__main__":
    main()
