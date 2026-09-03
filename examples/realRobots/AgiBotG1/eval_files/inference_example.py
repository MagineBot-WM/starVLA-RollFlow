#!/usr/bin/env python3
"""Minimal AgiBot G1 GR00T-ZMQ client example.

Replace ``capture_*`` and ``send_targets`` with the robot SDK calls. Call the
server ``reset`` endpoint only at a task/episode boundary, not after each
8-action chunk; RollFlow deliberately carries its rolling cache across chunks.
"""

from __future__ import annotations

import argparse

import numpy as np
from agibot_g1_interface import GripperCalibration, actions_to_robot_targets, build_observation


def capture_images() -> dict[str, np.ndarray]:
    raise NotImplementedError("connect head, left-wrist, and right-wrist cameras")


def capture_state() -> dict[str, np.ndarray]:
    raise NotImplementedError("read arms14, waist2, head2, and measured grippers2")


def send_targets(targets: dict[str, np.ndarray]) -> None:
    raise NotImplementedError("send one 20-D absolute target through the AgiBot SDK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--left-closed-mm", type=float, default=0.0)
    parser.add_argument("--left-open-mm", type=float, required=True)
    parser.add_argument("--right-closed-mm", type=float, default=0.0)
    parser.add_argument("--right-open-mm", type=float, required=True)
    args = parser.parse_args()

    # Isaac-GR00T clients may expose a different import path. Keep this import
    # local so contract/unit tests do not require the client package.
    from gr00t.eval.service import ExternalRobotInferenceClient

    client = ExternalRobotInferenceClient(host=args.host, port=args.port)
    calibrations = (
        GripperCalibration(args.left_closed_mm, args.left_open_mm),
        GripperCalibration(args.right_closed_mm, args.right_open_mm),
    )

    client.reset()  # exactly once when a new task starts
    while True:
        observation = build_observation(capture_images(), capture_state(), args.instruction)
        action, _ = client.get_action(observation)
        targets = actions_to_robot_targets(action, calibrations)
        for step in range(targets["arms"].shape[0]):
            send_targets({key: value[step] for key, value in targets.items()})


if __name__ == "__main__":
    main()
