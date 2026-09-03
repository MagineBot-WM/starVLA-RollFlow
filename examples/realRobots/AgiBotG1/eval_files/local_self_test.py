#!/usr/bin/env python3
"""Fast hardware-free contract test for the AgiBot G1 evaluation adapter."""

import numpy as np
from agibot_g1_interface import CAMERA_ORDER, GripperCalibration, actions_to_robot_targets, build_observation


def main() -> None:
    images = {key: np.zeros((48, 64, 3), dtype=np.uint8) for key in CAMERA_ORDER}
    state = {
        "arms": np.zeros(14),
        "waist": np.zeros(2),
        "head": np.zeros(2),
        "grippers": np.array([20.0, 40.0]),
    }
    observation = build_observation(images, state, "pick the mango")
    assert list(observation["video"]) == list(CAMERA_ORDER)

    action = {
        "arms": np.zeros((1, 8, 14)),
        "waist": np.zeros((1, 8, 2)),
        "head": np.zeros((1, 8, 2)),
        "grippers": np.broadcast_to(np.array([0.0, 1.0]), (1, 8, 2)),
    }
    targets = actions_to_robot_targets(
        action,
        (GripperCalibration(2.0, 62.0), GripperCalibration(5.0, 105.0)),
    )
    np.testing.assert_allclose(targets["grippers_mm"][0], [2.0, 105.0])
    assert targets["arms"].shape == (8, 14)
    print("AgiBot G1 local contract self-test passed")


if __name__ == "__main__":
    main()
