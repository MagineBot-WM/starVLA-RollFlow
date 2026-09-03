import importlib.util
import sys
from pathlib import Path

import numpy as np

from starVLA.dataloader.gr00t_lerobot.embodiment_tags import (
    EMBODIMENT_TAG_MAPPING,
    EmbodimentTag,
)

ROOT = Path(__file__).resolve().parents[1]


def _load(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_agibot_g1_data_contract_and_balanced_mixture():
    module = _load(
        "agibot_g1_data_config_test",
        "examples/realRobots/AgiBotG1/train_files/data_registry/data_config.py",
    )
    config = module.AgiBotG1DataConfig()

    assert config.embodiment_tag is EmbodimentTag.AGIBOT_G1
    assert sum(config.state_key_dims.values()) == 20
    assert sum(config.action_key_dims.values()) == 20
    assert config.action_indices == list(range(32))
    assert config.video_keys == ["video.head", "video.hand_left", "video.hand_right"]

    mixture = module.DATASET_NAMED_MIXTURES["agibot_g1_all"]
    public_weight = sum(weight for name, weight, _ in mixture if "pick_" not in name or "place_pink" not in name)
    real_weight = sum(weight for name, weight, _ in mixture if "place_pink_plate" in name)
    assert public_weight == real_weight == 3.0


def test_agibot_g1_has_stable_embedding_id():
    assert EMBODIMENT_TAG_MAPPING[EmbodimentTag.AGIBOT_G1.value] == 11


def test_agibot_g1_eval_contract_and_gripper_calibration():
    module = _load(
        "agibot_g1_eval_interface_test",
        "examples/realRobots/AgiBotG1/eval_files/agibot_g1_interface.py",
    )
    images = {key: np.zeros((32, 48, 3), dtype=np.uint8) for key in module.CAMERA_ORDER}
    state = {
        "arms": np.zeros(14),
        "waist": np.zeros(2),
        "head": np.zeros(2),
        "grippers": np.zeros(2),
    }
    obs = module.build_observation(images, state, "test")
    assert list(obs["video"]) == ["head", "hand_left", "hand_right"]

    actions = {
        "arms": np.zeros((1, 8, 14)),
        "waist": np.zeros((1, 8, 2)),
        "head": np.zeros((1, 8, 2)),
        "grippers": np.broadcast_to([[-0.2, 1.2]], (1, 8, 2)),
    }
    targets = module.actions_to_robot_targets(
        actions,
        (module.GripperCalibration(5, 65), module.GripperCalibration(10, 110)),
    )
    np.testing.assert_allclose(targets["grippers_mm"][0], [5, 110])
    assert targets["arms"].shape == (8, 14)
