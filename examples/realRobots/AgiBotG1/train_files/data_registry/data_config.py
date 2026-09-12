"""AgiBot G1 data contract and dataset mixtures.

Both source families are exposed through the canonical no-copy overlays under
``AgiBot-G1-G2-StarVLA/g1/manipulation``.  Their raw vector layouts differ,
but ``modality.json`` maps both into the model-facing order below.
"""

from typing import ClassVar

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class AgiBotG1DataConfig:
    """Canonical G1 contract with 20-D body state and 22-D action."""

    embodiment_tag = EmbodimentTag.AGIBOT_G1

    # Stable semantic camera order: overview, left wrist, right wrist.
    video_keys: ClassVar[list[str]] = ["video.head", "video.hand_left", "video.hand_right"]
    state_keys: ClassVar[list[str]] = [
        "state.arms",
        "state.waist",
        "state.head",
        "state.grippers",
    ]
    action_keys: ClassVar[list[str]] = [
        "action.arms",
        "action.waist",
        "action.head",
        "action.grippers",
        "action.base_velocity",
    ]
    language_keys: ClassVar[list[str]] = ["annotation.human.action.task_description"]

    state_key_dims: ClassVar[dict[str, int]] = {
        "state.arms": 14,
        "state.waist": 2,
        "state.head": 2,
        "state.grippers": 2,
    }
    action_key_dims: ClassVar[dict[str, int]] = {
        "action.arms": 14,
        "action.waist": 2,
        "action.head": 2,
        "action.grippers": 2,
        "action.base_velocity": 2,
    }

    observation_indices: ClassVar[list[int]] = [0]
    action_indices: ClassVar[list[int]] = list(range(32))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        # Grippers are continuous normalized opening commands, not binary
        # labels. q99 retains intermediate commands and is robust to outliers.
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={key: "q99" for key in self.state_keys},
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={key: "q99" for key in self.action_keys},
                ),
            ]
        )


class AgiBotG1BinaryDataConfig(AgiBotG1DataConfig):
    """Native real-G1 convention; separate statistics from continuous opening."""

    embodiment_tag = EmbodimentTag.AGIBOT_G1_BINARY
    state_keys: ClassVar[list[str]] = AgiBotG1DataConfig.state_keys + ["state.base_velocity"]
    state_key_dims: ClassVar[dict[str, int]] = {**AgiBotG1DataConfig.state_key_dims, "state.base_velocity": 2}

    def transform(self):
        keys = self.state_keys + self.action_keys
        modes = {key: "q99" for key in keys}
        modes["action.grippers"] = "binary"  # 0=open, 1=closed, unchanged
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=keys),
            StateActionTransform(apply_to=keys, normalization_modes=modes),
        ])


class AgiBotG1AppleDataConfig(AgiBotG1BinaryDataConfig):
    """Apple export: native G1 layout with continuous gripper commands.

    The export uses the same 22-D canonical fields as the binary G1 adapter,
    but its right gripper contains many intermediate positions.  Keep the
    binary tag for model routing while using q99 for both gripper channels.
    """

    embodiment_tag = EmbodimentTag.AGIBOT_G1_BINARY

    def transform(self):
        keys = self.state_keys + self.action_keys
        modes = {key: "q99" for key in keys}
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=keys),
            StateActionTransform(apply_to=keys, normalization_modes=modes),
        ])


ROBOT_TYPE_CONFIG_MAP = {
    "agibot_g1": AgiBotG1DataConfig(),
    "agibot-g1": AgiBotG1BinaryDataConfig(),
    "agibot-g1-apple": AgiBotG1AppleDataConfig(),
}
ROBOT_TYPE_TO_EMBODIMENT_TAG = {}


_PUBLIC_TASKS = [
    "supermarket_shelf_pickup",
    "supermarket_bag_packing",
    "load_dishwasher",
    "toast_bread",
    "sort_personal_care_products",
    "sort_food",
    "remove_toast_from_toaster",
    "tote_bag_packing",
    "make_tea",
    "set_dining_tray",
    "supermarket_freezer_pickup",
    "supermarket_checkout_scan",
    "sort_clothes",
    "industrial_logistics_packing",
    "clear_tabletop_trash",
    "iron_clothes",
]
_REAL_TASKS = [
    "pick_mango_place_pink_plate",
    "pick_red_pepper_place_pink_plate",
    "pick_yellow_pepper_place_pink_plate",
]


def _entries(names, weight):
    return [(f"g1/manipulation/{name}", weight, "agibot_g1") for name in names]


DATASET_NAMED_MIXTURES = {
    "agibot-g1": [
        (name, 1.0, "agibot-g1") for name in (
            "task_01_pick_the_orange",
            "task_02_pick_the_orange",
            "task_03_persimmon_orange_yellow_pepper_transfer",
            "task_03_pick_the_persimmon",
            "task_04_pick_the_persimmon",
            "task_04_pick_the_red_pepper_to_pink_plate",
            "task_05_pick_the_orange_to_pink_plate",
            "task_05_pick_the_yellow_pepper_to_pink_plate",
            "task_06_pick_the_mango_to_pink_plate",
            "task_07_pick_the_orange_to_pink_plate",
        )
    ],
    "agibot_g1_public": _entries(_PUBLIC_TASKS, 1.0),
    "agibot_g1_real": _entries(_REAL_TASKS, 1.0),
    # The three local object variants form one logical pick/place task. Their
    # weights sum to one public task, so public data remains dominant (16:1).
    # Keep balance_dataset_weights=false in the YAML.
    "agibot_g1_all": _entries(_PUBLIC_TASKS, 1.0) + _entries(_REAL_TASKS, 1.0 / len(_REAL_TASKS)),
}

# 50% LIBERO / 50% real G1; equal tasks within each family. Root is Datasets/.
DATASET_NAMED_MIXTURES["libero_agibot_g1"] = [
    (f"libero/{name}", 0.125, "libero_franka_rollflow") for name in (
        "libero_object_no_noops_1.0.0_lerobot",
        "libero_goal_no_noops_1.0.0_lerobot",
        "libero_spatial_no_noops_1.0.0_lerobot",
        "libero_10_no_noops_1.0.0_lerobot",
    )
] + [(f"agibot-g1/{name}", 0.05, tag) for name, _, tag in DATASET_NAMED_MIXTURES["agibot-g1"]]

# Clean second-stage experiment: only the Apple G1 export is mixed with LIBERO.
# LIBERO contributes 30% (0.075 per suite) and Apple contributes 70%.
# The Apple overlay is derived from pick_up_the_apple without modifying its source.
DATASET_NAMED_MIXTURES["libero_pick_up_the_apple"] = [
    (f"libero/{name}", 0.075, "libero_franka_rollflow") for name in (
        "libero_object_no_noops_1.0.0_lerobot",
        "libero_goal_no_noops_1.0.0_lerobot",
        "libero_spatial_no_noops_1.0.0_lerobot",
        "libero_10_no_noops_1.0.0_lerobot",
    )
] + [("agibot-g1-apple", 0.7, "agibot-g1-apple")]

# The controlled-rate counterpart uses the independent 10 Hz LIBERO overlay.
# The original 20 Hz LIBERO datasets remain available under the mixture above.
DATASET_NAMED_MIXTURES["libero10hz_pick_up_the_apple"] = [
    (f"libero_10hz/{name}", 0.075, "libero_franka_rollflow") for name in (
        "libero_object_no_noops_1.0.0_lerobot",
        "libero_goal_no_noops_1.0.0_lerobot",
        "libero_spatial_no_noops_1.0.0_lerobot",
        "libero_10_no_noops_1.0.0_lerobot",
    )
] + [("agibot-g1-apple", 0.7, "agibot-g1-apple")]

# Corrected Apple overlay.  Keep the old mixture name/path intact so previous
# checkpoints remain reproducible; new runs should use this explicit variant.
DATASET_NAMED_MIXTURES["libero10hz_pick_up_the_apple_corrected"] = [
    (f"libero_10hz/{name}", 0.075, "libero_franka_rollflow") for name in (
        "libero_object_no_noops_1.0.0_lerobot",
        "libero_goal_no_noops_1.0.0_lerobot",
        "libero_spatial_no_noops_1.0.0_lerobot",
        "libero_10_no_noops_1.0.0_lerobot",
    )
] + [("agibot-g1-apple-corrected", 0.7, "agibot-g1-apple")]

# Benchmark-rate joint counterpart for the adapter-first experiment.  Keep
# LIBERO at its native 20 Hz (the simulator also runs at 20 Hz), use the
# corrected Apple overlay, and give each embodiment family equal probability.
# This is intentionally a new name so existing 10 Hz experiments remain
# exactly reproducible.
DATASET_NAMED_MIXTURES["libero20hz_pick_up_the_apple_corrected_balanced"] = [
    (f"libero/{name}", 0.125, "libero_franka_rollflow") for name in (
        "libero_object_no_noops_1.0.0_lerobot",
        "libero_goal_no_noops_1.0.0_lerobot",
        "libero_spatial_no_noops_1.0.0_lerobot",
        "libero_10_no_noops_1.0.0_lerobot",
    )
] + [("agibot-g1-apple-corrected", 0.5, "agibot-g1-apple")]

# G1-only counterpart for clean from-scratch pretraining.  Keep this as a
# named mixture so the launcher and config cannot accidentally fall back to the
# legacy, uncorrected Apple overlay.
DATASET_NAMED_MIXTURES["agibot_g1_apple_corrected"] = [
    ("agibot-g1-apple-corrected", 1.0, "agibot-g1-apple")
]
