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


ROBOT_TYPE_CONFIG_MAP = {"agibot_g1": AgiBotG1DataConfig()}
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
    "agibot_g1_public": _entries(_PUBLIC_TASKS, 1.0),
    "agibot_g1_real": _entries(_REAL_TASKS, 1.0),
    # The three local object variants form one logical pick/place task. Their
    # weights sum to one public task, so public data remains dominant (16:1).
    # Keep balance_dataset_weights=false in the YAML.
    "agibot_g1_all": _entries(_PUBLIC_TASKS, 1.0) + _entries(_REAL_TASKS, 1.0 / len(_REAL_TASKS)),
}
