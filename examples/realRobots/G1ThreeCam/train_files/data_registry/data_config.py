"""Data registry for the isolated G1 right-arm red-pepper task."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig, CachedLeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform


class G1ThreeCamDataConfig:
    """20-D joint/waist/head/gripper state and action, with three RGB streams."""

    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = [
        "video.cam_high_rgb",
        "video.cam_left_wrist_rgb",
        "video.cam_right_wrist_rgb",
    ]
    state_keys = ["state.robot"]
    action_keys = ["action.robot"]
    state_key_dims = {"state.robot": 20}
    action_key_dims = {"action.robot": 20}
    # LeRobot v2.1 task metadata is converted to this standard logical field by the loader.
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(50))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(apply_to=self.state_keys, normalization_modes={"state.robot": "mean_std"}),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(apply_to=self.action_keys, normalization_modes={"action.robot": "mean_std"}),
            ]
        )


class G1Task05ThreeCamDataConfig:
    """Canonical task-05 contract with 20-D state and 22-D action."""

    # ACT keeps the historical mean/std action representation.  DP has its own
    # entry below because its DDPM scheduler expects bounded action targets.
    action_normalization_mode = "mean_std"

    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = [
        "video.cam_high_rgb",
        "video.cam_left_wrist_rgb",
        "video.cam_right_wrist_rgb",
    ]
    state_keys = ["state.robot"]
    action_keys = ["action.robot"]
    state_key_dims = {"state.robot": 20}
    action_key_dims = {"action.robot": 22}
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(50))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(apply_to=self.state_keys, normalization_modes={"state.robot": "mean_std"}),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={"action.robot": self.action_normalization_mode},
                ),
            ]
        )

    def make_dataset(self, dataset_path, modality_configs, transforms,
                     embodiment_tag, video_backend, delete_pause_frame,
                     data_cfg, dataset_name):
        """Cache task05 frames once so batch=96 does not repeatedly decode videos."""
        return CachedLeRobotSingleDataset(
            img_resize=(160, 160),
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
            delete_pause_frame=delete_pause_frame,
            data_cfg=data_cfg,
        )


class G1Task05DPThreeCamDataConfig(G1Task05ThreeCamDataConfig):
    """Task-05 DP view: bounded min/max action targets for DDPM sampling."""

    action_normalization_mode = "min_max"


class G1AcceptedACTDataConfig:
    """Accepted G1 export: 20-D state and 22-D canonical action for ACT."""

    embodiment_tag = EmbodimentTag.AGIBOT_G1_BINARY
    video_keys = ["video.head", "video.hand_left", "video.hand_right"]
    state_keys = ["state.arms", "state.waist", "state.head", "state.grippers"]
    action_keys = [
        "action.arms", "action.waist", "action.head", "action.grippers", "action.base_velocity"
    ]
    state_key_dims = {"state.arms": 14, "state.waist": 2, "state.head": 2, "state.grippers": 2}
    action_key_dims = {
        "action.arms": 14, "action.waist": 2, "action.head": 2,
        "action.grippers": 2, "action.base_velocity": 2,
    }
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(50))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        keys = self.state_keys + self.action_keys
        modes = {key: "q99" for key in keys}
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=keys),
            StateActionTransform(apply_to=keys, normalization_modes=modes),
        ])


ROBOT_TYPE_CONFIG_MAP = {
    "g1_20d_threecam": G1ThreeCamDataConfig(),
    "g1_task05_20s22a_threecam": G1Task05ThreeCamDataConfig(),
    "g1_task05_20s22a_dp_threecam": G1Task05DPThreeCamDataConfig(),
    "agibot-g1-accepted": G1AcceptedACTDataConfig(),
}
ROBOT_TYPE_TO_EMBODIMENT_TAG = {}
DATASET_NAMED_MIXTURES = {
    "g1_right_arm_pepper": [("01_task_04_right_arm", 1.0, "g1_20d_threecam")],
    "g1_task05_orange_to_pink": [
        ("task_05_pick_the_orange_to_pink_plate", 1.0, "g1_task05_20s22a_threecam")
    ],
    "g1_task05_orange_to_pink_dp_minmax": [
        ("task_05_pick_the_orange_to_pink_plate", 1.0, "g1_task05_20s22a_dp_threecam")
    ],
    "accepted_act": [
        ("accepted_act_overlay_corrected", 1.0, "agibot-g1-accepted")
    ],
}
