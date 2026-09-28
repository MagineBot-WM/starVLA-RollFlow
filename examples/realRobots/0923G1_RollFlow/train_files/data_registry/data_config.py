"""Dataset contract for the 0923 20-task/20-action custom G1 collection."""

from __future__ import annotations

from collections import Counter

import numpy as np

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


HORIZON = 64
STATE_ACTION_FIELDS = ("left_arm", "left_gripper", "right_arm", "right_gripper", "head", "waist")
VIDEO_KEYS = ["video.head", "video.left_wrist", "video.right_wrist"]
STATE_KEYS = [f"state.{field}" for field in STATE_ACTION_FIELDS]
ACTION_KEYS = [f"action.{field}" for field in STATE_ACTION_FIELDS]
LANGUAGE_KEYS = ["annotation.human.action.task_description"]
ACTION_INDICES = list(range(HORIZON))
TASK_DATASETS = [
    ("0923_g1_pick_apple/empty_text_task_1", "pick the apple to the plate"),
    ("618data/02_task", "pick the yellow pepper and place it into the pink plate"),
    ("618data/03_task", "pick the mango and place it into the pink plate"),
    ("618data/06_task", "pick the red pepper and place it into the pink plate"),
    ("record_0904/01_task_04", "pick the persimmon"),
    ("record_0904/02_task_05", "pick the orange and place it into the pink plate"),
    ("record_0904/empty_text_task_1", "pick the orange"),
    ("record_0904/empty_text_task_2", "pick the orange"),
    ("record_0904/empty_text_task_3", "pick the persimmon"),
    ("record_0904/empty_text_task_7", "pick the orange and place it into the pink plate"),
]


def chunk_candidates(horizon: int = HORIZON) -> list[int]:
    """Return baseline chunk 1 and all even divisors of the horizon."""
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    return [1, *(chunk for chunk in range(2, horizon + 1, 2) if horizon % chunk == 0)]


def candidate_refinement_steps(horizon: int, chunk: int, max_refinement: int = 8) -> int:
    """Choose the largest allowed K up to a cap that divides H/C."""
    horizon, chunk, max_refinement = int(horizon), int(chunk), int(max_refinement)
    if horizon <= 0 or chunk <= 0 or horizon % chunk:
        raise ValueError("chunk must be a positive divisor of horizon")
    if max_refinement <= 0:
        raise ValueError("max_refinement must be positive")
    block_count = horizon // chunk
    steps = min(max_refinement, block_count)
    while block_count % steps:
        steps -= 1
    return steps


class PaddedMaskLeRobotDataset(LeRobotSingleDataset):
    """Keep configured frame starts and mark any padded action-tail slots."""

    def __init__(
        self,
        *args,
        split: str = "train",
        holdout_modulus: int = 10,
        horizon: int = HORIZON,
        complete_windows_only: bool = False,
        **kwargs,
    ):
        if split not in {"train", "validation", "all"}:
            raise ValueError(f"unsupported episode split: {split}")
        if holdout_modulus < 2:
            raise ValueError("holdout_modulus must be at least 2")
        self.split = split
        self.holdout_modulus = int(holdout_modulus)
        self.action_horizon = int(horizon)
        self.complete_windows_only = bool(complete_windows_only)
        if self.action_horizon <= 0:
            raise ValueError("horizon must be positive")
        super().__init__(*args, **kwargs)
        self._length_by_episode = {
            int(episode_id): int(length)
            for episode_id, length in zip(self.trajectory_ids, self.trajectory_lengths)
        }
        validation_ids = {
            episode_id
            for episode_id in self._length_by_episode
            if episode_id % self.holdout_modulus == 0
        }
        if self.split != "all":
            keep = np.asarray(
                [
                    (int(episode_id) in validation_ids) == (self.split == "validation")
                    for episode_id in self.trajectory_ids
                ],
                dtype=bool,
            )
            selected_ids = {int(episode_id) for episode_id in self.trajectory_ids[keep]}
            self._trajectory_ids = self.trajectory_ids[keep]
            self._trajectory_lengths = self.trajectory_lengths[keep]
            self._all_steps = [step for step in self._all_steps if int(step[0]) in selected_ids]
        if self.complete_windows_only:
            before = len(self._all_steps)
            self._all_steps = [
                (episode_id, base_index)
                for episode_id, base_index in self._all_steps
                if int(base_index) + self.action_horizon
                <= self._length_by_episode[int(episode_id)]
            ]
            print(
                f"H={self.action_horizon}: retained {len(self._all_steps)}/{before} "
                "complete action windows"
            )

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        self._rollflow_episode_id = int(trajectory_id)
        self._rollflow_base_index = int(base_index)
        return super().get_step_data(trajectory_id, base_index)

    def _pack_sample(self, data: dict) -> dict:
        sample = super()._pack_sample(data)
        episode_id = self._rollflow_episode_id
        base_index = self._rollflow_base_index
        length = self._length_by_episode[episode_id]
        sample["action_padding_mask"] = (
            base_index + np.arange(self.action_horizon, dtype=np.int64) >= length
        )
        return sample


class G10923DataConfig:
    """Single custom embodiment retaining its source 20D joint order."""

    def __init__(self, action_horizon: int = HORIZON, complete_windows_only: bool = False):
        self.action_horizon = int(action_horizon)
        self.action_indices = list(range(self.action_horizon))
        self.complete_windows_only = bool(complete_windows_only)

    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = VIDEO_KEYS
    state_keys = STATE_KEYS
    action_keys = ACTION_KEYS
    language_keys = LANGUAGE_KEYS
    observation_indices = [0]
    action_indices = ACTION_INDICES
    state_key_dims = {
        "state.left_arm": 7,
        "state.left_gripper": 1,
        "state.right_arm": 7,
        "state.right_gripper": 1,
        "state.head": 2,
        "state.waist": 2,
    }
    action_key_dims = {
        "action.left_arm": 7,
        "action.left_gripper": 1,
        "action.right_arm": 7,
        "action.right_gripper": 1,
        "action.head": 2,
        "action.waist": 2,
    }

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
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

    def make_dataset(self, *, dataset_path, modality_configs, transforms, embodiment_tag,
                     video_backend, delete_pause_frame, data_cfg, dataset_name):
        del dataset_name
        return PaddedMaskLeRobotDataset(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
            delete_pause_frame=delete_pause_frame,
            data_cfg=data_cfg,
            split=data_cfg.get("episode_split", "train"),
            holdout_modulus=int(data_cfg.get("episode_holdout_modulus", 10)),
            horizon=self.action_horizon,
            complete_windows_only=self.complete_windows_only,
        )


ROBOT_TYPE_CONFIG_MAP = {
    "g1_0923_20d": G10923DataConfig(),
    "g1_0923_20d_h128": G10923DataConfig(action_horizon=128, complete_windows_only=True),
}
ROBOT_TYPE_TO_EMBODIMENT_TAG = {}


def _balanced_task_weights(robot_type="g1_0923_20d"):
    task_counts = Counter(task for _, task in TASK_DATASETS)
    return [(name, 1.0 / task_counts[task], robot_type) for name, task in TASK_DATASETS]


DATASET_NAMED_MIXTURES = {
    "g1_0923_20d_all": _balanced_task_weights(),
    "g1_0923_20d_all_h128": _balanced_task_weights("g1_0923_20d_h128"),
}
