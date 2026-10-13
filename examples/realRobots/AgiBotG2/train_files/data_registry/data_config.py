"""AgiBot G2 LeRobot contract for the white-carton placement task."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    Normalizer,
    StateActionToTensor,
    StateActionTransform,
)


HORIZON = 64
DATASET_NAME = "white_carton_to_left_tan_carton_20261009"
ROBOT_TYPE = "agibot_g2"
MIXTURE_NAME = "agibot_g2_white_carton_place"

# Slices follow the source observation.state/action joint-name order verbatim.
STATE_ACTION_FIELDS = ("left_arm", "left_gripper", "right_arm", "right_gripper", "head", "waist")
STATE_ACTION_SLICES = {
    "left_arm": (0, 7),
    "left_gripper": (7, 8),
    "right_arm": (8, 15),
    "right_gripper": (15, 16),
    "head": (16, 19),
    "waist": (19, 24),
}
VIDEO_KEYS = ["video.top_head", "video.hand_left", "video.hand_right"]
STATE_KEYS = [f"state.{name}" for name in STATE_ACTION_FIELDS]
ACTION_KEYS = [f"action.{name}" for name in STATE_ACTION_FIELDS]
LANGUAGE_KEYS = ["annotation.human.action.task_description"]
OBSERVATION_INDICES = [0]
ACTION_INDICES = list(range(HORIZON))


class StaticAwareQ99Normalizer(Normalizer):
    """Keep q99 scaling while suppressing sub-milliradian quantization noise.

    The source records fixed G2 joints with tiny float32 jitter. Plain q99
    normalization stretches any nonzero interval to [-1, 1], turning that
    jitter into a false control signal. Dimensions with q99-q01 <= 1e-3 rad
    map to zero and inverse-map to their robust midpoint.
    """

    def __init__(self, statistics: dict[str, Any], range_threshold: float = 1e-3):
        super().__init__(mode="q99", statistics=statistics)
        self.range_threshold = float(range_threshold)
        self.static_mask = (
            self.statistics["q99"] - self.statistics["q01"]
        ).abs() <= self.range_threshold
        self.static_value = (self.statistics["q99"] + self.statistics["q01"]) / 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q01 = self.statistics["q01"].to(device=x.device, dtype=x.dtype)
        q99 = self.statistics["q99"].to(device=x.device, dtype=x.dtype)
        static_mask = self.static_mask.to(device=x.device)
        dynamic_mask = ~static_mask
        normalized = torch.zeros_like(x)
        if bool(dynamic_mask.any()):
            normalized[..., dynamic_mask] = 2 * (
                (x[..., dynamic_mask] - q01[dynamic_mask])
                / (q99[dynamic_mask] - q01[dynamic_mask])
            ) - 1
        return normalized.clamp(-2.2, 2.2)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        q01 = self.statistics["q01"].to(device=x.device, dtype=x.dtype)
        q99 = self.statistics["q99"].to(device=x.device, dtype=x.dtype)
        output = (x + 1) / 2 * (q99 - q01) + q01
        static_mask = self.static_mask.to(device=x.device)
        if bool(static_mask.any()):
            static_value = self.static_value.to(device=x.device, dtype=x.dtype)
            output[..., static_mask] = static_value[static_mask]
        return output


class G2StateActionTransform(StateActionTransform):
    """Use the G2 static-aware q99 normalizer for both state and action."""

    static_range_threshold: float = 1e-3

    def set_metadata(self, dataset_metadata):
        super().set_metadata(dataset_metadata)
        for key, mode in self.normalization_modes.items():
            if mode != "q99":
                continue
            old = self._normalizers[key]
            statistics = {
                name: value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value
                for name, value in old.statistics.items()
            }
            self._normalizers[key] = StaticAwareQ99Normalizer(
                statistics, range_threshold=self.static_range_threshold
            )


class PaddingMaskG2Dataset(LeRobotSingleDataset):
    """Attach a per-action mask so the final padded frames do not train flow."""

    def __init__(
        self,
        *args,
        horizon: int = HORIZON,
        split: str = "train",
        validation_episode_count: int = 9,
        **kwargs,
    ):
        if split not in {"train", "validation", "all"}:
            raise ValueError(f"unsupported episode split: {split}")
        if validation_episode_count < 0:
            raise ValueError("validation_episode_count must be non-negative")
        self.action_horizon = int(horizon)
        self.split = split
        self.validation_episode_count = int(validation_episode_count)
        if self.action_horizon <= 0:
            raise ValueError("horizon must be positive")

        super().__init__(*args, **kwargs)
        self._length_by_episode = {
            int(episode_id): int(length)
            for episode_id, length in zip(self.trajectory_ids, self.trajectory_lengths)
        }
        episode_ids = sorted(self._length_by_episode)
        validation_count = min(self.validation_episode_count, len(episode_ids))
        validation_positions = np.linspace(
            0, len(episode_ids) - 1, num=validation_count, dtype=np.int64
        )
        validation_ids = {episode_ids[position] for position in validation_positions}
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
            self._all_steps = [
                step for step in self._all_steps if int(step[0]) in selected_ids
            ]

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        self._g2_episode_id = int(trajectory_id)
        self._g2_base_index = int(base_index)
        return super().get_step_data(trajectory_id, base_index)

    def _pack_sample(self, data: dict) -> dict:
        sample = super()._pack_sample(data)
        length = self._length_by_episode[self._g2_episode_id]
        sample["action_padding_mask"] = (
            self._g2_base_index + np.arange(self.action_horizon) >= length
        )
        return sample


class AgiBotG2DataConfig:
    """G2 qpos contract in original 24D order, with one-step observations."""

    embodiment_tag = EmbodimentTag.AGIBOT_G2
    video_keys = VIDEO_KEYS
    state_keys = STATE_KEYS
    action_keys = ACTION_KEYS
    language_keys = LANGUAGE_KEYS
    observation_indices = OBSERVATION_INDICES
    action_indices = ACTION_INDICES
    state_key_dims = {
        f"state.{name}": end - start for name, (start, end) in STATE_ACTION_SLICES.items()
    }
    action_key_dims = {
        f"action.{name}": end - start for name, (start, end) in STATE_ACTION_SLICES.items()
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
                G2StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={key: "q99" for key in self.state_keys},
                ),
                StateActionToTensor(apply_to=self.action_keys),
                G2StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={key: "q99" for key in self.action_keys},
                ),
            ]
        )

    def make_dataset(
        self,
        *,
        dataset_path,
        modality_configs,
        transforms,
        embodiment_tag,
        video_backend,
        delete_pause_frame,
        data_cfg,
        dataset_name,
    ):
        del dataset_name
        return PaddingMaskG2Dataset(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
            delete_pause_frame=delete_pause_frame,
            data_cfg=data_cfg,
            horizon=int(data_cfg.get("action_horizon", HORIZON)),
            split=data_cfg.get("episode_split", "train"),
            validation_episode_count=int(data_cfg.get("validation_episode_count", 9)),
        )


ROBOT_TYPE_CONFIG_MAP = {ROBOT_TYPE: AgiBotG2DataConfig()}
ROBOT_TYPE_TO_EMBODIMENT_TAG = {}
DATASET_NAMED_MIXTURES = {
    MIXTURE_NAME: [(DATASET_NAME, 1.0, ROBOT_TYPE)],
}
