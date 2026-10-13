"""Read-only LeRobot replay source with the same frame fields as EVA transport."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq

from .contract import CAMERA_ORDER


@dataclass(frozen=True)
class ReplayObservation:
    """Equivalent of EVA's Observation for the policy-facing sensor fields."""

    images: dict[str, np.ndarray]
    state_qpos: np.ndarray
    timestamp: float | None = None


class LeRobotReplayTransport:
    """Read raw G2 observations/actions from a LeRobot v2.1 dataset.

    It exposes EVA's read-only `get_frame`, `seek`, task, and camera-key surface.
    Frames are decoded by OpenCV and returned BGR/HWC uint8, matching both the
    EVA G2 ZMQ bridge and EVA's DatasetTransport frame contract. This source has
    no `publish_action` method by design.
    """

    def __init__(self, dataset_dir: str | Path, episode_id: int) -> None:
        self.dataset_dir = Path(dataset_dir)
        with (self.dataset_dir / "meta" / "info.json").open() as file:
            self._info = json.load(file)
        self._episode_id = int(episode_id)
        total = int(self._info["total_episodes"])
        if not 0 <= self._episode_id < total:
            raise IndexError(f"episode_id={episode_id} outside [0, {total})")

        self._chunk_size = int(self._info.get("chunks_size", 1000))
        episode_chunk = self._episode_id // self._chunk_size
        values = {"episode_chunk": episode_chunk, "episode_index": self._episode_id}
        self._parquet_path = self.dataset_dir / self._info["data_path"].format(**values)
        if not self._parquet_path.is_file():
            raise FileNotFoundError(self._parquet_path)
        table = pq.read_table(self._parquet_path)
        self._states = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
        self._actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
        if self._states.shape != self._actions.shape or self._states.ndim != 2 or self._states.shape[1] != 24:
            raise ValueError(
                f"G2 replay expected matching [N,24] state/action, got "
                f"{self._states.shape}/{self._actions.shape}"
            )
        self._timestamps = (
            np.asarray(table.column("timestamp").to_pylist(), dtype=np.float64)
            if "timestamp" in table.column_names
            else None
        )
        task_index = int(table.column("task_index")[0].as_py()) if "task_index" in table.column_names else 0
        self.current_task = self._load_task(task_index)
        self._videos: dict[str, cv2.VideoCapture] = {}
        self._video_paths: dict[str, Path] = {}
        for camera in CAMERA_ORDER:
            video_key = f"observation.images.{camera}"
            video_path = self.dataset_dir / self._info["video_path"].format(
                **values, video_key=video_key
            )
            if not video_path.is_file():
                raise FileNotFoundError(f"episode {episode_id} missing camera video {video_path}")
            self._video_paths[camera] = video_path
        self._frame_index = 0

    def _load_task(self, task_index: int) -> str:
        tasks_path = self.dataset_dir / "meta" / "tasks.jsonl"
        with tasks_path.open() as file:
            for line in file:
                row = json.loads(line)
                if int(row["task_index"]) == task_index:
                    task = str(row["task"]).strip()
                    if task:
                        return task
        raise KeyError(f"task_index={task_index} missing from {tasks_path}")

    @property
    def episode_id(self) -> int:
        return self._episode_id

    @property
    def n_steps(self) -> int:
        return int(self._states.shape[0])

    def available_camera_keys(self) -> tuple[str, ...]:
        return CAMERA_ORDER

    def get_action_trajectory(self) -> np.ndarray:
        return self._actions.copy()

    def seek(self, frame_index: int) -> None:
        self._frame_index = max(0, min(int(frame_index), self.n_steps - 1))

    def get_frame(self) -> ReplayObservation | None:
        if not 0 <= self._frame_index < self.n_steps:
            return None
        images: dict[str, np.ndarray] = {}
        for camera, video_path in self._video_paths.items():
            capture = self._videos.get(camera)
            if capture is None:
                capture = cv2.VideoCapture(str(video_path))
                if not capture.isOpened():
                    raise RuntimeError(f"OpenCV cannot open G2 video: {video_path}")
                self._videos[camera] = capture
            capture.set(cv2.CAP_PROP_POS_FRAMES, self._frame_index)
            ok, bgr = capture.read()
            if not ok or bgr is None:
                raise RuntimeError(f"failed decoding {camera} frame {self._frame_index} ({video_path})")
            images[camera] = np.ascontiguousarray(bgr)
        timestamp = (
            float(self._timestamps[self._frame_index]) if self._timestamps is not None else None
        )
        return ReplayObservation(
            images=images,
            state_qpos=self._states[self._frame_index].copy(),
            timestamp=timestamp,
        )

    def close(self) -> None:
        for capture in self._videos.values():
            capture.release()
        self._videos.clear()

    def __enter__(self) -> "LeRobotReplayTransport":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
