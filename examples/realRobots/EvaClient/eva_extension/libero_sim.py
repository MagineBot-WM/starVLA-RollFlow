"""LIBERO simulator adapters registered into an external EVA-CLIENT runtime."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from core.registry import ROBOT_REGISTRY, TRANSPORT_REGISTRY
from core.types import Observation
from robots.base import ActuatorGroup, CameraSpec, ObservationSchema, Robot
from transport.base import TransportBridge


@ROBOT_REGISTRY.register("eva_libero_virtual")
class LiberoVirtualRobot(Robot):
    """Seven-dimensional LIBERO end-effector delta action represented by EVA."""

    def __init__(self) -> None:
        super().__init__(
            name="libero_virtual_franka",
            actuator_groups=(
                ActuatorGroup(
                    name="eef_delta",
                    dof=7,
                    joint_names=("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"),
                    gripper_index=6,
                ),
            ),
            initial_qpos=np.zeros(7, dtype=np.float32),
            observation_schema=ObservationSchema(
                cameras=(
                    CameraSpec("agentview", "agentview"),
                    CameraSpec("wrist", "wrist"),
                ),
                state_composition=("eef_delta",),
            ),
        )


@TRANSPORT_REGISTRY.register("libero_sim")
class LiberoSimTransport(TransportBridge):
    """Run LIBERO as an offline EVA transport; action publishing steps the sim."""

    DUMMY_ACTION = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    SUITE_MAX_STEPS = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }

    def __init__(self, config: Any, robot: Robot) -> None:
        self._config = config
        self._robot = robot
        options = config.transport.get("libero", {})
        self._suite_name = str(options.get("suite", "libero_goal"))
        self._task_id = int(options.get("task_id", 0))
        self._init_state_id = int(options.get("init_state_id", 0))
        self._warmup_steps = int(options.get("warmup_steps", 10))
        self._max_steps = int(options.get("max_steps", 0))
        self._seed = int(options.get("seed", 7))
        self._step_count = 0
        self._done = False
        self._success = False
        self._last_info: dict[str, Any] = {}
        self._obs: dict[str, Any] | None = None
        self._closed = False
        self._benchmark_module = None
        self._env = None
        self._task = None
        self.current_task = ""
        self.reset_episode(self._task_id, self._init_state_id)

    def _lazy_import_libero(self):
        if self._benchmark_module is not None:
            return
        try:
            from libero.libero import benchmark, get_libero_path
            from libero.libero.envs import OffScreenRenderEnv
        except ImportError as error:
            raise RuntimeError(
                "LIBERO simulator dependencies are missing. Activate the LIBERO environment "
                "described in examples/simBenchmarks/LIBERO/eval_files/install_libero.sh."
            ) from error
        self._benchmark_module = (benchmark, get_libero_path, OffScreenRenderEnv)

    @property
    def done(self) -> bool:
        return self._done

    @property
    def success(self) -> bool:
        return self._success

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def task_description(self) -> str:
        return self.current_task

    def reset_episode(self, task_id: int | None = None, init_state_id: int | None = None) -> None:
        """Reset to one deterministic task/init-state pair and stabilize objects."""
        self._lazy_import_libero()
        if task_id is not None:
            self._task_id = int(task_id)
        if init_state_id is not None:
            self._init_state_id = int(init_state_id)
        benchmark, get_libero_path, env_cls = self._benchmark_module
        suite = benchmark.get_benchmark_dict()[self._suite_name]()
        if self._max_steps <= 0:
            try:
                self._max_steps = self.SUITE_MAX_STEPS[self._suite_name]
            except KeyError as error:
                raise ValueError(
                    f"Set transport.libero.max_steps for unsupported suite {self._suite_name!r}"
                ) from error
        if not 0 <= self._task_id < suite.n_tasks:
            raise IndexError(f"task_id={self._task_id} outside suite {self._suite_name} ({suite.n_tasks})")
        self._task = suite.get_task(self._task_id)
        states = suite.get_task_init_states(self._task_id)
        if not 0 <= self._init_state_id < len(states):
            raise IndexError(
                f"init_state_id={self._init_state_id} outside task state count {len(states)}"
            )
        if self._env is not None:
            self._env.close()
        task_bddl = Path(get_libero_path("bddl_files")) / self._task.problem_folder / self._task.bddl_file
        resolution = int(self._config.transport.get("libero", {}).get("render_size", 256))
        self._env = env_cls(
            bddl_file_name=task_bddl,
            camera_heights=resolution,
            camera_widths=resolution,
        )
        self._env.seed(self._seed)
        self._env.reset()
        self._env.reset()
        self._obs = self._env.set_init_state(states[self._init_state_id])
        self.current_task = str(self._task.language)
        self._done = False
        self._success = False
        self._step_count = 0
        self._last_info = {}
        for _ in range(self._warmup_steps):
            self._obs, _reward, _done, info = self._env.step(self.DUMMY_ACTION.tolist())
            self._last_info = info or {}

    def _observation_images(self) -> dict[str, np.ndarray]:
        if self._obs is None:
            raise RuntimeError("LIBERO has no current observation; reset_episode first")
        # The shared policy client applies the contract's 180-degree rotation,
        # matching examples/simBenchmarks/LIBERO/eval_files/eval_libero.py.
        agentview = np.ascontiguousarray(self._obs["agentview_image"])
        wrist = np.ascontiguousarray(self._obs["robot0_eye_in_hand_image"])
        return {"agentview": agentview, "wrist": wrist}

    def get_frame(self) -> Observation | None:
        if self._closed or self._done:
            return None
        return Observation(
            images=self._observation_images(),
            # This checkpoint's state_dim is 0; the configured StarVLA adapter omits it.
            state_qpos=np.zeros(7, dtype=np.float32),
            timestamp=time.monotonic(),
        )

    def publish_action(self, action: np.ndarray, target: str = "real") -> None:
        _ = target  # LIBERO is always a local simulation target.
        if self._closed:
            raise RuntimeError("LIBERO transport is closed")
        vector = np.asarray(action, dtype=np.float32).reshape(-1)
        if vector.shape != (7,) or not np.isfinite(vector).all():
            raise ValueError(f"LIBERO expects a finite 7D action, got {vector.shape}")
        if self._done:
            return
        self._obs, _reward, self._done, info = self._env.step(vector.tolist())
        self._last_info = info or {}
        self._step_count += 1
        self._success = bool(self._last_info.get("success", self._done))
        if self._step_count >= self._max_steps:
            self._done = True

    def get_latest_qpos(self) -> np.ndarray:
        return np.zeros(7, dtype=np.float32)

    def is_offline(self) -> bool:
        return True

    def is_shutdown(self) -> bool:
        return self._closed

    def close(self) -> None:
        self._closed = True
        if self._env is not None:
            self._env.close()
            self._env = None
