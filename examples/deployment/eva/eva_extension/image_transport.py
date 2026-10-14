"""Apply checkpoint image interpolation before EVA's existing observation builder.

These subclasses preserve the stock dataset/ZMQ behavior. EVA subsequently
resizes to the same dimensions, so it doesn't resample the prepared pixels.
"""

from dataclasses import replace
import time

import numpy as np
from PIL import Image

from core.registry import TRANSPORT_REGISTRY
from transport.dataset import DatasetTransport
from transport.zmq import ZmqTransport
from core.types import Observation

from examples.deployment.eva.bridge_contract import load_contract
from examples.deployment.eva.runtime import validate_robot
from examples.deployment.eva.execution_trace import ExecutionTrace


class ContractImageMixin:
    def __init__(self, config, robot, *args, **kwargs):
        self._contract = load_contract(config.policy.backend_options.contract_path)
        validate_robot(robot, self._contract)
        self._resample = {"bicubic": Image.Resampling.BICUBIC,
                          "bilinear": Image.Resampling.BILINEAR}[
                              self._contract["image"].get("interpolation", "bicubic")]
        super().__init__(config, robot, *args, **kwargs)

    def get_frame(self):
        frame = super().get_frame()
        if frame is None:
            return None
        return self._prepare_frame(frame)

    def _prepare_frame(self, frame):
        contract = self._contract
        height, width = contract["image"]["hw"]
        if (height, width) != (self._config.transport.image_height,
                              self._config.transport.image_width):
            raise ValueError("EVA transport resolution must match the image contract")
        images = {}
        for camera in contract["cameras"]:
            key = camera["eva_key"]
            if key not in frame.images:
                raise ValueError(f"Observation missing policy camera {key!r}")
            value = np.asarray(frame.images[key])
            if (value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3
                    or value.shape[0] == 0 or value.shape[1] == 0):
                raise ValueError(f"Camera {key!r} must supply HWC uint8 color pixels, got {value.dtype}/{value.shape}")
            images[key] = np.asarray(Image.fromarray(value).resize(
                (width, height), resample=self._resample)).copy()
        return replace(frame, images=images)


class ContractDatasetTransport(ContractImageMixin, DatasetTransport):
    def get_frame(self):
        from examples.deployment.eva.adapters.eva import complete_dataset_frame
        frame = complete_dataset_frame(self, [camera["eva_key"] for camera in self._contract["cameras"]])
        return None if frame is None else self._prepare_frame(frame)


class ContractZmqTransport(ContractImageMixin, ZmqTransport):
    """Policy frames must come from one complete wire message, without camera cache mixing."""

    def __init__(self, config, robot):
        self._init_wire_guards(config)
        super().__init__(config, robot)
        from examples.deployment.eva.adapters.eva import subscription_aware_publisher, latest_only_robot_readers
        latest_only_robot_readers(self)
        self._command_publisher = subscription_aware_publisher(self)
        self._command_subscribed = False
        self._command_connect_timeout_s = float(config.transport.get("command_connect_timeout_s", 2.0))
        if not np.isfinite(self._command_connect_timeout_s) or self._command_connect_timeout_s <= 0:
            self.close()
            raise ValueError("command_connect_timeout_s must be finite and positive")

    def _init_wire_guards(self, config):
        self._max_frame_age_s = float(config.transport.get("max_frame_age_s", 2.0))
        if not np.isfinite(self._max_frame_age_s) or self._max_frame_age_s <= 0:
            raise ValueError("transport.max_frame_age_s must be finite and positive")
        self._last_source_timestamp = None
        self._last_progress = time.monotonic()
        self._feedback_timestamp = None
        self._feedback_progress = self._last_progress
        self._pending_chunk = None
        self._chunk_identity = None
        self._issued_in_chunk = 0
        self._chunk_target = "real"
        self._execution_trace = ExecutionTrace(config.transport.get("trace_dir"))
        self._action_issued_callback = None

    def set_action_issued_callback(self, callback):
        self._action_issued_callback = callback

    def _checked_snapshot(self, reader_name="_reader"):
        # The version-specific reader access is isolated in adapters.eva.
        from examples.deployment.eva.adapters.eva import latest_wire_observation
        wire = latest_wire_observation(self, reader_name)
        now = time.monotonic()
        policy_reader = reader_name == "_reader"
        stamp_attr = "_last_source_timestamp" if policy_reader else "_feedback_timestamp"
        progress_attr = "_last_progress" if policy_reader else "_feedback_progress"
        progress = getattr(self, progress_attr)
        if wire is None:
            if now - progress > self._max_frame_age_s:
                raise ValueError("No complete robot observation received within max_frame_age_s")
            return None
        stamp = float(wire.t)
        if not np.isfinite(stamp):
            raise ValueError("Robot observation timestamp is not finite")
        previous_stamp = getattr(self, stamp_attr)
        if previous_stamp is not None and stamp < previous_stamp:
            raise ValueError("Robot source clock moved backwards; reconnect/reset the execution stream")
        if previous_stamp is None or stamp > previous_stamp:
            setattr(self, stamp_attr, stamp)
            setattr(self, progress_attr, now)
        age_fn = getattr(getattr(self, reader_name), "seconds_since_last_recv", None)
        receipt_age = age_fn() if age_fn is not None else None
        if receipt_age is not None and receipt_age > self._max_frame_age_s:
            raise ValueError("Robot observation receipt is stale; refusing cached feedback")
        if now - getattr(self, progress_attr) > self._max_frame_age_s:
            raise ValueError("Robot observation timestamp stopped advancing; refusing stale policy input")
        parts = []
        dims = {group.name: group.dof for group in self._robot.actuator_groups}
        for name in self._robot.observation_schema.state_composition:
            part = np.asarray(wire.state.get(name), dtype=np.float32)
            if part.shape != (dims[name],) or not np.isfinite(part).all():
                raise ValueError(f"Wire state group {name!r} requires finite ({dims[name]},), got {part.shape}")
            parts.append(part)
        return wire, np.concatenate(parts)

    def get_frame(self):
        snapshot = self._checked_snapshot()
        if snapshot is None:
            return None
        wire, state = snapshot
        # _prepare_frame checks cameras in this exact wire snapshot. UI previews
        # continue using EVA's camera cache, independently of model input.
        return self._prepare_frame(Observation(
            images=wire.images, state_qpos=state, timestamp=float(wire.t),
        ))

    def get_latest_qpos(self):
        snapshot = self._checked_snapshot("_qpos_reader")
        return None if snapshot is None else snapshot[1]

    def begin_policy_chunk(self, chunk, identity=None, *, target="real"):
        # Called after inference by the external strategy. Validate the robot
        # stream again before exposing actions computed from an old observation.
        self._require_complete_snapshot()
        self._ensure_command_subscriber()
        if self._pending_chunk is not None and self._issued_in_chunk < len(self._pending_chunk):
            raise ValueError("Previous rolling chunk was not completely issued; reset before fetching another")
        self._pending_chunk = np.asarray(chunk, dtype=np.float32).copy()
        self._chunk_identity = dict(identity or {})
        self._issued_in_chunk = 0
        self._chunk_target = target
        self._execution_trace.write("chunk_ready", **self._chunk_identity,
                                    actions_after_eva_processing=self._pending_chunk.tolist())

    def reset_policy_chunk(self):
        if self._pending_chunk is not None:
            self._execution_trace.write("chunk_reset", **(self._chunk_identity or {}),
                                        issued=self._issued_in_chunk, total=len(self._pending_chunk))
        self._pending_chunk = None
        self._chunk_identity = None
        self._issued_in_chunk = 0

    def _require_complete_snapshot(self):
        snapshot = self._checked_snapshot()
        if snapshot is None:
            raise ValueError("No complete robot observation; refusing actuator command")
        wire, state = snapshot
        for camera in self._contract["cameras"]:
            value = np.asarray(wire.images.get(camera["eva_key"]))
            if (value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3
                    or value.shape[0] == 0 or value.shape[1] == 0):
                raise ValueError(f"Missing/invalid policy camera {camera['eva_key']!r}; refusing actuator command")
        return wire, state

    def _ensure_command_subscriber(self):
        deadline = time.monotonic() + self._command_connect_timeout_s
        while True:
            while self._command_publisher.poll(0):
                subscription = self._command_publisher.recv()
                if subscription == b"\x01":
                    self._command_subscribed = True
                elif subscription == b"\x00":
                    self._command_subscribed = False
            if self._command_subscribed:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError("No action subscriber is ready; refusing silently dropped ZMQ commands")
            self._command_publisher.poll(max(1, int(remaining * 1000)))

    def publish_action(self, action, target="real"):
        vector = np.asarray(action, dtype=np.float32)
        if vector.shape != (self._robot.total_action_dim,) or not np.isfinite(vector).all():
            raise ValueError("Refusing an invalid actuator command")
        self._ensure_command_subscriber()
        wire, feedback = self._require_complete_snapshot()
        identity = self._chunk_identity or {}
        index = self._issued_in_chunk
        expected = None
        if target == self._chunk_target and self._pending_chunk is not None:
            if index >= len(self._pending_chunk):
                raise ValueError("Rolling chunk already completely issued; refusing duplicate command")
            expected = self._pending_chunk[index]
        result = super().publish_action(vector, target=target)
        if expected is not None:
            self._issued_in_chunk += 1
        self._execution_trace.write("command_issued", **identity, target=target,
            chunk_action_index=index if expected is not None else None,
            source_timestamp=float(wire.t), command=vector.tolist(), feedback=feedback.tolist(),
            processing_override_max_abs=float(np.max(np.abs(vector - expected))) if expected is not None else None,
            hardware_execution_acknowledged=False)
        if expected is not None and self._action_issued_callback is not None:
            self._action_issued_callback()
        return result


class ReadOnlyContractObservation(ContractZmqTransport):
    """Only a SUB observation socket; no model or action publisher is created."""

    def __init__(self, config, robot):
        self._init_wire_guards(config)
        self._config, self._robot = config, robot
        self._contract = load_contract(config.policy.backend_options.contract_path)
        validate_robot(robot, self._contract)
        self._resample = {"bicubic": Image.Resampling.BICUBIC,
                          "bilinear": Image.Resampling.BILINEAR}[
                              self._contract["image"].get("interpolation", "bicubic")]
        from examples.deployment.eva.adapters.eva import read_only_observation_reader
        self._reader = read_only_observation_reader(config, robot)

    def publish_action(self, *args, **kwargs):
        raise RuntimeError("Read-only observation capture cannot publish actions")

    def close(self):
        self._reader.close()


@TRANSPORT_REGISTRY.register("starvla_dataset")
def build_contract_dataset(config, robot):
    return ContractDatasetTransport(config, robot, config.transport.dataset_dir,
                                    config.transport.episode_id)


@TRANSPORT_REGISTRY.register("starvla_zmq")
def build_contract_zmq(config, robot):
    return ContractZmqTransport(config, robot)
