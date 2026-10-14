"""Run native G2 observation/action code with Mock GDK over private IPC sockets.

Recorded observations are injected explicitly; this validates serialization and
execution plumbing, not physical robot dynamics or task success.
"""

import importlib.util
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_mock_node(eva_root):
    directory = Path(eva_root) / "examples/hardware/agibot_g2"
    mock = _load(directory / "agibot_gdk_mock.py", "eva_validation_mock_gdk")
    previous = sys.modules.get("agibot_gdk")
    try:
        sys.modules["agibot_gdk"] = mock
        native = _load(directory / "node.py", "eva_validation_g2_node")
    finally:
        if previous is None:
            sys.modules.pop("agibot_gdk", None)
        else:
            sys.modules["agibot_gdk"] = previous
    assert native.agibot_gdk is mock
    return native, mock


class G2WireMock:
    def __init__(self, eva_root, directory):
        import zmq
        self.native, self.gdk = load_mock_node(eva_root)
        self.node = self.native.AgibotHardwareNode.__new__(self.native.AgibotHardwareNode)
        self.node._obs_hz = 1000.0
        self.node._control_hz = 100.0  # native node's default servo period: 10 ms
        self.node.robot = self.gdk.Robot()
        self.node.cam_mapping = {
            self.gdk.CameraType.kHeadColor: "top_head",
            self.gdk.CameraType.kHandLeftColor: "hand_left",
            self.gdk.CameraType.kHandRightColor: "hand_right",
        }
        self.node.camera = SimpleNamespace(get_latest_image=self._camera)
        self._pixels = {}
        self.received = []
        self.servo_requests = []
        self._context = zmq.Context.instance()
        # Paths are always generated locally; there are no real robot endpoints.
        directory = Path(directory).resolve()
        self.obs_endpoint = f"ipc://{directory / 'observation.sock'}"
        self.action_endpoint = f"ipc://{directory / 'action.sock'}"
        self._pub = self._context.socket(zmq.PUB)
        self._pub.bind(self.obs_endpoint)
        self.node._action_sub = self._context.socket(zmq.SUB)
        self.node._action_sub.setsockopt(zmq.SUBSCRIBE, b"")
        from ..adapters.hardware import configure_action_subscriber
        configure_action_subscriber(self.node, self.action_endpoint)
        self.node._obs_pub = SimpleNamespace(send=self._send_observation)
        decode = self.native.unpack_action
        def capture(payload):
            message = decode(payload)
            if message.target == "real":
                self.received.append(np.asarray(message.action, dtype=np.float32).copy())
            return message
        self.native.unpack_action = capture
        servo = self.node.robot.joint_servo_control
        def capture_servo(request):
            self.servo_requests.append(dict(joint_names=list(request.joint_names),
                joint_positions=list(request.joint_positions), control_period=request.control_period))
            self.node._is_running = False
            return servo(request)
        self.node.robot.joint_servo_control = capture_servo
        self.node.target_qpos = self.node._read_current_qpos()

    def _camera(self, camera_type, timeout_ms=10):
        import cv2
        image = self._pixels[self.node.cam_mapping[camera_type]]
        ok, encoded = cv2.imencode(".png", image)  # lossless, pixel parity check
        if not ok:
            raise RuntimeError("Cannot encode simulated camera frame")
        return SimpleNamespace(data=encoded.tobytes())

    def _send_observation(self, payload):
        self._pub.send(payload)
        self.node._is_running = False  # execute exactly one native observation tick

    def push_frame(self, frame):
        self._pixels = frame.images
        state = np.asarray(frame.state_qpos, dtype=np.float32)
        if state.shape != (24,):
            raise ValueError("G2 mock requires 24D recorded state")
        self.node.robot.current_state = dict(zip(self.native.JOINT_NAMES, state, strict=True))
        self.node._is_running = True
        self.node._observation_loop()

    def wait_frame(self, transport, frame, timeout=2.0):
        deadline = time.monotonic() + timeout
        # Re-publish only observations while the PUB/SUB subscription settles.
        # Commands are never retried by this harness.
        while time.monotonic() < deadline:
            self.push_frame(frame)
            if transport._reader._sub.poll(20):
                actual = transport.get_frame()
                if actual is not None:
                    return actual
        raise RuntimeError("Mock observation did not reach EVA over ZMQ")

    def consume_command(self, timeout_ms=2000):
        if not self.node._action_sub.poll(timeout_ms):
            raise RuntimeError("EVA command did not reach the mock execution node")
        before = len(self.received)
        self.node._is_running = True
        self.node.serve_forever()  # native drain -> JointServoControlReq -> Mock GDK
        if len(self.received) != before + 1:
            raise RuntimeError("Expected exactly one real command per native execution tick")
        vector = np.asarray(self.servo_requests[-1]["joint_positions"], dtype=np.float32)
        np.testing.assert_array_equal(vector, self.received[-1])
        if self.servo_requests[-1]["joint_names"] != self.native.JOINT_NAMES:
            raise RuntimeError("GDK command joint order changed")
        return vector

    def close(self):
        self.node._action_sub.close(linger=0)
        self._pub.close(linger=0)
