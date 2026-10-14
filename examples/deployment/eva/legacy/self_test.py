"""CPU-only contract and WebSocket round-trip test for the EVA bridge."""

from __future__ import annotations

import asyncio
import os
import socket
import threading
import time
from pathlib import Path

import numpy as np

from .client import EvaPolicyClient
from .contract import CAMERA_ORDER, STATE_ACTION_SLICES, server_metadata
from .observation import prepare_policy_example
from .replay_transport import LeRobotReplayTransport
from .server import EvaWebsocketPolicyServer, normalize_g2_state


class _IdentityTransform:
    def apply(self, data):
        return data


class _IdentityProcessor:
    state_keys = [f"state.{key}" for key in STATE_ACTION_SLICES]
    transform = _IdentityTransform()


class _MockPolicy:
    def __init__(self):
        self.reset_count = 0
        self.inference_count = 0

    def reset(self):
        self.reset_count += 1

    def infer_eva(self, payload):
        images = payload["images_rgb"]
        assert len(images) == 3
        assert all(np.asarray(image).shape == (224, 224, 3) for image in images)
        assert np.asarray(payload["state_qpos"]).shape == (24,)
        assert isinstance(payload["prompt"], str) and payload["prompt"].strip()
        self.inference_count += 1
        return {"actions": np.full((16, 24), self.inference_count, dtype=np.float32)}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> None:
    # Channel order test: raw G2 BGR blue becomes RGB blue (channel 2).
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[..., 0] = 255
    raw_images = {name: image.copy() for name in CAMERA_ORDER}
    example = prepare_policy_example(raw_images, np.zeros(24, dtype=np.float32), "mock g2 task")
    assert len(example["images_rgb"]) == 3
    assert all(frame.shape == (224, 224, 3) for frame in example["images_rgb"])
    assert all(int(frame[100, 100, 2]) == 255 for frame in example["images_rgb"])

    normalized = normalize_g2_state(np.arange(24, dtype=np.float32), _IdentityProcessor())
    assert normalized.shape == (1, 24)
    assert normalized.dtype == np.float16
    np.testing.assert_allclose(normalized[0], np.arange(24, dtype=np.float16))

    port = _free_port()
    mock_policy = _MockPolicy()
    server = EvaWebsocketPolicyServer(
        mock_policy,
        host="127.0.0.1",
        port=port,
        metadata=server_metadata("mock-checkpoint", ["agibot_g2"]),
    )
    thread = threading.Thread(target=lambda: asyncio.run(server.run()), daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    client = None
    while time.monotonic() < deadline:
        try:
            client = EvaPolicyClient("127.0.0.1", port, connect_timeout_s=1)
            break
        except (OSError, TimeoutError):
            time.sleep(0.1)
    if client is None:
        raise RuntimeError("mock EVA policy endpoint did not start")

    with client:
        assert client.metadata["checkpoint"] == "mock-checkpoint"
        client.reset()
        actions = client.infer(raw_images, np.arange(24, dtype=np.float32), "mock g2 task")
        assert actions.shape == (16, 24)
        np.testing.assert_array_equal(actions, np.ones((16, 24), dtype=np.float32))
        dataset = Path(os.environ.get("STARVLA_G2_DATASET", str(
            Path(__file__).resolve().parents[4] / "datasets/AgiBotG2/white_carton_to_left_tan_carton_20261009")))
        if dataset.is_dir():
            with LeRobotReplayTransport(dataset, 0) as replay:
                frame = replay.get_frame()
                assert frame is not None
                actions = client.infer(frame.images, frame.state_qpos, replay.current_task)
                assert actions.shape == (16, 24)
                assert np.isfinite(actions).all()
            assert mock_policy.inference_count == 2
    assert mock_policy.reset_count == 1
    assert mock_policy.inference_count == 2 if dataset.is_dir() else mock_policy.inference_count == 1
    print("EvaClient CPU contract + WebSocket + G2 replay observation round-trip passed")


if __name__ == "__main__":
    main()
