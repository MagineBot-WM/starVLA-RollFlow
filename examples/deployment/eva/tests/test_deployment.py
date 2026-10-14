"""CPU regression: no model weights, physical transport or robot publishing."""

import json
import shutil
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from examples.deployment.eva.bridge_contract import (
    contract_sha256, load_contract, split_state, validate_contract,
)
from examples.deployment.eva.protocol import StreamSession
from examples.deployment.eva.runtime import ROOT, bootstrap, load_config, validate_robot

PACKAGE = ROOT / "examples/deployment/eva"
G2 = PACKAGE / "contracts/agibot_g2_rollflow_h64_c16.json"
LIBERO = PACKAGE / "contracts/libero_rollflow_h32_c8_k4.json"


@pytest.fixture(scope="session", autouse=True)
def eva_runtime():
    bootstrap()


@pytest.mark.parametrize("name", ["agibot_g2_rollflow_h64_c16", "libero_rollflow_h32_c8_k4", "libero_rollflow_h64_c8"])
def test_contract_presets(name):
    contract = load_contract(PACKAGE / f"contracts/{name}.json")
    assert contract["execution"]["execute_steps"] == contract["action"]["chunk_size"]
    assert len(contract_sha256(contract)) == 64


@pytest.mark.parametrize("section,key,value", [
    ("image", "resize", "pad"), ("image", "color", "bgr"),
    ("image", "hw", [224, 0]), ("action", "dim", True),
    ("execution", "execute_steps", 5), ("execution", "control_hz", float("nan")),
    ("model", "horizon", 8), ("eva", "source_color", "unknown"),
    ("eva", "joint_names", ["joint"] * 24), ("eva", "state_groups", {}),
])
def test_contract_rejects_invalid_settings(section, key, value):
    contract = load_contract(G2)
    contract[section][key] = value
    with pytest.raises(ValueError):
        validate_contract(contract)


def test_state_slices_and_order_rejected():
    contract = load_contract(G2)
    contract["state"]["fields"][1]["slice"] = [0, 1]
    with pytest.raises(ValueError, match="overlap"):
        validate_contract(contract)
    contract = load_contract(G2)
    contract["eva"]["state_groups"].reverse()
    with pytest.raises(ValueError, match="order"):
        validate_contract(contract)
    with pytest.raises(ValueError):
        split_state(np.zeros((1, 24)), load_contract(G2)["state"]["fields"], 24)
    raw = np.arange(24, dtype=np.float32)
    values = split_state(raw, load_contract(G2)["state"]["fields"], 24)
    np.testing.assert_array_equal(values["state.right_arm"], raw[8:15][None])


@pytest.mark.parametrize("name", ["agibot_g2_live", "g2_replay", "libero_sim", "libero_sim_verified"])
def test_configuration_and_legacy_entry(name):
    config = load_config(PACKAGE / f"configs/{name}.py")
    legacy = load_config(ROOT / f"examples/realRobots/EvaClient/configs/{name}.py")
    assert config.policy.backend_options.contract_path == legacy.policy.backend_options.contract_path
    assert not config.transport.resize_pad
    assert (config.transport.image_height, config.transport.image_width) == (224, 224)
    assert config.inference_strategies.rolling.type == ("StarVlaAsyncRollingStrategy" if name == "agibot_g2_live" else "StarVlaRollingStrategy")
    from core.registry import ROBOT_REGISTRY
    robot = ROBOT_REGISTRY.build(config.robot.type)
    validate_robot(robot, load_contract(config.policy.backend_options.contract_path))


def test_joint_names_must_match_hardware():
    from core.registry import ROBOT_REGISTRY
    contract = load_contract(G2)
    contract["eva"]["joint_names"][0:2] = reversed(contract["eva"]["joint_names"][0:2])
    with pytest.raises(ValueError, match="joint order"):
        validate_robot(ROBOT_REGISTRY.build("agibot_g2"), contract)


def test_independent_eva_checkout(tmp_path):
    checkout = tmp_path / "separate-eva"
    for directory in ("src", "configs", "examples/hardware/agibot_g2"):
        shutil.copytree(ROOT / "EVA-CLIENT" / directory, checkout / directory,
                        ignore=shutil.ignore_patterns("__pycache__"))
    output = tmp_path / "doctor.json"
    subprocess.run([sys.executable, "-m", "examples.deployment.eva", "doctor", "--eva-root", str(checkout),
        "--config", str(PACKAGE / "configs/g2_replay.py"), "--output", str(output)],
        cwd=ROOT, check=True, capture_output=True)
    report = json.loads(output.read_text())
    assert all(Path(entry["path"]).is_relative_to(checkout) for entry in report["eva_sources"])


def test_different_eva_cannot_silently_reuse_imports(tmp_path):
    from examples.deployment.eva.runtime import bootstrap
    checkout = tmp_path / "other-eva"
    (checkout / "src/core").mkdir(parents=True)
    (checkout / "src/core/config.py").write_text("# mock upstream")
    with pytest.raises(RuntimeError, match="different EVA checkout"):
        bootstrap(checkout)


@pytest.mark.parametrize("path,config_name", [(G2, "g2_replay"), (LIBERO, "libero_sim_verified")])
def test_pixels_match_native_preprocessing(path, config_name):
    from core.app.handlers.recording import build_policy_observation
    from core.app.state import RuntimeState
    from core.registry import ROBOT_REGISTRY
    from core.types import Observation
    from examples.deployment.eva.eva_extension.image_transport import ContractImageMixin
    from examples.deployment.eva.eva_extension.starvla_policy import StarVlaEvaPolicyClient
    config = load_config(PACKAGE / f"configs/{config_name}.py")
    contract = load_contract(path)
    robot = ROBOT_REGISTRY.build(config.robot.type)
    rng = np.random.default_rng(7)
    pixels = {camera["eva_key"]: rng.integers(0, 256, (239, 317, 3), dtype=np.uint8)
              for camera in contract["cameras"]}
    frame = Observation(images=pixels, state_qpos=np.zeros(robot.total_action_dim), timestamp=1.0)
    prepare = ContractImageMixin.__new__(ContractImageMixin)
    prepare._contract, prepare._config = contract, config
    prepare._resample = Image.Resampling.BICUBIC if path == G2 else Image.Resampling.BILINEAR
    prepared = prepare._prepare_frame(frame) if path == G2 else frame
    observation = build_policy_observation(prepared, "task", config, RuntimeState(robot=robot, transport=None))
    policy = StarVlaEvaPolicyClient.__new__(StarVlaEvaPolicyClient)
    policy.contract = contract
    actual = policy._ordered_images(observation["images"])
    for index, camera in enumerate(contract["cameras"]):
        reference = pixels[camera["eva_key"]]
        if contract["eva"]["source_color"] == "bgr":
            reference = reference[..., ::-1]
        if contract["image"]["rotate_180"]:
            reference = reference[::-1, ::-1]
        reference = np.asarray(Image.fromarray(reference).resize((224, 224), prepare._resample))
        np.testing.assert_array_equal(actual[index], reference)


def test_live_snapshot_complete_and_fresh(monkeypatch):
    from core.registry import ROBOT_REGISTRY
    from examples.deployment.eva.eva_extension.image_transport import ContractZmqTransport
    config = load_config(PACKAGE / "configs/agibot_g2_live_sync.py")
    robot = ROBOT_REGISTRY.build(config.robot.type)
    contract = load_contract(G2)
    wire = SimpleNamespace(t=100000.0, images={c["eva_key"]: np.zeros((31, 43, 3), np.uint8)
                                           for c in contract["cameras"]},
                           state={g.name: np.zeros(g.dof) for g in robot.actuator_groups})
    transport = ContractZmqTransport.__new__(ContractZmqTransport)
    transport._contract, transport._config, transport._robot = contract, config, robot
    transport._resample = Image.Resampling.BICUBIC
    transport._reader = SimpleNamespace(_drain_latest=lambda: wire)
    transport._last_source_timestamp = None
    transport._last_progress, transport._max_frame_age_s = 10.0, 2.0
    clock = [10.0]
    monkeypatch.setattr("examples.deployment.eva.eva_extension.image_transport.time.monotonic", lambda: clock[0])
    frame = transport.get_frame()
    assert frame.timestamp == wire.t  # source and client clocks have different origins
    assert frame.state_qpos.shape == (24,)
    camera = contract["cameras"][0]["eva_key"]
    missing = wire.images.pop(camera)
    with pytest.raises(ValueError, match="missing policy camera"):
        transport.get_frame()
    wire.images[camera] = missing
    group = robot.actuator_groups[0].name
    wire.state.pop(group)
    with pytest.raises(ValueError, match="Wire state"):
        transport.get_frame()
    wire.state[group] = np.zeros(robot.actuator_groups[0].dof)
    clock[0] = 13.0
    with pytest.raises(ValueError, match="stopped advancing"):
        transport.get_frame()
    wire.t += 1
    assert transport.get_frame() is not None
    wire.t -= 2
    with pytest.raises(ValueError, match="backwards"):
        transport.get_frame()


def test_dataset_cannot_substitute_black_frame():
    from examples.deployment.eva.adapters.eva import complete_dataset_frame
    transport = SimpleNamespace(_lock=threading.Lock(), _frame_index=0, _n_steps=1,
        _obs_state=np.zeros((1, 24)), _frame_source=lambda key: None,
        _frame_timestamp=lambda index: 0.0, is_shutdown=lambda: False)
    with pytest.raises(ValueError, match="refusing substituted pixels"):
        complete_dataset_frame(transport, ["top_head"])


def test_rolling_schedule_and_warmup():
    from examples.deployment.eva.eva_extension.rolling_strategy import StarVlaRollingStrategy
    strategy = StarVlaRollingStrategy(contract_path=str(G2))
    calls = []
    fetch = lambda *_: calls.append(True) or np.zeros((16, 24), np.float32)
    assert strategy.prepare_warmup_chunk("task", fetch).shape == (0, 24)
    assert not calls
    assert strategy.take_or_fetch_chunk("task", fetch).shape == (16, 24)
    assert len(calls) == 1
    for bad in (np.zeros((5, 24)), np.full((16, 24), np.nan)):
        with pytest.raises(ValueError):
            strategy._crop_chunk(bad)
    with pytest.raises(ValueError):
        StarVlaRollingStrategy(contract_path=str(G2), execute_horizon=5)


def test_native_setup_discards_validation_without_advancing_execution():
    from core.app.handlers.control import run_setup
    from core.app.handlers.io import ensure_action_chunk, run_warmup_and_start
    from core.app.state import RuntimeState, SessionState
    from core.types import Observation
    from core.app.state import SessionMode
    from core.registry import ROBOT_REGISTRY
    from examples.deployment.eva.eva_extension.rolling_strategy import StarVlaRollingStrategy
    config = load_config(PACKAGE / "configs/agibot_g2_live_sync.py")
    robot = ROBOT_REGISTRY.build(config.robot.type)
    class CounterPolicy:
        def __init__(self):
            self.cache = self.calls = self.resets = 0
        def infer(self, observation):
            self.cache += 1
            self.calls += 1
            return {"actions": np.full((16, 24), self.cache, np.float32), "action_mode": "joint"}
        def reset(self):
            self.cache = 0
            self.resets += 1
    observation = policy_observation()
    frame = Observation(images=observation["images"], state_qpos=observation["state"], timestamp=1.0)
    # SIM on an online transport skips physical setup trajectories. Any publishing
    # is an assertion failure; only native setup and action buffering are called.
    def refuse_publish(*args, **kwargs):
        raise AssertionError("CPU setup regression must never publish")
    transport = SimpleNamespace(is_offline=lambda: False, is_shutdown=lambda: False,
        get_latest_qpos=lambda: np.zeros(24), get_frame=lambda: frame,
        create_rate=lambda hz: SimpleNamespace(sleep=lambda: None), publish_action=refuse_publish)
    policy = CounterPolicy()
    runtime = RuntimeState(robot=robot, transport=transport, policy=policy)
    runtime.infer_strategy = StarVlaRollingStrategy(contract_path=str(G2))
    session = SessionState(mode=SessionMode.SIM, selected_task="place carton")
    assert run_setup(config, runtime, session, fail_fast=True)
    assert policy.calls == 1 and policy.cache == 0
    assert run_warmup_and_start(config, runtime, session)
    assert policy.calls == 1  # no discarded prewarm predictions
    assert ensure_action_chunk(config, runtime, session)
    np.testing.assert_array_equal(session.action_chunk[:, 0], np.ones(16))
    assert policy.calls == 2 and policy.cache == 1


def test_releasing_client_closes_websocket():
    from examples.deployment.eva.eva_extension.starvla_policy import StarVlaEvaPolicyClient
    calls = []
    policy = StarVlaEvaPolicyClient.__new__(StarVlaEvaPolicyClient)
    policy._ws = SimpleNamespace(close=lambda: calls.append(True))
    del policy
    assert calls == [True]


def test_stateless_prefix_schedule(tmp_path):
    from examples.deployment.eva.eva_extension.rolling_strategy import StarVlaChunkStrategy
    contract = load_contract(G2)
    contract["execution"].update(kind="chunk", execute_steps=5)
    path = tmp_path / "chunk.json"
    path.write_text(json.dumps(contract))
    strategy = StarVlaChunkStrategy(contract_path=str(path))
    assert strategy.take_or_fetch_chunk("task", lambda *_: np.zeros((16, 24))).shape == (5, 24)


def test_libero_gripper_exact_threshold():
    from examples.deployment.eva.eva_extension.libero_sim import LiberoVirtualRobot
    actions = np.zeros((3, 7), np.float32)
    actions[:, 6] = [0.49, 0.5, 0.51]
    result = LiberoVirtualRobot().snap_grippers(actions, 0.5, -1, 1)
    np.testing.assert_array_equal(result[:, 6], [1, 1, -1])


def test_protocol_reset_sequence_and_session():
    session = StreamSession(load_contract(G2))
    msg = dict(type="infer", request_id="test", payload={}, bridge={
        "session_id": session.session_id, "contract_sha256": session.contract_hash, "sequence": 0})
    calls = []
    def fallback(request):
        calls.append(request)
        return {"status": "ok", "request_id": "test", "data": {"actions": []}}
    assert session.route(msg, fallback, lambda _: None)["status"] == "ok"
    assert session.route(msg, fallback, lambda _: None)["status"] == "error"
    assert len(calls) == 1
    msg["bridge"]["sequence"] = True
    assert session.route(msg, fallback, lambda _: None)["status"] == "error"
    msg["bridge"]["sequence"] = 1
    def failure(_):
        return {"status": "error", "error": {"message": "model failed"}}
    assert session.route(msg, failure, lambda _: None)["status"] == "error"
    assert session.needs_reset
    assert session.route(msg, fallback, lambda _: None)["status"] == "error"
    msg["type"] = "reset"
    assert session.route(msg, fallback, lambda _: None)["status"] == "ok"
    assert session.sequence == 0 and not session.needs_reset
    msg["type"], msg["bridge"]["sequence"] = "infer", 0
    msg["bridge"]["contract_sha256"] = "wrong"
    assert session.route(msg, fallback, lambda _: None)["status"] == "error"
    msg["bridge"]["contract_sha256"], msg["bridge"]["session_id"] = session.contract_hash, "wrong"
    assert session.route(msg, fallback, lambda _: None)["status"] == "error"


@pytest.fixture
def loopback():
    from websockets.sync.server import serve
    from websockets.exceptions import ConnectionClosed
    from examples.deployment.eva import msgpack_numpy as codec
    state = SimpleNamespace(mode="ok", calls=0, examples=[], contract=load_contract(G2))
    def handler(socket):
        contract = state.contract
        session = StreamSession(contract)
        socket.send(codec.Packer().pack({"contract": contract, "protocol": contract["protocol"],
            "contract_sha256": session.contract_hash, "session_id": session.session_id,
            "action_chunk_size": 16, "available_unnorm_keys": [contract["action"]["unnorm_key"]]}))
        def predict(msg):
            state.calls += 1
            state.examples.append(msg.get("payload", {}).get("examples"))
            if state.mode == "timeout":
                time.sleep(0.15)
            actions = np.zeros((1, 5 if state.mode == "bad_shape" else 16, 24), np.float32)
            if state.mode == "bad_numeric":
                actions = "not numeric"
            elif state.mode == "ramp":
                actions = (np.arange(16 * 24, dtype=np.float32).reshape(1, 16, 24) / 1000 + state.calls)
            return {"status": "ok", "request_id": msg["request_id"], "data": {
                "actions": actions,
                "diagnostics": {"normalized_state": np.zeros((1, 24), np.float32)}}}
        try:
            for raw in socket:
                reply = session.route(codec.unpackb(raw), predict, lambda _: None)
                if state.mode == "wrong_sequence" and "sequence" in reply.get("data", {}):
                    reply["data"]["sequence"] += 1
                if state.mode == "boolean_sequence" and "sequence" in reply.get("data", {}):
                    reply["data"]["sequence"] = False
                if state.mode == "bad_reset" and reply.get("data", {}).get("reset"):
                    reply["data"]["reset"] = False
                socket.send(codec.Packer().pack(reply))
        except ConnectionClosed:
            pass
    with serve(handler, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        state.port = server.socket.getsockname()[1]
        yield state
        server.shutdown()
        thread.join(timeout=2)


def policy_observation():
    return {"prompt": "place carton", "state": np.zeros(24, np.float32),
            "images": {c["eva_key"]: np.zeros((224, 224, 3), np.uint8) for c in load_contract(G2)["cameras"]}}


def test_loopback_trace_reset_and_legacy(loopback, tmp_path):
    from examples.deployment.eva.eva_extension.starvla_policy import StarVlaEvaPolicyClient
    from examples.realRobots.EvaClient.eva_extension.starvla_policy import StarVlaEvaPolicyClient as Legacy
    assert Legacy is StarVlaEvaPolicyClient
    policy = StarVlaEvaPolicyClient("127.0.0.1", loopback.port, str(G2), trace_dir=str(tmp_path), trace_limit=1)
    try:
        assert policy.infer(policy_observation())["actions"].shape == (16, 24)
        assert policy._sequence == 1
        arrays = np.load(next(tmp_path.glob("*.npz")))
        assert arrays["normalized_state"].shape == (1, 24)
        assert arrays["raw_policy_actions"].shape == (16, 24)
        policy.reset(seed=7)
        assert policy._sequence == 0
        policy.infer(policy_observation())
        assert len(list(tmp_path.glob("*.npz"))) == 1
    finally:
        policy.close()


@pytest.mark.parametrize("mode", ["timeout", "bad_shape", "wrong_sequence", "bad_numeric", "boolean_sequence"])
def test_uncertain_or_invalid_chunk_closes_stream(loopback, mode):
    from policy_client.base import PolicyRequestError
    from examples.deployment.eva.eva_extension.starvla_policy import StarVlaEvaPolicyClient
    loopback.mode = mode
    policy = StarVlaEvaPolicyClient("127.0.0.1", loopback.port, str(G2), timeout_s=0.05 if mode == "timeout" else 2)
    with pytest.raises(PolicyRequestError):
        policy.infer(policy_observation())
    assert policy._closed
    with pytest.raises(PolicyRequestError, match="closed"):
        policy.infer(policy_observation())
    assert loopback.calls == 1


def test_contract_hash_mismatch_closes_connection(loopback, tmp_path):
    from policy_client.base import PolicyConnectionError
    from examples.deployment.eva.eva_extension.starvla_policy import StarVlaEvaPolicyClient
    contract = load_contract(G2)
    contract["name"] = "other-valid-contract"
    path = tmp_path / "mismatched.json"
    path.write_text(json.dumps(contract))
    with pytest.raises(PolicyConnectionError, match="hash mismatch"):
        StarVlaEvaPolicyClient("127.0.0.1", loopback.port, str(path))
    assert loopback.calls == 0


def test_concurrent_inference_is_serialized(loopback):
    from concurrent.futures import ThreadPoolExecutor
    from examples.deployment.eva.eva_extension.starvla_policy import StarVlaEvaPolicyClient
    policy = StarVlaEvaPolicyClient("127.0.0.1", loopback.port, str(G2))
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(policy.infer, policy_observation()) for _ in range(4)]
            assert all(future.result()["actions"].shape == (16, 24) for future in futures)
        assert policy._sequence == loopback.calls == 4
    finally:
        policy.close()


def test_trace_io_failure_preserves_returned_chunk(loopback, tmp_path):
    from examples.deployment.eva.eva_extension.starvla_policy import StarVlaEvaPolicyClient
    invalid_dir = tmp_path / "file"
    invalid_dir.write_text("not a directory")
    policy = StarVlaEvaPolicyClient("127.0.0.1", loopback.port, str(G2), trace_dir=str(invalid_dir))
    try:
        assert policy.infer(policy_observation())["actions"].shape == (16, 24)
        assert not policy._closed
    finally:
        policy.close()


def test_cpu_doctor_handshake_only(loopback, tmp_path):
    output = tmp_path / "doctor.json"
    subprocess.run([sys.executable, "-m", "examples.deployment.eva", "doctor",
        "--contract", str(G2), "--host", "127.0.0.1", "--port", str(loopback.port),
        "--output", str(output)], cwd=ROOT, check=True, capture_output=True)
    report = json.loads(output.read_text())
    assert report["server"]["contract_sha256"] == contract_sha256(load_contract(G2))
    assert not report["action_publishing"] and not report["inference"]
    assert loopback.calls == 0


def test_client_import_does_not_load_model_dependencies():
    code = "from examples.deployment.eva import runtime, bridge_contract, protocol; import sys; assert 'torch' not in sys.modules; assert 'starVLA' not in sys.modules; assert 'deployment.model_server.policy_wrapper' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_execution_package_needs_no_model_source(tmp_path):
    package_root = tmp_path / "execution-machine"
    ignore = shutil.ignore_patterns("__pycache__", "validation", "*.npz")
    shutil.copytree(PACKAGE, package_root / "examples/deployment/eva", ignore=ignore)
    for directory in ("src", "configs", "examples/hardware/agibot_g2"):
        shutil.copytree(ROOT / "EVA-CLIENT" / directory, package_root / "EVA-CLIENT" / directory, ignore=ignore)
    code = "from pathlib import Path; from examples.deployment.eva.runtime import load_config; config=load_config(Path('examples/deployment/eva/configs/agibot_g2_live_sync.py')); import sys; assert 'torch' not in sys.modules; assert 'starVLA' not in sys.modules; assert config.transport.resize_pad is False"
    subprocess.run([sys.executable, "-c", code], cwd=package_root, check=True, capture_output=True)


def test_starvla_api_upgrade_fails_explicitly():
    from examples.deployment.eva.adapters.starvla import CheckpointAdapter
    with pytest.raises(RuntimeError, match="wrapper API changed"):
        CheckpointAdapter(SimpleNamespace(predict_action=lambda x: None), "g2")


def test_prompt_changes_require_reset(loopback):
    from policy_client.base import PolicyRequestError
    from examples.deployment.eva.eva_extension.starvla_policy import StarVlaEvaPolicyClient
    policy = StarVlaEvaPolicyClient("127.0.0.1", loopback.port, str(G2))
    try:
        observation = policy_observation()
        policy.infer(observation)
        observation["prompt"] = "different task"
        with pytest.raises(PolicyRequestError, match="Task changed"):
            policy.infer(observation)
        assert loopback.calls == 1
        policy.reset()
        policy.infer(observation)
        assert loopback.calls == 2
    finally:
        policy.close()


def test_reset_requires_acknowledgement(loopback):
    from policy_client.base import PolicyRequestError
    from examples.deployment.eva.eva_extension.starvla_policy import StarVlaEvaPolicyClient
    policy = StarVlaEvaPolicyClient("127.0.0.1", loopback.port, str(G2))
    loopback.mode = "bad_reset"
    with pytest.raises(PolicyRequestError, match="acknowledge"):
        policy.reset()
    assert policy._closed
    policy.reset()  # native teardown of a closed stream is harmless


def test_late_inference_is_not_executed(loopback):
    from policy_client.base import PolicyRequestError
    from examples.deployment.eva.eva_extension.starvla_policy import StarVlaEvaPolicyClient
    policy = StarVlaEvaPolicyClient("127.0.0.1", loopback.port, str(G2),
                                   timeout_s=2, max_inference_latency_s=0.03)
    loopback.mode = "timeout"
    with pytest.raises(PolicyRequestError, match="latency limit"):
        policy.infer(policy_observation())
    assert policy._closed and policy._sequence == 0


def test_native_force_reconnect_releases_old_client(loopback):
    from core.registry import ROBOT_REGISTRY
    from core.app.handlers.io import connect_policy
    from core.app.state import RuntimeState, SessionState
    config = load_config(PACKAGE / "configs/g2_replay.py")
    config.policy.host, config.policy.port = "127.0.0.1", loopback.port
    runtime = RuntimeState(robot=ROBOT_REGISTRY.build("agibot_g2"), transport=SimpleNamespace())
    session = SessionState()
    assert connect_policy(config, runtime, session)
    old = runtime.policy
    assert connect_policy(config, runtime, session, force_reconnect=True)
    assert old._closed and runtime.policy is not old
    runtime.policy.close()


def test_hardware_order_detects_driver_changes(tmp_path):
    from examples.deployment.eva.adapters.hardware import inspect_hardware_joint_order
    contract = load_contract(G2)
    path = tmp_path / contract["eva"]["hardware_node"]
    path.parent.mkdir(parents=True)
    names = contract["eva"]["hardware_joint_names"]
    path.write_text(f"JOINT_NAMES = {names!r}\n")
    report = inspect_hardware_joint_order(tmp_path, contract)
    assert report["joint_names"][7] == "idx31_gripper_l_inner_joint1"
    assert report["eva_visualization_joint_names"][7] == "idx41_gripper_l_outer_joint1"
    names[0], names[1] = names[1], names[0]
    with pytest.raises(ValueError, match="Hardware JOINT_NAMES"):
        inspect_hardware_joint_order(tmp_path, contract)


@pytest.mark.parametrize("mode", ["real", "sim"])
def test_native_zmq_websocket_execution_and_gdk(loopback, tmp_path, mode):
    from core.registry import ROBOT_REGISTRY, TRANSPORT_REGISTRY
    from core.types import Observation
    from core.app.state import RuntimeState, SessionState, SessionMode, SessionStatus
    from core.app.handlers.control import publish_next_action
    from core.app.handlers.utils import reset_run_state
    from examples.deployment.eva.eva_extension.rolling_strategy import StarVlaRollingStrategy
    from examples.deployment.eva.simulation.g2_wire import G2WireMock
    config = load_config(PACKAGE / "configs/agibot_g2_live_sync.py")
    config.policy.host, config.policy.port = "127.0.0.1", loopback.port
    config.transport.trace_dir = str(tmp_path)
    robot = ROBOT_REGISTRY.build("agibot_g2")
    node = G2WireMock(ROOT / "EVA-CLIENT", tmp_path)
    config.transport.sub_endpoint, config.transport.pub_endpoint = node.obs_endpoint, node.action_endpoint
    transport = TRANSPORT_REGISTRY.build(config.transport.type, config, robot)
    import zmq
    assert transport._reader._sub.getsockopt(zmq.CONFLATE) == 1
    assert transport._qpos_reader._sub.getsockopt(zmq.CONFLATE) == 1
    strategy = StarVlaRollingStrategy(contract_path=str(G2))
    assert strategy.sync_wait_max_ticks == 0
    runtime = RuntimeState(robot=robot, transport=transport, infer_strategy=strategy)
    session = SessionState(mode=SessionMode(mode), status=SessionStatus.RUNNING,
                           selected_task="place carton", is_setup_done=True)
    loopback.mode = "ramp"
    pixels = {c["eva_key"]: np.arange(39*51*3, dtype=np.uint8).reshape(39,51,3)
              for c in load_contract(G2)["cameras"]}
    frame = Observation(images=pixels, state_qpos=np.arange(24, dtype=np.float32)/100, timestamp=1)
    try:
        for index in range(32):
            node.wait_frame(transport, frame)
            assert publish_next_action(config, runtime, session)
            if mode == "real":
                command = node.consume_command()
                np.testing.assert_array_equal(command, session.action_chunk[session.chunk_index-1])
        assert loopback.calls == 2 and session.step_index == 32
        assert transport._issued_in_chunk == 16
        np.testing.assert_array_equal(loopback.examples[0][0]["state"], frame.state_qpos)
        expected = np.asarray(Image.fromarray(pixels["top_head"][...,::-1]).resize((224,224), Image.Resampling.BICUBIC))
        np.testing.assert_array_equal(loopback.examples[0][0]["image"][0], expected)
        if mode == "real":
            assert len(node.received) == len(node.servo_requests) == 32
            events = [json.loads(line) for line in (tmp_path/"execution.jsonl").read_text().splitlines()]
            issued = [event for event in events if event["event"] == "command_issued"]
            assert [event["chunk_action_index"] for event in issued] == list(range(16))*2
            assert all(event["processing_override_max_abs"] == 0 for event in issued)
        # Interrupt midway through a new chunk and use native teardown/reset.
        node.wait_frame(transport, frame)
        assert publish_next_action(config, runtime, session)
        if mode == "real": node.consume_command()
        reset_run_state(config, runtime, session)
        assert session.action_chunk is None and runtime.policy._sequence == 0
        assert transport._pending_chunk is None
        node.wait_frame(transport, frame)
        assert publish_next_action(config, runtime, session)
        if mode == "real": node.consume_command()
        assert runtime.policy.last_chunk["sequence"] == 0
        # A stopped source blocks further cached actions, within the current chunk.
        transport._last_progress -= 2
        with pytest.raises(ValueError, match="stopped advancing"):
            publish_next_action(config, runtime, session)
    finally:
        if runtime.policy: runtime.policy.close()
        transport.close()
        node.close()


def test_no_action_subscriber_stops_native_chunk(loopback, tmp_path):
    from core.registry import ROBOT_REGISTRY, TRANSPORT_REGISTRY
    from core.types import Observation
    from core.app.state import RuntimeState, SessionState, SessionMode
    from core.app.handlers.io import ensure_action_chunk
    from examples.deployment.eva.eva_extension.rolling_strategy import StarVlaRollingStrategy
    from examples.deployment.eva.simulation.g2_wire import G2WireMock
    config = load_config(PACKAGE / "configs/agibot_g2_live_sync.py")
    config.policy.host, config.policy.port = "127.0.0.1", loopback.port
    config.transport.command_connect_timeout_s = 0.05
    robot = ROBOT_REGISTRY.build("agibot_g2")
    node = G2WireMock(ROOT / "EVA-CLIENT", tmp_path)
    node.node._action_sub.close(linger=0)
    config.transport.sub_endpoint, config.transport.pub_endpoint = node.obs_endpoint, node.action_endpoint
    transport = TRANSPORT_REGISTRY.build(config.transport.type, config, robot)
    runtime = RuntimeState(robot=robot, transport=transport,
        infer_strategy=StarVlaRollingStrategy(contract_path=str(G2)))
    frame = Observation(images=policy_observation()["images"], state_qpos=np.zeros(24), timestamp=1)
    session = SessionState(mode=SessionMode.REAL, selected_task="place carton")
    try:
        node.wait_frame(transport, frame)
        assert not ensure_action_chunk(config, runtime, session)
        assert "No action subscriber" in session.last_error
        assert not node.received and session.action_chunk is None
        assert runtime.policy._sequence == 0
    finally:
        if runtime.policy: runtime.policy.close()
        transport.close()
        node.close()


def test_execution_trace_failure_does_not_repeat_command(tmp_path):
    from examples.deployment.eva.execution_trace import ExecutionTrace
    path = tmp_path / "not-a-directory"
    path.write_text("occupied")
    trace = ExecutionTrace(path)
    trace.write("command_issued", command=[0]*24)
    assert trace.count == 0


@pytest.mark.parametrize("field,value", [("action_chunk_size", True), ("session_id", 7),
    ("protocol", "unknown"), ("available_unnorm_keys", "agibot_g2")])
def test_handshake_rejects_malformed_metadata(field, value):
    from examples.deployment.eva.protocol import validate_handshake
    c = load_contract(G2)
    metadata = dict(contract_sha256=contract_sha256(c), action_chunk_size=16,
                    protocol=c["protocol"], session_id="session", available_unnorm_keys=["agibot_g2"])
    metadata[field] = value
    with pytest.raises(ValueError): validate_handshake(metadata, c)


def test_v1_client_without_execution_section(loopback, tmp_path):
    from examples.deployment.eva.bridge_contract import execution_spec
    c = load_contract(ROOT / "examples/realRobots/EvaClient/contracts/agibot_g2_rollflow_h64_c16.json")
    assert "execution" not in c and execution_spec(c)["kind"] == "rolling"


def test_numpy_codec_preserves_native_wire_bytes():
    from examples.deployment.eva import msgpack_numpy
    # The upstream EVA wire codec is independent from the model environment.
    from transport.zmq import pack_observation, unpack_observation, WireObservation
    frame = WireObservation(t=123., images={"top_head":np.zeros((3,4,3),np.uint8)},
                            state={"left_arm":np.arange(8,dtype=np.float32)})
    decoded = msgpack_numpy.unpackb(pack_observation(frame))
    np.testing.assert_array_equal(decoded["state"]["left_arm"], frame.state["left_arm"])
    roundtrip = unpack_observation(msgpack_numpy.packb(decoded))
    np.testing.assert_array_equal(roundtrip.images["top_head"], frame.images["top_head"])


def test_capture_observation_uses_no_action_socket(tmp_path):
    from core.types import Observation
    from examples.deployment.eva.simulation.g2_wire import G2WireMock
    node = G2WireMock(ROOT / "EVA-CLIENT", tmp_path)
    frame = Observation(images=policy_observation()["images"], state_qpos=np.arange(24)/100, timestamp=1)
    output = tmp_path / "capture.json"
    process = subprocess.Popen([sys.executable, "-m", "examples.deployment.eva", "capture",
        "--obs-endpoint", node.obs_endpoint, "--count", "1", "--timeout-s", "3", "--output", str(output)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 5
        while process.poll() is None and time.monotonic() < deadline:
            node.push_frame(frame)
            time.sleep(.02)
        stdout, stderr = process.communicate(timeout=2)
        assert process.returncode == 0, stderr.decode()
        report = json.loads(output.read_text())
        assert not report["action_socket_created"] and not report["model_inference"]
        assert not node.received
        with np.load(output.with_suffix(".npz")) as arrays:
            np.testing.assert_array_equal(arrays["sample_0_raw_state"], np.asarray(frame.state_qpos, np.float32))
            assert arrays["sample_0_model_image_0"].shape == (224,224,3)
    finally:
        if process.poll() is None: process.kill(); process.wait()
        node.close()

@pytest.mark.parametrize('trigger', [4, 8])
def test_async_full_chunks_trigger_after_issue_and_sensor_thread(trigger):
    from functools import partial
    from examples.deployment.eva.eva_extension.async_rolling_strategy import StarVlaAsyncRollingStrategy
    path = PACKAGE / ('contracts/agibot_g2_rollflow_h64_c16_async4.json' if trigger == 4 else
                      'contracts/agibot_g2_rollflow_h64_c16_async.json')
    strategy = StarVlaAsyncRollingStrategy(contract_path=str(path))
    main_thread = threading.get_ident()
    activated, observations = [], []
    transport = SimpleNamespace(set_action_issued_callback=lambda cb: setattr(transport, 'callback', cb),
        begin_policy_chunk=lambda chunk, identity, **kw: activated.append((identity['action_start'], strategy._issued)),
        reset_policy_chunk=lambda: None)
    policy = SimpleNamespace(last_chunk=None, reset=lambda: None)
    runtime = SimpleNamespace(transport=transport, policy=policy)
    def fetch(runtime, prompt, obs):
        assert threading.get_ident() != main_thread
        start = len(observations) * 16
        observations.append(obs['step'])
        runtime.policy.last_chunk = dict(sequence=start//16)
        return np.repeat(np.arange(start, start+16, dtype=np.float32)[:,None], 24, axis=1)
    def observe():
        assert threading.get_ident() == main_thread
        return dict(step=strategy._issued, array=np.array([strategy._issued]))
    try:
        strategy.start_loop('carton', partial(fetch, runtime), observe)
        result = []
        deadline = time.monotonic() + 3
        while len(result) < 32:
            assert time.monotonic() < deadline
            action = strategy.pop_next_action()
            if action is None:
                time.sleep(.001)
                continue
            result.append(float(action[0]))
            # A pop isn't an issue, so no prefetch yet.
            if len(result) == trigger:
                assert len(strategy.requests) == 1
            transport.callback()
        strategy.stop_loop()
        assert result == list(range(32))
        assert activated == [(0, 0), (16, 16)]
        assert [r['observation_step'] for r in strategy.requests] == [0, trigger, 16+trigger]
        assert [r['action_start'] for r in strategy.requests] == [0, 16, 32]
        assert observations[:2] == [0, trigger]
        assert strategy._issued == 32
        assert transport.callback is None
    finally:
        strategy.stop_loop()


def test_async_late_chunk_waits_without_skip_and_reset_discards_pending():
    from functools import partial
    from examples.deployment.eva.eva_extension.async_rolling_strategy import StarVlaAsyncRollingStrategy
    strategy = StarVlaAsyncRollingStrategy(contract_path=str(PACKAGE/'contracts/agibot_g2_rollflow_h64_c16_async.json'))
    release = threading.Event()
    entered = threading.Event()
    calls = []
    reset_count = []
    transport = SimpleNamespace(set_action_issued_callback=lambda cb: setattr(transport, 'callback', cb),
                                reset_policy_chunk=lambda: reset_count.append('transport'))
    runtime = SimpleNamespace(transport=transport, policy=SimpleNamespace(last_chunk={}, reset=lambda: reset_count.append('policy')))
    def fetch(runtime, prompt, obs):
        start = 16*len(calls); calls.append(start)
        if start == 16:
            entered.set(); assert release.wait(2)
        return np.repeat(np.arange(start,start+16,dtype=np.float32)[:,None],24,axis=1)
    try:
        strategy.start_loop('task', partial(fetch, runtime), lambda: dict(step=strategy._issued))
        for index in range(16):
            deadline = time.monotonic()+2
            while (a := strategy.pop_next_action()) is None:
                assert time.monotonic()<deadline; time.sleep(.001)
            assert a[0] == index
            if index == 0:
                with pytest.raises(RuntimeError, match='not issued'): strategy.pop_next_action()
            transport.callback()
        assert entered.wait(1)
        assert all(strategy.pop_next_action() is None for _ in range(10))
        assert strategy._issued == 16 and calls == [0,16]
        release.set()
        strategy.reset()
        assert not strategy.is_loop_running()
        assert strategy._active is None and reset_count == ['transport','policy']
        assert strategy._current_chunk_end_step() == 0
    finally:
        release.set(); strategy.stop_loop()


def test_async_worker_error_cannot_fall_back_to_sync():
    from functools import partial
    from policy_client.base import PolicyRequestError
    from examples.deployment.eva.eva_extension.async_rolling_strategy import StarVlaAsyncRollingStrategy
    strategy = StarVlaAsyncRollingStrategy(contract_path=str(PACKAGE/'contracts/agibot_g2_rollflow_h64_c16_async.json'))
    transport = SimpleNamespace(set_action_issued_callback=lambda cb: None)
    runtime = SimpleNamespace(transport=transport, policy=SimpleNamespace(last_chunk={}))
    def fetch(runtime, prompt, obs): raise ValueError('bad network chunk')
    try:
        strategy.start_loop('task', partial(fetch,runtime), lambda: {})
        deadline=time.monotonic()+2
        while strategy._error is None:
            assert time.monotonic()<deadline; time.sleep(.001)
        assert strategy.is_loop_running()
        with pytest.raises(PolicyRequestError, match='reset/reconnect'): strategy.pop_next_action()
    finally: strategy.stop_loop()


@pytest.mark.parametrize('trigger,aggregation', [(0,'none'),(16,'none'),(True,'none'),(8,'act')])
def test_async_contract_rejects_invalid_schedule(trigger, aggregation):
    contract=load_contract(PACKAGE/'contracts/agibot_g2_rollflow_h64_c16_async.json')
    contract['execution'].update(prefetch_after_steps=trigger,aggregation=aggregation)
    with pytest.raises(ValueError): validate_contract(contract)


def test_default_g2_async_and_training_pose():
    config=load_config(PACKAGE/'configs/agibot_g2_live.py')
    assert config.inference_strategies.rolling.type=='StarVlaAsyncRollingStrategy'
    contract=load_contract(config.policy.backend_options.contract_path)
    assert contract['execution']['prefetch_after_steps']==8
    np.testing.assert_array_equal(config.robot.initial_qpos,contract['eva']['initial_qpos'])
    assert len(config.robot.initial_qpos)==24 and config.robot.initial_qpos[3]<-2


def test_async_native_eva_websocket_zmq_gdk(loopback, tmp_path):
    from core.registry import ROBOT_REGISTRY, TRANSPORT_REGISTRY, STRATEGY_REGISTRY
    from core.types import Observation
    from core.app.state import RuntimeState, SessionState, SessionMode, SessionStatus
    from core.app.handlers.control import publish_next_action
    from core.app.handlers.io import start_inference_loop, connect_policy
    from core.app.handlers.utils import reset_run_state
    from examples.deployment.eva.simulation.g2_wire import G2WireMock
    config = load_config(PACKAGE/'configs/agibot_g2_live.py')
    loopback.contract = load_contract(config.policy.backend_options.contract_path)
    loopback.mode = 'ramp'
    config.policy.host,config.policy.port = '127.0.0.1',loopback.port
    robot=ROBOT_REGISTRY.build('agibot_g2')
    node=G2WireMock(ROOT/'EVA-CLIENT',tmp_path)
    config.transport.sub_endpoint,config.transport.pub_endpoint=node.obs_endpoint,node.action_endpoint
    transport=TRANSPORT_REGISTRY.build(config.transport.type,config,robot)
    spec=config.inference_strategies.rolling
    strategy=STRATEGY_REGISTRY.build(spec.type,**spec.args)
    runtime=RuntimeState(robot=robot,transport=transport,infer_strategy=strategy)
    session=SessionState(mode=SessionMode.REAL,status=SessionStatus.RUNNING,selected_task='place carton',is_setup_done=True)
    frame=Observation(images={c['eva_key']:np.zeros((30,40,3),np.uint8) for c in loopback.contract['cameras']},state_qpos=np.zeros(24,np.float32),timestamp=1)
    try:
        node.wait_frame(transport,frame)
        assert connect_policy(config,runtime,session)
        start_inference_loop(config,runtime,session)
        deadline=time.monotonic()+5
        while session.step_index<32:
            assert time.monotonic()<deadline
            node.wait_frame(transport,frame)
            if not publish_next_action(config,runtime,session): continue
            command=node.consume_command()
            np.testing.assert_array_equal(command,transport._pending_chunk[transport._issued_in_chunk-1])
        strategy.stop_loop()
        assert len(node.received)==len(node.servo_requests)==session.step_index==strategy._issued==32
        assert [r['observation_step'] for r in strategy.requests]==[0,8,24]
        assert [r['action_start'] for r in strategy.requests]==[0,16,32]
        reset_run_state(config,runtime,session)
        assert transport._pending_chunk is None and runtime.policy._sequence==0
        assert not strategy.is_loop_running()
    finally:
        strategy.stop_loop()
        if runtime.policy: runtime.policy.close()
        transport.close();node.close()


def test_async_reconnect_invalidates_active_stream(loopback, tmp_path):
    from core.registry import ROBOT_REGISTRY, TRANSPORT_REGISTRY, STRATEGY_REGISTRY
    from core.types import Observation
    from core.app.state import RuntimeState, SessionState, SessionMode
    from core.app.handlers.io import start_inference_loop,connect_policy
    from core.app.handlers.control import publish_next_action
    from core.app.handlers.utils import reset_run_state
    from policy_client.base import PolicyRequestError
    from examples.deployment.eva.simulation.g2_wire import G2WireMock
    config=load_config(PACKAGE/'configs/agibot_g2_live.py')
    loopback.contract=load_contract(config.policy.backend_options.contract_path)
    loopback.mode='timeout'
    config.policy.host,config.policy.port='127.0.0.1',loopback.port
    robot=ROBOT_REGISTRY.build('agibot_g2')
    node=G2WireMock(ROOT/'EVA-CLIENT',tmp_path)
    config.transport.sub_endpoint,config.transport.pub_endpoint=node.obs_endpoint,node.action_endpoint
    transport=TRANSPORT_REGISTRY.build(config.transport.type,config,robot)
    spec=config.inference_strategies.rolling
    strategy=STRATEGY_REGISTRY.build(spec.type,**spec.args)
    runtime=RuntimeState(robot=robot,transport=transport,infer_strategy=strategy)
    session=SessionState(mode=SessionMode.REAL,selected_task='place carton')
    frame=Observation(images={c['eva_key']:np.zeros((30,40,3),np.uint8) for c in loopback.contract['cameras']},state_qpos=np.zeros(24,np.float32),timestamp=1)
    try:
        node.wait_frame(transport,frame)
        assert connect_policy(config,runtime,session)
        start_inference_loop(config,runtime,session)
        deadline=time.monotonic()+2
        while loopback.calls==0:
            assert time.monotonic()<deadline; time.sleep(.001)
        old=runtime.policy
        assert connect_policy(config,runtime,session,force_reconnect=True)
        assert old._closed
        with pytest.raises(PolicyRequestError, match='reset/reconnect'):
            publish_next_action(config,runtime,session)
        assert session.step_index==0 and not node.received
        reset_run_state(config,runtime,session)
        assert not strategy.is_loop_running() and runtime.policy._sequence==0
    finally:
        strategy.stop_loop()
        if runtime.policy:runtime.policy.close()
        transport.close();node.close()


def test_cold_start_has_separate_limit_and_warm_timeout_closes(loopback):
    from policy_client.base import PolicyRequestError
    from examples.deployment.eva.eva_extension.starvla_policy import StarVlaEvaPolicyClient
    policy=StarVlaEvaPolicyClient('127.0.0.1',loopback.port,str(G2),timeout_s=2,
        max_inference_latency_s=.03,max_cold_start_latency_s=.5)
    loopback.mode='timeout'
    try:
        assert policy.infer(policy_observation())['actions'].shape==(16,24)
        assert policy._sequence==1
        with pytest.raises(PolicyRequestError,match='latency limit'):
            policy.infer(policy_observation())
        assert policy._closed and policy._sequence==1
    finally:policy.close()
