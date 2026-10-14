"""EVA bootstrap and binding; model dependencies are never imported here."""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any

from .bridge_contract import execution_spec, load_contract

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EVA_ROOT = ROOT / "EVA-CLIENT"


def bootstrap(eva_root: Path = DEFAULT_EVA_ROOT) -> None:
    eva_root = eva_root.expanduser().resolve()
    if not (eva_root / "src/core/config.py").is_file():
        raise ValueError(f"EVA checkout missing src/core/config.py: {eva_root}")
    imported = sys.modules.get("core.config")
    if imported is not None and not Path(imported.__file__).resolve().is_relative_to(eva_root):
        raise RuntimeError("A different EVA checkout is already imported; use a fresh process for --eva-root")
    for path in (ROOT, eva_root, eva_root / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ["STARVLA_EVA_ROOT"] = str(eva_root)
    import policy_client  # noqa: F401
    import robots  # noqa: F401
    import transport  # noqa: F401
    from . import eva_extension  # noqa: F401
    from .adapters.eva import check_api
    check_api()


def bind_config(config: Any) -> dict:
    """Derive preprocessing/execution settings from one shared contract."""
    contract = load_contract(config.policy.backend_options.contract_path)
    binding = contract.get("eva", {})
    if binding and config.robot.type != binding["robot_type"]:
        raise ValueError(f"EVA robot {config.robot.type!r} != contract {binding['robot_type']!r}")
    disabled = set(config.transport.disabled_cameras)
    if disabled & {camera["eva_key"] for camera in contract["cameras"]}:
        raise ValueError("A policy camera is disabled in EVA")
    if config.transport.disabled_groups:
        raise ValueError("Disabled actuator groups would substitute initial state into policy inputs")
    height, width = contract["image"]["hw"]
    config.transport.image_height = height
    config.transport.image_width = width
    config.transport.resize_pad = False
    config.transport.image_layout = "hwc"
    if binding:
        config.transport.convert_bgr_to_rgb = binding["source_color"] == "bgr"
        if "action_socket_role" in binding:
            config.transport.action_socket_role = binding["action_socket_role"]
        if "initial_qpos" in binding:
            config.robot.initial_qpos = list(binding["initial_qpos"])
        if type(config.inference_cfg.obs_space).__name__ != binding.get("obs_space", "JointState"):
            raise ValueError("EVA observation space differs from the contract")
        if type(config.inference_cfg.action_space).__name__ != binding.get("action_space", "JointState"):
            raise ValueError("EVA action space differs from the contract")
        gripper = binding.get("gripper")
        if gripper:
            config.robot.gripper_threshold = float(gripper["threshold"])
            config.robot.gripper_open = float(gripper["open"])
            config.robot.gripper_close = float(gripper["close"])
    execution = execution_spec(contract)
    if "control_hz" in execution:
        config.inference_cfg.publish_rate = execution["control_hz"]
    # Each rolling request advances one complete chunk, including in prefetch mode.
    from core.config import ConfigDict
    config.inference_strategies = ConfigDict(rolling=dict(
        type=("StarVlaAsyncRollingStrategy" if execution.get("scheduling") == "async_prefetch"
              else "StarVlaRollingStrategy" if execution["kind"] == "rolling" else "StarVlaChunkStrategy"),
        args=dict(contract_path=config.policy.backend_options.contract_path),
    ))
    return contract


def validate_robot(robot: Any, contract: dict) -> None:
    cameras = {camera.observation_key for camera in robot.observation_schema.cameras}
    missing = {camera["eva_key"] for camera in contract["cameras"]} - cameras
    if missing:
        raise ValueError(f"EVA robot lacks policy cameras: {sorted(missing)}")
    if contract["action"]["mode"] != "eef" and robot.total_action_dim != contract["action"]["dim"]:
        raise ValueError("EVA robot DOF and policy action dimension differ")
    names = contract.get("eva", {}).get("joint_names")
    if names is not None:
        actual = [name for group in robot.actuator_groups for name in group.joint_names]
        if actual != names:
            raise ValueError("EVA joint order differs from eva.joint_names in the contract")
    groups = contract.get("eva", {}).get("state_groups")
    if groups:
        if list(robot.observation_schema.state_composition) != [group["name"] for group in groups]:
            raise ValueError("EVA state group order differs from the contract")
        dims = {group.name: group.dof for group in robot.actuator_groups}
        for group in groups:
            start, stop = group["slice"]
            if dims.get(group["name"]) != stop - start:
                raise ValueError(f"EVA state group dimension differs for {group['name']}")


def load_config(path: Path, eva_root: Path = DEFAULT_EVA_ROOT):
    bootstrap(eva_root)
    from core.config import load_config as eva_load_config
    from core.registry import ROBOT_REGISTRY
    config = eva_load_config(path.expanduser().resolve())
    contract = bind_config(config)
    robot = ROBOT_REGISTRY.build(config.robot.type)
    validate_robot(robot, contract)
    from .adapters.hardware import inspect_hardware_joint_order
    inspect_hardware_joint_order(eva_root, contract)
    return config
