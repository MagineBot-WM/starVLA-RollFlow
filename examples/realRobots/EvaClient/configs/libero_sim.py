"""EVA running a LIBERO suite through the additive local-simulation transport."""

import os
from pathlib import Path

_base_ = ["../../../../EVA-CLIENT/configs/00_base/defaults.py"]
_package_root = Path("{{ fileDirname }}").resolve().parent

console = dict(initial_tab="debug")
robot = dict(
    type="eva_libero_virtual",
    gripper_threshold=0.5,
    gripper_open=-1.0,
    gripper_close=1.0,
)
transport = dict(
    type="libero_sim",
    image_height=224,
    image_width=224,
    resize_pad=False,
    image_layout="hwc",
    convert_bgr_to_rgb=False,
    libero=dict(
        suite="libero_goal",
        task_id=0,
        init_state_id=0,
        render_size=256,
        warmup_steps=10,
        max_steps=0,
        seed=7,
    ),
)
policy = dict(
    type="starvla_eva",
    host=os.environ.get("STARVLA_HOST", "127.0.0.1"),
    port=int(os.environ.get("STARVLA_PORT", "10093")),
    backend_options=dict(
        contract_path=str(_package_root / "contracts" / "libero_rollflow_h64_c8.json"),
    ),
)
inference_cfg = dict(
    obs_space=dict(type="JointState"),
    action_space=dict(type="JointState"),
    publish_rate=20,
    debug_tasks=["open the middle drawer of the cabinet"],
)
work_dir = "work_dirs/eva_libero_sim"
inference_strategies = dict(
    _delete_=True,
    rolling=dict(type="StarVlaRollingStrategy", args=dict(
        contract_path=str(_package_root / "contracts" / "libero_rollflow_h64_c8.json"))),
)
