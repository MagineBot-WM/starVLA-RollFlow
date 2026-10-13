"""Live G2 deployment via the additive StarVLA contract client."""

import os
from pathlib import Path

_base_ = ["../../../../EVA-CLIENT/configs/01_deploy/agibot_g2/_base.py"]
_package_root = Path("{{ fileDirname }}").resolve().parent

policy = dict(
    type="starvla_eva",
    host=os.environ.get("STARVLA_HOST", "127.0.0.1"),
    port=int(os.environ.get("STARVLA_PORT", "10094")),
    backend_options=dict(
        contract_path=str(_package_root / "contracts" / "agibot_g2_rollflow_h64_c16.json"),
    ),
)
transport = dict(
    type="starvla_zmq",
    sub_endpoint=os.environ.get("G2_OBS_ENDPOINT", "tcp://127.0.0.1:5555"),
    pub_endpoint=os.environ.get("G2_ACTION_ENDPOINT", "tcp://127.0.0.1:5556"),
    image_height=224,
    image_width=224,
    resize_pad=False,
    image_layout="hwc",
    convert_bgr_to_rgb=True,
)
inference_cfg = dict(
    obs_space=dict(type="JointState"),
    action_space=dict(type="JointState"),
    publish_rate=30,
    debug_tasks=["Pick up the white carton and place it inside the large tan carton on the left."],
)
inference_strategies = dict(
    _delete_=True,
    rolling=dict(type="StarVlaRollingStrategy", args=dict(
        contract_path=str(_package_root / "contracts" / "agibot_g2_rollflow_h64_c16.json"))),
)
