"""Live G2 deployment via the additive StarVLA contract client."""

import os
from pathlib import Path

_base_ = ["{{ $STARVLA_EVA_ROOT:../../../../EVA-CLIENT }}/configs/01_deploy/agibot_g2/_base.py"]
_package_root = Path("{{ fileDirname }}").resolve().parent

policy = dict(
    type="starvla_eva",
    host=os.environ.get("STARVLA_HOST", "127.0.0.1"),
    port=int(os.environ.get("STARVLA_PORT", "10094")),
    backend_options=dict(
        contract_path=str(_package_root / "contracts" / "agibot_g2_rollflow_h64_c16.json"),
        timeout_s=5.0,
        max_inference_latency_s=1.0,
        max_cold_start_latency_s=5.0,
    ),
)
transport = dict(
    type="starvla_zmq",
    sub_endpoint=os.environ.get("G2_OBS_ENDPOINT", "tcp://127.0.0.1:5555"),
    pub_endpoint=os.environ.get("G2_ACTION_ENDPOINT", "tcp://127.0.0.1:5556"),
    max_frame_age_s=0.5,
)
inference_cfg = dict(
    obs_space=dict(type="JointState"),
    action_space=dict(type="JointState"),
    debug_tasks=["Pick up the white carton and place it inside the large tan carton on the left."],
)
inference_strategies = dict(
    _delete_=True,
    rolling=dict(type="StarVlaRollingStrategy", args=dict(
        contract_path=str(_package_root / "contracts" / "agibot_g2_rollflow_h64_c16.json"))),
)
