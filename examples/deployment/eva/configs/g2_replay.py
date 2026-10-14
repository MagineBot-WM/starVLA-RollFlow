"""Read-only EVA replay config for the nine held-out AgiBot G2 episodes."""

import os
from pathlib import Path

_base_ = ["{{ $STARVLA_EVA_ROOT:../../../../EVA-CLIENT }}/configs/01_deploy/agibot_g2/_base.py"]
_package_root = Path("{{ fileDirname }}").resolve().parent

robot = dict(type="agibot_g2")
transport = dict(
    type="starvla_dataset",
    dataset_dir=os.environ.get(
        "STARVLA_G2_DATASET",
        str(_package_root.parents[2] / "datasets" / "AgiBotG2" / "white_carton_to_left_tan_carton_20261009"),
    ),
    episode_id=0,
    dataset_keys=dict(
        state_key="observation.state",
        action_key="action",
        video_keys={},
    ),
)
policy = dict(
    type="starvla_eva",
    host=os.environ.get("STARVLA_HOST", "127.0.0.1"),
    port=int(os.environ.get("STARVLA_PORT", "10094")),
    backend_options=dict(
        contract_path=str(_package_root / "contracts" / "agibot_g2_rollflow_h64_c16.json"),
    ),
)
inference_cfg = dict(
    obs_space=dict(type="JointState"),
    action_space=dict(type="JointState"),
)
work_dir = "work_dirs/eva_g2_dataset_replay"
inference_strategies = dict(
    _delete_=True,
    rolling=dict(type="StarVlaRollingStrategy", args=dict(
        contract_path=str(_package_root / "contracts" / "agibot_g2_rollflow_h64_c16.json"))),
)
