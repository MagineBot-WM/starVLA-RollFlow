"""LIBERO checkpoint with saved 500-trial evaluation: H32/C8, 55K steps."""

from pathlib import Path

_base_ = ["./libero_sim.py"]
_package_root = Path("{{ fileDirname }}").resolve().parent
_contract = str(_package_root / "contracts" / "libero_rollflow_h32_c8_k4.json")

policy = dict(backend_options=dict(contract_path=_contract))
inference_strategies = dict(
    _delete_=True,
    rolling=dict(type="StarVlaRollingStrategy", args=dict(contract_path=_contract)),
)
work_dir = "work_dirs/eva_libero_verified_55k"
