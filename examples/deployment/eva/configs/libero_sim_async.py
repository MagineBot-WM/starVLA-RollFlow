"""LIBERO H32/C8: prefetch at step 4 with the verified 55K model."""
from pathlib import Path
_base_ = ["./libero_sim_verified.py"]
_package_root = Path("{{ fileDirname }}").resolve().parent
policy = dict(backend_options=dict(
    contract_path=str(_package_root / "contracts/libero_rollflow_h32_c8_k4_async.json"), timeout_s=5.0))
