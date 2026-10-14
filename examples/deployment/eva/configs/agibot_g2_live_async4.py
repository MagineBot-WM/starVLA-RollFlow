"""G2: earlier prefetch at step 4 gives 12 steps of inference runway."""
from pathlib import Path
_base_ = ["./agibot_g2_live.py"]
_package_root = Path("{{ fileDirname }}").resolve().parent
policy = dict(backend_options=dict(
    contract_path=str(_package_root / "contracts/agibot_g2_rollflow_h64_c16_async4.json")))
