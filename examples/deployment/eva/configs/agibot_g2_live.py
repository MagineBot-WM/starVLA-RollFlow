"""Default G2: prefetch at issued step 8, execute the entire 16-step chunk."""
from pathlib import Path
_base_ = ["./agibot_g2_live_sync.py"]
_package_root = Path("{{ fileDirname }}").resolve().parent
policy = dict(backend_options=dict(
    contract_path=str(_package_root / "contracts/agibot_g2_rollflow_h64_c16_async.json")))
