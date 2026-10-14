#!/usr/bin/env bash
# Install into a separate Python 3.11 environment; existing training envs are untouched.
set -euo pipefail
eva_bridge_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
eva_eval_env="${1:?Usage: bash setup_eval_environment.sh ENV_DIR LIBERO_CHECKOUT}"
eva_libero_checkout="${2:?Pass a standard LIBERO checkout with assets and init_files}"
uv venv --python 3.11 "$eva_eval_env"
uv pip install --python "$eva_eval_env/bin/python" -r "$eva_bridge_dir/requirements-eval.txt"
uv pip install --python "$eva_eval_env/bin/python" torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$eva_eval_env/bin/python" --no-deps -e "$eva_libero_checkout"
mkdir -p "$eva_eval_env/libero-config"
"$eva_eval_env/bin/python" - "$eva_libero_checkout" "$eva_eval_env/libero-config" <<'PY'
from pathlib import Path
import sys
import yaml
root = Path(sys.argv[1]).resolve() / "libero" / "libero"
paths = {"benchmark_root": str(root), "assets": str(root / "assets"),
         "bddl_files": str(root / "bddl_files"), "init_states": str(root / "init_files"),
         "datasets": str(root.parent / "datasets")}
Path(sys.argv[2], "config.yaml").write_text(yaml.safe_dump(paths))
PY
printf 'Environment ready: %s\n' "$eva_eval_env"
printf 'For simulator runs set PYTHONPATH=%s LIBERO_CONFIG_PATH=%s/libero-config MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1\n' "$eva_libero_checkout" "$eva_eval_env"
