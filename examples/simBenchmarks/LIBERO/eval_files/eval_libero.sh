#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
LIBERO_HOME="${LIBERO_HOME:-/home/taizun/tzq/LIBERO}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/data/miniconda3/envs/lerobot/bin/python}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/data/tzq/starVLA_checkpoints/libero_qwengroot_rollflow_h32_c8_b128_unfrozen_463a1f7}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-6694}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_goal}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
MAX_TASKS="${MAX_TASKS:--1}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
SEED="${SEED:-7}"
EVAL_GPU_ID="${EVAL_GPU_ID:-0}"
SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-300}"

[[ -x "${LIBERO_PYTHON}" ]] || { echo "Invalid LIBERO_PYTHON: ${LIBERO_PYTHON}" >&2; exit 1; }
[[ -f "${LIBERO_HOME}/libero/libero/__init__.py" ]] || {
  echo "Invalid LIBERO_HOME: ${LIBERO_HOME}" >&2
  exit 1
}

# Explicit CKPT wins; otherwise use the latest numeric step in CHECKPOINT_ROOT.
if [[ -z "${CKPT:-}" ]]; then
  CKPT="$({ find "${CHECKPOINT_ROOT}/checkpoints" -maxdepth 1 -type f \
    -name 'steps_*_pytorch_model.pt' -print 2>/dev/null || true; } | sort -V | tail -n 1)"
fi
[[ -n "${CKPT}" && -f "${CKPT}" ]] || {
  echo "No checkpoint found. Set CKPT or CHECKPOINT_ROOT." >&2
  exit 1
}

export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${XDG_CACHE_HOME:-${HOME}/.cache}/starvla/libero}"
export PYTHONPATH="${LIBERO_HOME}:${STARVLA_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM=false
# Trusted LIBERO init-state files contain NumPy arrays. PyTorch 2.6+ otherwise
# rejects them because torch.load now defaults to weights_only=True.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
mkdir -p "${LIBERO_CONFIG_PATH}"

# Do not reuse ~/.libero/config.yaml: it may belong to a stale checkout.
"${LIBERO_PYTHON}" - "${LIBERO_HOME}" "${LIBERO_CONFIG_PATH}/config.yaml" <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
config = pathlib.Path(sys.argv[2])
benchmark = root / "libero" / "libero"
paths = {
    "assets": benchmark / "assets",
    "bddl_files": benchmark / "bddl_files",
    "benchmark_root": benchmark,
    "datasets": root / "datasets",
    "init_states": benchmark / "init_files",
}
config.write_text("".join(f"{key}: {value}\n" for key, value in paths.items()), encoding="utf-8")
PY

"${LIBERO_PYTHON}" - <<'PY'
import imageio, matplotlib, tyro, websockets
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

assert get_libero_path("bddl_files")
assert benchmark.get_benchmark_dict()
PY

CKPT_NAME="$(basename "${CKPT}")"
MODEL_ROOT="${CKPT%%/checkpoints/*}"
VIDEO_OUT_PATH="${VIDEO_OUT_PATH:-${MODEL_ROOT}/results/${TASK_SUITE_NAME}/${CKPT_NAME}}"
LOG_FILE="${LOG_FILE:-${VIDEO_OUT_PATH}/eval.log}"
mkdir -p "${VIDEO_OUT_PATH}"

echo "LIBERO evaluation:"
printf '  %-11s %s\n' \
  checkpoint "${CKPT}" \
  python "${LIBERO_PYTHON}" \
  suite "${TASK_SUITE_NAME}" \
  tasks "${MAX_TASKS} (-1 = all)" \
  trials "${NUM_TRIALS_PER_TASK} per task" \
  output "${VIDEO_OUT_PATH}"

if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
  echo "Preflight checks passed."
  exit 0
fi

echo "Waiting up to ${SERVER_WAIT_SECONDS}s for ws://${HOST}:${PORT} ..."
"${LIBERO_PYTHON}" - "${HOST}" "${PORT}" "${SERVER_WAIT_SECONDS}" <<'PY'
import sys
import time
from websockets.sync.client import connect

host, port, timeout = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
deadline = time.monotonic() + timeout
while True:
    try:
        with connect(f"ws://{host}:{port}", open_timeout=2) as websocket:
            websocket.recv()
            break
    except (OSError, TimeoutError):
        if time.monotonic() >= deadline:
            raise SystemExit(f"Policy server unavailable at ws://{host}:{port}")
        time.sleep(2)
PY

CMD=(
  "${LIBERO_PYTHON}" "${STARVLA_DIR}/examples/simBenchmarks/LIBERO/eval_files/eval_libero.py"
  --args.pretrained-path "${CKPT}"
  --args.host "${HOST}"
  --args.port "${PORT}"
  --args.task-suite-name "${TASK_SUITE_NAME}"
  --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}"
  --args.max-tasks "${MAX_TASKS}"
  --args.num-steps-wait "${NUM_STEPS_WAIT}"
  --args.seed "${SEED}"
  --args.video-out-path "${VIDEO_OUT_PATH}"
)
[[ -z "${UNNORM_KEY:-}" ]] || CMD+=(--args.unnorm-key "${UNNORM_KEY}")

cd "${STARVLA_DIR}"
CUDA_VISIBLE_DEVICES="${EVAL_GPU_ID}" DEBUG="${LIBERO_DEBUG:-}" \
  "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
