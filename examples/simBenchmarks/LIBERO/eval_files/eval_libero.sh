#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=eval_config.sh
source "${SCRIPT_DIR}/eval_config.sh"
STARVLA_DIR="${STARVLA_DIR:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"

[[ -x "${LIBERO_PYTHON}" ]] || { echo "Invalid LIBERO_PYTHON: ${LIBERO_PYTHON}" >&2; exit 1; }
[[ -f "${LIBERO_HOME}/libero/libero/__init__.py" ]] || {
  echo "Invalid LIBERO_HOME: ${LIBERO_HOME}" >&2
  exit 1
}

[[ -n "${CKPT}" && -f "${CKPT}" ]] || {
  echo "Checkpoint not found: ${CKPT}" >&2
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
CMD+=(--args.unnorm-key "${UNNORM_KEY}")

cd "${STARVLA_DIR}"
CUDA_VISIBLE_DEVICES="${EVAL_GPU_ID}" DEBUG="${LIBERO_DEBUG:-}" \
  "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
