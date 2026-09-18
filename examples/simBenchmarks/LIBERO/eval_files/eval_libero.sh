#!/usr/bin/env bash
# Run the LIBERO simulator against an already running StarVLA policy server.
#
# Start the server in the StarVLA environment first:
#   CKPT=/path/to/checkpoints/steps_5000_pytorch_model.pt \
#   bash examples/simBenchmarks/LIBERO/eval_files/run_policy_server.sh
#
# Then run this script from the LIBERO environment.  The checkpoint is used to
# identify the output directory and to keep the evaluation command explicit;
# the server performs model loading and action un-normalization.

set -euo pipefail

# ------------------------------ User configuration -------------------------
STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
LIBERO_HOME="${LIBERO_HOME:-}"
LIBERO_PYTHON="${LIBERO_PYTHON:-python}"
CKPT="${CKPT:-${STARVLA_DIR}/playground/Checkpoints/libero_example/checkpoints/steps_50000_pytorch_model.pt}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-6694}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_goal}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
MAX_TASKS="${MAX_TASKS:--1}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
SEED="${SEED:-7}"
UNNORM_KEY="${UNNORM_KEY:-}"
MUJOCO_GL_VALUE="${MUJOCO_GL_VALUE:-egl}"
PYOPENGL_PLATFORM_VALUE="${PYOPENGL_PLATFORM_VALUE:-egl}"

if [[ -z "${LIBERO_HOME}" ]]; then
  echo "LIBERO_HOME is required." >&2
  echo "Example: LIBERO_HOME=/path/to/LIBERO LIBERO_PYTHON=/path/to/python bash $0" >&2
  exit 1
fi
if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  echo "Set CKPT to a steps_*_pytorch_model.pt file." >&2
  exit 1
fi
if ! command -v "${LIBERO_PYTHON}" >/dev/null 2>&1; then
  echo "LIBERO_PYTHON was not found: ${LIBERO_PYTHON}" >&2
  exit 1
fi
if [[ ! "${PORT}" =~ ^[1-9][0-9]*$ ]] || (( PORT > 65535 )); then
  echo "PORT must be an integer in [1, 65535], got: ${PORT}" >&2
  exit 1
fi
for value_pair in \
  "NUM_TRIALS_PER_TASK:${NUM_TRIALS_PER_TASK}" \
  "MAX_TASKS:${MAX_TASKS}" \
  "NUM_STEPS_WAIT:${NUM_STEPS_WAIT}" \
  "SEED:${SEED}"; do
  name="${value_pair%%:*}"
  value="${value_pair#*:}"
  if [[ "${name}" == "MAX_TASKS" && "${value}" == "-1" ]]; then
    continue
  fi
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "${name} must be a non-negative integer, got: ${value}" >&2
    exit 1
  fi
done
case "${TASK_SUITE_NAME}" in
  libero_spatial|libero_object|libero_goal|libero_10|libero_90) ;;
  *)
    echo "Unknown TASK_SUITE_NAME: ${TASK_SUITE_NAME}" >&2
    echo "Use libero_spatial, libero_object, libero_goal, libero_10, or libero_90." >&2
    exit 1
    ;;
esac

cd "${STARVLA_DIR}"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${PYTHONPATH:-}:${LIBERO_HOME}:${STARVLA_DIR}"
export MUJOCO_GL="${MUJOCO_GL_VALUE}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM_VALUE}"

# Keep videos next to the checkpoint unless the caller explicitly overrides it.
if [[ -z "${VIDEO_OUT_PATH:-}" ]]; then
  if [[ "${CKPT}" == */checkpoints/* ]]; then
    model_root="${CKPT%%/checkpoints/*}"
  else
    model_root="$(cd "$(dirname "${CKPT}")" && pwd)"
  fi
  checkpoint_name="$(basename "${CKPT}")"
  VIDEO_OUT_PATH="${model_root}/results/${TASK_SUITE_NAME}/${checkpoint_name}"
fi
mkdir -p "${VIDEO_OUT_PATH}"

echo "LIBERO evaluation configuration:"
echo "  checkpoint: ${CKPT}"
echo "  suite:      ${TASK_SUITE_NAME}"
echo "  trials/task:${NUM_TRIALS_PER_TASK}"
echo "  max tasks:  ${MAX_TASKS}"
echo "  server:     ${HOST}:${PORT}"
echo "  videos:     ${VIDEO_OUT_PATH}"

eval_args=(
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
if [[ -n "${UNNORM_KEY}" ]]; then
  eval_args+=(--args.unnorm-key "${UNNORM_KEY}")
fi

exec "${LIBERO_PYTHON}" ./examples/simBenchmarks/LIBERO/eval_files/eval_libero.py "${eval_args[@]}"
