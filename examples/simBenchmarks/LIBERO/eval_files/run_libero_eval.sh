#!/usr/bin/env bash
set -euo pipefail

# Four-suite LIBERO evaluation in one tmux session.
#
# GPU / port mapping:
#   GPU 0, port 6694 -> libero_spatial
#   GPU 1, port 6695 -> libero_object
#   GPU 2, port 6696 -> libero_goal
#   GPU 3, port 6697 -> libero_10 (Long)
#
# Each suite gets its own policy server because RollFlow keeps inference state.
# Sharing one server between concurrent environments would mix rolling caches.
#
# Usage:
#   bash run_libero_eval.sh          Start all four suites and attach tmux.
#   bash run_libero_eval.sh status   List windows and completed videos.
#   bash run_libero_eval.sh attach   Reattach after leaving the terminal.
#   bash run_libero_eval.sh stop     Stop all eight processes.
#
# Inside tmux:
#   Ctrl-b, then 0..7  Switch window
#   Ctrl-b, then d     Detach without stopping evaluation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_DIR="${STARVLA_DIR:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"

# Usually this is the only line to change.
CKPT="${CKPT:-/data/tzq/starVLA_checkpoints/libero_qwengroot_rollflow_h32_c8_b128_unfrozen_463a1f7/checkpoints/steps_10000_pytorch_model.pt}"

STARVLA_PYTHON="${STARVLA_PYTHON:-/data/miniconda3/envs/starVLA/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/data/miniconda3/envs/lerobot/bin/python}"
LIBERO_HOME="${LIBERO_HOME:-/home/taizun/tzq/LIBERO}"
SESSION="${SESSION:-rollflow_libero_4suite}"
BASE_PORT="${BASE_PORT:-6694}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
MAX_TASKS="${MAX_TASKS:--1}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
SEED="${SEED:-7}"
UNNORM_KEY="${UNNORM_KEY:-franka}"
# Set to 0 to skip replay-frame buffering and MP4 encoding. The simulator must
# still render camera observations because they are policy inputs.
SAVE_VIDEO="${SAVE_VIDEO:-1}"
ATTACH="${ATTACH:-1}"
# Optional server config overrides, passed as separate KEY=VALUE arguments:
# bash run_libero_eval.sh start framework.action_model.execution_horizon=4 framework.action_model.inference_steps=8
# For comparisons, also set distinct SESSION, BASE_PORT and RESULTS_ROOT.
SERVER_OVERRIDES=()
for override in "${@:2}"; do
  [[ "${override}" == *=* ]] || { echo "Expected KEY=VALUE: ${override}" >&2; exit 2; }
  SERVER_OVERRIDES+=(--config_override "${override}")
done
SERVER_EXTRA_ARGS=""
if (( ${#SERVER_OVERRIDES[@]} )); then
  printf -v SERVER_EXTRA_ARGS ' %q' "${SERVER_OVERRIDES[@]}"
fi

SUITES=(libero_spatial libero_object libero_goal libero_10)
LABELS=(spatial object goal long)
GPUS=(0 1 2 3)
ACTION="${1:-start}"

usage() {
  sed -n '14,18p' "${BASH_SOURCE[0]}"
}

case "${ACTION}" in
  status)
    tmux has-session -t "${SESSION}" 2>/dev/null || {
      echo "tmux session is not running: ${SESSION}"
      exit 1
    }
    tmux list-windows -t "${SESSION}"
    echo
    for suite in "${SUITES[@]}"; do
      result_dir="${RESULTS_ROOT:-${CKPT%%/checkpoints/*}/results}/${suite}/$(basename "${CKPT}")"
      count=0
      [[ ! -d "${result_dir}" ]] || count="$(find "${result_dir}" -maxdepth 1 -name 'rollout_*.mp4' | wc -l)"
      printf '%-16s %s completed videos\n' "${suite}" "${count}"
    done
    exit
    ;;
  attach)
    tmux has-session -t "${SESSION}" 2>/dev/null || {
      echo "tmux session is not running: ${SESSION}"
      exit 1
    }
    exec tmux attach -t "${SESSION}"
    ;;
  stop)
    tmux has-session -t "${SESSION}" 2>/dev/null || {
      echo "tmux session is not running: ${SESSION}"
      exit 0
    }
    tmux kill-session -t "${SESSION}"
    echo "Stopped tmux session: ${SESSION}"
    exit
    ;;
  -h|--help)
    usage
    exit
    ;;
  start) ;;
  *) usage; exit 2 ;;
esac

[[ -f "${CKPT}" ]] || { echo "Checkpoint not found: ${CKPT}" >&2; exit 1; }
[[ -x "${STARVLA_PYTHON}" ]] || { echo "Invalid STARVLA_PYTHON: ${STARVLA_PYTHON}" >&2; exit 1; }
[[ -x "${LIBERO_PYTHON}" ]] || { echo "Invalid LIBERO_PYTHON: ${LIBERO_PYTHON}" >&2; exit 1; }
[[ "${SAVE_VIDEO}" == "0" || "${SAVE_VIDEO}" == "1" ]] || {
  echo "SAVE_VIDEO must be 0 or 1; got: ${SAVE_VIDEO}" >&2
  exit 2
}
[[ -f "${LIBERO_HOME}/libero/libero/__init__.py" ]] || {
  echo "Invalid LIBERO_HOME: ${LIBERO_HOME}" >&2
  exit 1
}
tmux has-session -t "${SESSION}" 2>/dev/null && {
  echo "Session already exists: ${SESSION}. Use '$0 attach' or '$0 stop'." >&2
  exit 1
}

# Refuse partial startup. An occupied port usually means another policy server
# is active; launching another evaluator against it would corrupt RollFlow state.
for i in "${!SUITES[@]}"; do
  port=$((BASE_PORT + i))
  if [[ -n "$(ss -H -ltn "sport = :${port}" 2>/dev/null)" ]]; then
    echo "Port ${port} is already in use; nothing was started." >&2
    exit 1
  fi
done

MODEL_ROOT="${CKPT%%/checkpoints/*}"
CKPT_NAME="$(basename "${CKPT}")"
RESULTS_ROOT="${RESULTS_ROOT:-${MODEL_ROOT}/results}"
SERVER_LOG_DIR="${RESULTS_ROOT}/servers/${CKPT_NAME}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${XDG_CACHE_HOME:-${HOME}/.cache}/starvla/libero}"
mkdir -p "${SERVER_LOG_DIR}" "${LIBERO_CONFIG_PATH}"

# Use a private config because ~/.libero/config.yaml on this machine points to
# an old checkout. The files below are trusted local LIBERO benchmark assets.
"${LIBERO_PYTHON}" - "${LIBERO_HOME}" "${LIBERO_CONFIG_PATH}/config.yaml" <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
benchmark = root / "libero" / "libero"
paths = {
    "assets": benchmark / "assets",
    "bddl_files": benchmark / "bddl_files",
    "benchmark_root": benchmark,
    "datasets": root / "datasets",
    "init_states": benchmark / "init_files",
}
pathlib.Path(sys.argv[2]).write_text(
    "".join(f"{key}: {value}\n" for key, value in paths.items()),
    encoding="utf-8",
)
PY

keep_open() {
  local name="$1"
  local command="$2"
  printf '%s; code=$?; echo "[%s exited with code $code]"; exec bash' "${command}" "${name}"
}

if [[ "${SAVE_VIDEO}" == "1" ]]; then
  VIDEO_FLAG="--args.save-video"
else
  VIDEO_FLAG="--args.no-save-video"
fi

for i in "${!SUITES[@]}"; do
  suite="${SUITES[$i]}"
  label="${LABELS[$i]}"
  gpu="${GPUS[$i]}"
  port=$((BASE_PORT + i))
  output_dir="${RESULTS_ROOT}/${suite}/${CKPT_NAME}"
  mkdir -p "${output_dir}"

  printf -v server_run \
    'cd %q && DEBUG= NO_ALBUMENTATIONS_UPDATE=1 PYTHONPATH=%q CUDA_VISIBLE_DEVICES=%q %q %q --ckpt_path %q --port %q --use_bf16%s 2>&1 | tee %q' \
    "${STARVLA_DIR}" "${STARVLA_DIR}" "${gpu}" "${STARVLA_PYTHON}" \
    "${STARVLA_DIR}/deployment/model_server/server_policy.py" "${CKPT}" "${port}" \
    "${SERVER_EXTRA_ARGS}" "${SERVER_LOG_DIR}/${suite}.log"
  printf -v eval_run \
    'cd %q && DEBUG= LIBERO_CONFIG_PATH=%q PYTHONPATH=%q MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=%q %q %q --args.pretrained-path %q --args.host 127.0.0.1 --args.port %q --args.task-suite-name %q --args.num-trials-per-task %q --args.max-tasks %q --args.num-steps-wait %q --args.seed %q --args.video-out-path %q --args.unnorm-key %q %q 2>&1 | tee %q' \
    "${STARVLA_DIR}" "${LIBERO_CONFIG_PATH}" "${LIBERO_HOME}:${STARVLA_DIR}" "${gpu}" \
    "${LIBERO_PYTHON}" "${STARVLA_DIR}/examples/simBenchmarks/LIBERO/eval_files/eval_libero.py" \
    "${CKPT}" "${port}" "${suite}" "${NUM_TRIALS_PER_TASK}" "${MAX_TASKS}" \
    "${NUM_STEPS_WAIT}" "${SEED}" "${output_dir}" \
    "${UNNORM_KEY}" "${VIDEO_FLAG}" "${output_dir}/eval.log"

  server_cmd="$(keep_open "server-${label}" "set -o pipefail; ${server_run}")"
  eval_cmd="$(keep_open "eval-${label}" "set -o pipefail; ${eval_run}")"
  if (( i == 0 )); then
    tmux new-session -d -s "${SESSION}" -n "server-${label}" "${server_cmd}"
  else
    tmux new-window -d -t "${SESSION}" -n "server-${label}" "${server_cmd}"
  fi
  tmux new-window -d -t "${SESSION}" -n "eval-${label}" "${eval_cmd}"
done

tmux select-window -t "${SESSION}:eval-spatial"
echo "Started ${SESSION}: four servers + four evaluators"
echo "  checkpoint : ${CKPT}"
echo "  save video : ${SAVE_VIDEO}"
echo "  windows    : Ctrl-b then 0..7"
echo "  detach     : Ctrl-b then d"
echo "  status     : $0 status"
echo "  reattach   : $0 attach"
echo "  stop all   : $0 stop"

[[ "${ATTACH}" == "0" ]] || exec tmux attach -t "${SESSION}"
