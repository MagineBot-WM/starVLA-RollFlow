#!/usr/bin/env bash
set -euo pipefail

# One-command LIBERO evaluation manager.
#
#   bash run_libero_eval.sh          Start server + evaluator, then attach.
#   bash run_libero_eval.sh status   Show both tmux windows.
#   bash run_libero_eval.sh attach   Reattach after closing the terminal.
#   bash run_libero_eval.sh stop     Stop both processes and delete the session.
#
# Inside tmux, press Ctrl-b then 0/1 to switch server/eval windows.
# Detach without stopping anything: Ctrl-b then d. Reattach with `attach`.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=eval_config.sh
source "${SCRIPT_DIR}/eval_config.sh"

SESSION="${SESSION:-rollflow_libero_eval}"
ACTION="${1:-start}"
VIDEO_OUT_PATH="${VIDEO_OUT_PATH:-}"
LOG_FILE="${LOG_FILE:-}"

usage() {
  sed -n '3,10p' "${BASH_SOURCE[0]}"
}

case "${ACTION}" in
  status)
    tmux has-session -t "${SESSION}" 2>/dev/null || {
      echo "tmux session is not running: ${SESSION}"
      exit 1
    }
    tmux list-windows -t "${SESSION}"
    exit
    ;;
  attach)
    exec tmux attach -t "${SESSION}"
    ;;
  stop)
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

tmux has-session -t "${SESSION}" 2>/dev/null && {
  echo "Session already exists: ${SESSION}"
  echo "Use '$0 attach' or '$0 stop'."
  exit 1
}
if [[ -n "$(ss -H -ltn "sport = :${PORT}" 2>/dev/null)" ]]; then
  echo "Port ${PORT} is already in use; refusing to start a second policy server." >&2
  echo "This also protects RollFlow cache state from concurrent evaluators." >&2
  exit 1
fi

[[ -n "${CKPT}" && -f "${CKPT}" ]] || {
  echo "Checkpoint not found: ${CKPT}" >&2
  exit 1
}

MODEL_ROOT="${CKPT%%/checkpoints/*}"
SERVER_LOG="${SERVER_LOG:-${MODEL_ROOT}/results/server_$(basename "${CKPT}").log}"
mkdir -p "$(dirname "${SERVER_LOG}")"

printf -v server_cmd \
  'cd %q && CKPT=%q PORT=%q GPU_ID=%q USE_BF16=%q STARVLA_PYTHON=%q bash %q 2>&1 | tee %q' \
  "${SCRIPT_DIR}" "${CKPT}" "${PORT}" "${GPU_ID}" "${USE_BF16}" "${STARVLA_PYTHON}" \
  "${SCRIPT_DIR}/run_policy_server.sh" "${SERVER_LOG}"
printf -v eval_cmd \
  'cd %q && CKPT=%q HOST=%q PORT=%q EVAL_GPU_ID=%q LIBERO_PYTHON=%q LIBERO_HOME=%q TASK_SUITE_NAME=%q NUM_TRIALS_PER_TASK=%q MAX_TASKS=%q NUM_STEPS_WAIT=%q SEED=%q UNNORM_KEY=%q VIDEO_OUT_PATH=%q LOG_FILE=%q bash %q' \
  "${SCRIPT_DIR}" "${CKPT}" "${HOST}" "${PORT}" "${EVAL_GPU_ID}" \
  "${LIBERO_PYTHON}" "${LIBERO_HOME}" "${TASK_SUITE_NAME}" \
  "${NUM_TRIALS_PER_TASK}" "${MAX_TASKS}" "${NUM_STEPS_WAIT}" "${SEED}" \
  "${UNNORM_KEY}" "${VIDEO_OUT_PATH}" "${LOG_FILE}" "${SCRIPT_DIR}/eval_libero.sh"

# Keep each window open after a process exits so its traceback and exit code
# remain visible. Type `exit` in that window after inspection.
server_cmd="set -o pipefail; ${server_cmd}; code=\$?; echo \"[server exited with code \$code]\"; exec bash"
eval_cmd="${eval_cmd}; code=\$?; echo \"[eval exited with code \$code]\"; exec bash"

tmux new-session -d -s "${SESSION}" -n server "${server_cmd}"
tmux new-window -t "${SESSION}" -n eval "${eval_cmd}"

echo "Started tmux session: ${SESSION}"
echo "  checkpoint : ${CKPT}"
echo "  server log : ${SERVER_LOG}"
echo "  windows    : Ctrl-b then 0 (server) or 1 (eval)"
echo "  detach     : Ctrl-b, then d"
echo "  reattach   : $0 attach"
echo "  stop       : $0 stop"

[[ "${ATTACH:-1}" == "0" ]] || exec tmux attach -t "${SESSION}"
