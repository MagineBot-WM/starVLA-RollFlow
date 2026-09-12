#!/usr/bin/env bash
# Queue the three requested formal experiments on the same four GPUs.
# Each child is launched by the maintained launcher and gets its own tmux
# session, port, run_id, and checkpoint directory.  Runs are intentionally
# sequential because three four-GPU jobs would otherwise compete for memory.
set -euo pipefail

ROOT_DIR="/data/tzq/starVLA-RollFlow"
LAUNCHER="$ROOT_DIR/examples/realRobots/AgiBotG1/train_files/run_libero10hz_rollflow_train.sh"
SCRIPT_PATH="$(realpath "$0")"
QUEUE_SESSION="${TMUX_SESSION:-agibot_g1_formal_suite}"
QUEUE_LOG="${QUEUE_LOG:-/data/tzq/starVLA_checkpoints/agibot_g1_formal_suite.log}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

if [[ "${1:-}" == "--worker" ]]; then
  experiments=(mixed_55k_lsd_target_gate from_scratch_g1 from_scratch_mixed)
  sessions=(rollflow_libero10hz_apple_corrected_55k_lsd_target_gate rollflow_agibot_g1_apple_from_scratch rollflow_libero10hz_apple_from_scratch_mixed)
  for index in "${!experiments[@]}"; do
    experiment="${experiments[index]}"
    child="${sessions[index]}"
    echo "[formal-suite] starting $experiment (child tmux: $child)"
    EXPERIMENT="$experiment" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$LAUNCHER"
    while tmux has-session -t "$child" 2>/dev/null; do
      sleep 30
    done
    echo "[formal-suite] child $experiment ended; moving to the next experiment"
  done
  echo "[formal-suite] all three experiments finished"
  exit 0
fi

if tmux has-session -t "$QUEUE_SESSION" 2>/dev/null; then
  echo "Refusing to replace existing tmux session: $QUEUE_SESSION" >&2
  echo "Attach with: tmux attach -t $QUEUE_SESSION" >&2
  exit 1
fi

tmux new-session -d -s "$QUEUE_SESSION" "$SCRIPT_PATH --worker"
tmux pipe-pane -t "$QUEUE_SESSION:0" -o "cat >> '$QUEUE_LOG'"

echo "Started queue in tmux session: $QUEUE_SESSION"
echo "Attach: tmux attach -t $QUEUE_SESSION"
echo "Child sessions run sequentially: mixed_55k_lsd_target_gate -> from_scratch_g1 -> from_scratch_mixed"
echo "Queue log: $QUEUE_LOG"
