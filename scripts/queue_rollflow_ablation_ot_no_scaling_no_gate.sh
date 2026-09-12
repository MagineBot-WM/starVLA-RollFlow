#!/usr/bin/env bash
set -euo pipefail

# Queue the controlled historical ablation without occupying any GPU while
# the three 55K ablation runs are still active.  The queued job starts only
# after all expected runs have exited successfully and their 20K checkpoints
# exist.
ROOT_DIR="/data/tzq/starVLA-RollFlow"
RUN_SCRIPT="$ROOT_DIR/examples/realRobots/AgiBotG1/train_files/run_libero10hz_rollflow_train.sh"
QUEUE_SESSION="rollflow_ablation_ot_no_scaling_no_gate_queue"
QUEUE_LOG="/data/tzq/starVLA_checkpoints/rollflow_ablation_analysis/ot_no_scaling_queue.log"
POLL_SECONDS=60

WATCH_SESSIONS=(
  rollflow_ablation_no_ot_no_gate_55k
  rollflow_ablation_ot_no_gate_55k
  rollflow_ablation_ot_gate_55k
)

EXPECTED_CHECKPOINTS=(
  /data/tzq/starVLA_checkpoints/libero10hz_pick_up_the_apple_ablation_no_ot_no_gate_55k_20k_v1/checkpoints/steps_20000_pytorch_model.pt
  /data/tzq/starVLA_checkpoints/libero10hz_pick_up_the_apple_ablation_ot_no_gate_55k_20k_v1/checkpoints/steps_20000_pytorch_model.pt
  /data/tzq/starVLA_checkpoints/libero10hz_pick_up_the_apple_ablation_ot_gate_55k_20k_v1/checkpoints/steps_20000_pytorch_model.pt
)

if tmux has-session -t "$QUEUE_SESSION" 2>/dev/null; then
  echo "Queue watcher already exists: $QUEUE_SESSION" >&2
  exit 1
fi

mkdir -p "$(dirname "$QUEUE_LOG")"

tmux new-session -d -s "$QUEUE_SESSION" bash -lc "
  set -euo pipefail
  exec > >(tee -a '$QUEUE_LOG') 2>&1
  echo '[queue] waiting for current RollFlow ablations to finish'
  while true; do
    alive=0
    for session in ${WATCH_SESSIONS[*]}; do
      if tmux has-session -t \"\$session\" 2>/dev/null; then
        alive=1
        echo \"[queue] still running: \$session\"
      fi
    done
    if [[ \"\$alive\" == 0 ]]; then
      break
    fi
    sleep '$POLL_SECONDS'
  done

  for checkpoint in ${EXPECTED_CHECKPOINTS[*]}; do
    if [[ ! -f \"\$checkpoint\" ]]; then
      echo \"[queue] refusing to start: missing expected checkpoint \$checkpoint\" >&2
      exit 1
    fi
  done

  echo '[queue] all current ablations completed; starting OT + No Scaling + No Gate'
  export EXPERIMENT=ablation_ot_no_scaling_no_gate
  exec '$RUN_SCRIPT'
"

echo "Queued experiment in tmux session: $QUEUE_SESSION"
echo "Attach: tmux attach -t $QUEUE_SESSION"
echo "Queue log: $QUEUE_LOG"
