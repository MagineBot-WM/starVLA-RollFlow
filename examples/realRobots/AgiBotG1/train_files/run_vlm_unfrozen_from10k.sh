#!/usr/bin/env bash
set -euo pipefail

# Fine-tune the completed mixed 10K checkpoint with VLM unfrozen.  The current
# adapter warmup uses all four GPUs, so this job waits for that tmux session to
# finish instead of competing for memory or interrupting it.
ROOT_DIR="/data/tzq/starVLA-RollFlow"
PYTHON_BIN="${STARVLA_PYTHON:-/data/miniconda3/envs/starVLA/bin/python}"
CONFIG_PATH="$ROOT_DIR/examples/realRobots/AgiBotG1/train_files/finetune_from_mixed10k_vlm_unfrozen.yaml"
RUN_DIR="/data/tzq/starVLA_checkpoints/libero10hz_pick_up_the_apple_vlm_unfrozen_from10k_v1"
SESSION="${TMUX_SESSION:-agibot_g1_vlm_unfrozen_from10k}"
WAIT_SESSION="${WAIT_SESSION:-agibot_g1_adapter_then_joint}"
PORT="${MAIN_PROCESS_PORT:-29762}"
LOG_PATH="${LOG_PATH:-$RUN_DIR/train.log}"

command -v tmux >/dev/null 2>&1 || { echo "tmux is required" >&2; exit 1; }
[[ -f "$CONFIG_PATH" ]] || { echo "Missing config: $CONFIG_PATH" >&2; exit 1; }
[[ -f "/data/tzq/starVLA_checkpoints/libero10hz_pick_up_the_apple_corrected_55k_lsd_finite_hard_gate_v1/checkpoints/steps_10000_pytorch_model.pt" ]] || {
  echo "The completed mixed 10K checkpoint is missing" >&2
  exit 1
}
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Refusing to replace existing tmux session: $SESSION" >&2
  echo "Attach with: tmux attach -t $SESSION" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE=disabled

tmux new-session -d -s "$SESSION" bash -lc "
  set -euo pipefail
  export CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'
  export PYTORCH_CUDA_ALLOC_CONF='$PYTORCH_CUDA_ALLOC_CONF'
  export WANDB_MODE=disabled
  export WANDB_DISABLED=true
  source /data/miniconda3/etc/profile.d/conda.sh
  conda activate starVLA
  cd '$ROOT_DIR'
  echo '[wait] waiting for tmux session $WAIT_SESSION to release the GPUs'
  while tmux has-session -t '$WAIT_SESSION' 2>/dev/null; do sleep 30; done
  echo '[run] VLM-unfrozen fine-tune from completed mixed 10K checkpoint'
  exec '$PYTHON_BIN' -u -m accelerate.commands.launch \
    --config_file '$ROOT_DIR/starVLA/config/deepseeds/deepspeed_zero2.yaml' \
    --main_process_port '$PORT' --num_processes 4 \
    starVLA/training/train_starvla.py --config_yaml '$CONFIG_PATH'
"
tmux pipe-pane -t "$SESSION:0" -o "cat >> '$LOG_PATH'"

echo "Started queued run: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Waiting for: $WAIT_SESSION"
echo "Log: $LOG_PATH"
