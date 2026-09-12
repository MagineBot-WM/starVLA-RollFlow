#!/usr/bin/env bash
set -euo pipefail

# Queue the VLM-unfrozen joint stage after the currently running G1 adapter
# warmup has produced its 10K checkpoint.  The existing warmup session also
# has a preconfigured follow-up stage; waiting for that whole session avoids
# GPU contention and guarantees this run starts from the immutable 10K file.

ROOT_DIR="/data/tzq/starVLA-RollFlow"
CONFIG_PATH="$ROOT_DIR/examples/realRobots/AgiBotG1/train_files/finetune_g1_vlm_unfrozen_joint_from10k.yaml"
PYTHON_BIN="${STARVLA_PYTHON:-/data/miniconda3/envs/starVLA/bin/python}"
SESSION="${TMUX_SESSION:-agibot_g1_vlm_unfrozen_joint_from10k}"
WAIT_SESSION="${WAIT_SESSION:-agibot_g1_adapter_then_joint}"
CHECKPOINT="/data/tzq/starVLA_checkpoints/agibot_g1_adapter_warmup_55k_10k_v1/checkpoints/steps_10000_pytorch_model.pt"
RUN_DIR="/data/tzq/starVLA_checkpoints/agibot_g1_vlm_unfrozen_joint_from10k_b16_ga2_20k_v3"
LOG_PATH="$RUN_DIR/train.log"
PORT="${MAIN_PROCESS_PORT:-29763}"

command -v tmux >/dev/null 2>&1 || { echo "tmux is required" >&2; exit 1; }
[[ -f "$CONFIG_PATH" ]] || { echo "Missing config: $CONFIG_PATH" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 1; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Refusing to replace existing tmux session: $SESSION" >&2
  echo "Attach with: tmux attach -t $SESSION" >&2
  exit 1
fi
if [[ -e "$RUN_DIR" && -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to reuse non-empty run directory: $RUN_DIR" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export NO_COLOR=1
export STARVLA_PLAIN_LOGS=1

tmux new-session -d -s "$SESSION" bash -lc "
  set -euo pipefail
  export CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'
  export PYTORCH_CUDA_ALLOC_CONF='$PYTORCH_CUDA_ALLOC_CONF'
  export WANDB_MODE=disabled
  export WANDB_DISABLED=true
  export NO_COLOR=1
  export STARVLA_PLAIN_LOGS=1
  source /data/miniconda3/etc/profile.d/conda.sh
  conda activate starVLA
  cd '$ROOT_DIR'

  echo '[queue] waiting for 10K checkpoint: $CHECKPOINT'
  while [[ ! -f '$CHECKPOINT' ]]; do
    if ! tmux has-session -t '$WAIT_SESSION' 2>/dev/null; then
      echo '[queue] warmup session ended before the 10K checkpoint appeared' >&2
      exit 1
    fi
    sleep 30
  done
  echo '[queue] 10K checkpoint is ready; waiting for $WAIT_SESSION to release GPUs'
  while tmux has-session -t '$WAIT_SESSION' 2>/dev/null; do sleep 30; done

  echo '[run] VLM-unfrozen LIBERO/G1 joint fine-tune from adapter 10K'
  exec '$PYTHON_BIN' -u -m accelerate.commands.launch \
    --config_file '$ROOT_DIR/starVLA/config/deepseeds/deepspeed_zero2.yaml' \
    --main_process_port '$PORT' --num_processes 4 \
    starVLA/training/train_starvla.py --config_yaml '$CONFIG_PATH'
"
tmux pipe-pane -t "$SESSION:0" -o "cat >> '$LOG_PATH'"

echo "Queued: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Waiting for: $WAIT_SESSION"
echo "Checkpoint: $CHECKPOINT"
echo "Log/checkpoints: $RUN_DIR"
