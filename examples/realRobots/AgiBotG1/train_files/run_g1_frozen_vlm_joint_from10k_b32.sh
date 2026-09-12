#!/usr/bin/env bash
set -euo pipefail

# Frozen-VLM second stage: start from the completed 10K G1 adapter warmup and
# train the shared DiT plus native action I/O on the corrected LIBERO/G1 mix.
# The launcher owns one tmux session and never overwrites a non-empty run dir.

ROOT_DIR="/data/tzq/starVLA-RollFlow"
CONFIG_PATH="$ROOT_DIR/examples/realRobots/AgiBotG1/train_files/finetune_g1_frozen_vlm_joint_from10k_b32.yaml"
PYTHON_BIN="${STARVLA_PYTHON:-/data/miniconda3/envs/starVLA/bin/python}"
SESSION="${TMUX_SESSION:-agibot_g1_frozen_vlm_joint_from10k_b32}"
CHECKPOINT="/data/tzq/starVLA_checkpoints/agibot_g1_adapter_warmup_55k_10k_v1/checkpoints/steps_10000_pytorch_model.pt"
RUN_DIR="/data/tzq/starVLA_checkpoints/agibot_g1_frozen_vlm_joint_from10k_b32_v1"
LOG_PATH="$RUN_DIR/train.log"
PORT="${MAIN_PROCESS_PORT:-29764}"

command -v tmux >/dev/null 2>&1 || { echo "tmux is required" >&2; exit 1; }
[[ -f "$CONFIG_PATH" ]] || { echo "Missing config: $CONFIG_PATH" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 1; }
[[ -f "$CHECKPOINT" ]] || { echo "Missing 10K checkpoint: $CHECKPOINT" >&2; exit 1; }
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
export STARVLA_PLAIN_LOGS=1

tmux new-session -d -s "$SESSION" bash -lc "
  set -euo pipefail
  export CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'
  export PYTORCH_CUDA_ALLOC_CONF='$PYTORCH_CUDA_ALLOC_CONF'
  export WANDB_MODE=disabled
  export WANDB_DISABLED=true
  export STARVLA_PLAIN_LOGS=1
  export NO_COLOR=1
  source /data/miniconda3/etc/profile.d/conda.sh
  conda activate starVLA
  cd '$ROOT_DIR'
  echo '[run] frozen-VLM joint fine-tune: DiT + Franka/G1 action I/O'
  exec '$PYTHON_BIN' -u -m accelerate.commands.launch \
    --config_file '$ROOT_DIR/starVLA/config/deepseeds/deepspeed_zero2.yaml' \
    --main_process_port '$PORT' --num_processes 4 \
    starVLA/training/train_starvla.py --config_yaml '$CONFIG_PATH'
"
tmux pipe-pane -t "$SESSION:0" -o "cat >> '$LOG_PATH'"

echo "Started: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Log/checkpoints: $RUN_DIR"
