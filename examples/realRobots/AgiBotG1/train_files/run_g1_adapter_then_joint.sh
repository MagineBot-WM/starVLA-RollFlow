#!/usr/bin/env bash
set -euo pipefail

# Two-stage, isolated experiment:
#   1) 55K LIBERO -> G1-only, adapter-only warmup for 10K steps.
#   2) Stage-1 checkpoint -> balanced LIBERO/G1 joint fine-tune for 5K steps.
#
# Both stages run in one durable tmux session.  Each stage keeps its own
# checkpoint directory and train.log; the launcher log records the transition.

ROOT_DIR="/data/tzq/starVLA-RollFlow"
CONFIG_DIR="$ROOT_DIR/examples/realRobots/AgiBotG1/train_files"
PYTHON_BIN="${STARVLA_PYTHON:-/data/miniconda3/envs/starVLA/bin/python}"
DATA_ROOT="${DATA_ROOT:-/data/tzq/datasets/starVLA/Datasets}"
SESSION="${TMUX_SESSION:-agibot_g1_adapter_then_joint}"
PORT="${MAIN_PROCESS_PORT:-29761}"

STAGE1_CONFIG="$CONFIG_DIR/finetune_g1_adapter_warmup_55k.yaml"
STAGE2_CONFIG="$CONFIG_DIR/finetune_g1_adapter_joint_5k.yaml"
STAGE1_DIR="/data/tzq/starVLA_checkpoints/agibot_g1_adapter_warmup_55k_10k_v1"
STAGE2_DIR="/data/tzq/starVLA_checkpoints/agibot_g1_adapter_warmup_55k_joint_5k_v1"
STAGE1_CKPT="$STAGE1_DIR/checkpoints/steps_10000_pytorch_model.pt"
LAUNCH_LOG="${LAUNCH_LOG:-$STAGE2_DIR/launcher.log}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required but was not found" >&2
  exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Refusing to replace existing tmux session: $SESSION" >&2
  echo "Attach with: tmux attach -t $SESSION" >&2
  exit 1
fi
for config in "$STAGE1_CONFIG" "$STAGE2_CONFIG"; do
  [[ -f "$config" ]] || { echo "Missing config: $config" >&2; exit 1; }
done

mkdir -p "$STAGE1_DIR" "$STAGE2_DIR"
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

  echo '[stage1] auditing corrected Apple mapping'
  '$PYTHON_BIN' examples/realRobots/AgiBotG1/dataset_tools/audit_apple_mapping.py \
    --source '$DATA_ROOT/pick_up_the_apple' \
    --overlay '$DATA_ROOT/agibot-g1-apple-corrected'

  echo '[stage1] G1 adapter-only warmup: 10000 steps'
  '$PYTHON_BIN' -u -m accelerate.commands.launch \
    --config_file '$ROOT_DIR/starVLA/config/deepseeds/deepspeed_zero2.yaml' \
    --main_process_port '$PORT' --num_processes 4 \
    starVLA/training/train_starvla.py --config_yaml '$STAGE1_CONFIG' \
    2>&1 | tee -a '$STAGE1_DIR/train.log'

  [[ -f '$STAGE1_CKPT' ]] || { echo '[stage2] stage-1 checkpoint missing: $STAGE1_CKPT' >&2; exit 1; }
  echo '[stage2] balanced joint fine-tune: 5000 steps'
  '$PYTHON_BIN' -u -m accelerate.commands.launch \
    --config_file '$ROOT_DIR/starVLA/config/deepseeds/deepspeed_zero2.yaml' \
    --main_process_port '$PORT' --num_processes 4 \
    starVLA/training/train_starvla.py --config_yaml '$STAGE2_CONFIG' \
    2>&1 | tee -a '$STAGE2_DIR/train.log'

  echo '[done] both stages completed'
"

tmux pipe-pane -t "$SESSION:0" -o "cat >> '$LAUNCH_LOG'"
echo "Started: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Stage-1 log: $STAGE1_DIR/train.log"
echo "Stage-2 log: $STAGE2_DIR/train.log"
echo "Launcher log: $LAUNCH_LOG"
