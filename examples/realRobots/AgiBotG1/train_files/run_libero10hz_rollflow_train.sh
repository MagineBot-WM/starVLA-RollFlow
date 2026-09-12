#!/usr/bin/env bash
set -euo pipefail

# Run one controlled-rate experiment at a time.  The ``from_scratch_*`` modes
# use the local base VLM and freshly initialized action modules; ``mixed_55k``
# initializes from the clean LIBERO 55K checkpoint.  The controlled ablation
# modes use
# the semantically corrected Apple overlay, freeze the VLM, save every 1000
# steps, and have independent tmux sessions/ports/output directories.
ROOT_DIR="/data/tzq/starVLA-RollFlow"
CONFIG_DIR="$ROOT_DIR/examples/realRobots/AgiBotG1/train_files"
DATA_ROOT="${DATA_ROOT:-/data/tzq/datasets/starVLA/Datasets}"
PYTHON_BIN="${STARVLA_PYTHON:-/data/miniconda3/envs/starVLA/bin/python}"
EXPERIMENT="${EXPERIMENT:-standard}"

case "$EXPERIMENT" in
  standard)
    CONFIG_PATH="$CONFIG_DIR/finetune_libero10hz_rollflow_standard.yaml"
    SESSION="rollflow_libero10hz_standard"
    PORT=29741
    ;;
  armweighted)
    CONFIG_PATH="$CONFIG_DIR/finetune_libero10hz_rollflow_armweighted.yaml"
    SESSION="rollflow_libero10hz_armweighted"
    PORT=29742
    ;;
  raw_standard)
    CONFIG_PATH="$CONFIG_DIR/finetune_libero10hz_rollflow_raw.yaml"
    SESSION="rollflow_libero10hz_raw_standard"
    PORT=29743
    ;;
  raw_armweighted)
    CONFIG_PATH="$CONFIG_DIR/finetune_libero10hz_rollflow_raw_armweighted.yaml"
    SESSION="rollflow_libero10hz_raw_armweighted"
    PORT=29744
    ;;
  corrected_adapters)
    CONFIG_PATH="$CONFIG_DIR/finetune_libero10hz_apple_corrected_adapters.yaml"
    SESSION="rollflow_libero10hz_apple_corrected_adapters"
    PORT=29745
    ;;
  mixed_55k)
    CONFIG_PATH="$CONFIG_DIR/finetune_libero10hz_apple_corrected_55k.yaml"
    SESSION="rollflow_libero10hz_apple_corrected_55k"
    PORT=29746
    ;;
  mixed_55k_lsd_gate)
    CONFIG_PATH="$CONFIG_DIR/finetune_libero10hz_apple_corrected_55k_lsd_gate.yaml"
    SESSION="rollflow_libero10hz_apple_corrected_55k_lsd_gate"
    PORT=29749
    ;;
  mixed_55k_lsd_target_gate)
    CONFIG_PATH="$CONFIG_DIR/finetune_libero10hz_apple_corrected_55k_lsd_target_gate.yaml"
    SESSION="rollflow_libero10hz_apple_corrected_55k_lsd_target_gate"
    PORT=29750
    ;;
  ablation_no_ot_no_gate)
    CONFIG_PATH="$CONFIG_DIR/finetune_libero10hz_apple_corrected_55k_ablation_no_ot_no_gate.yaml"
    SESSION="rollflow_ablation_no_ot_no_gate_55k"
    PORT=29751
    ;;
  ablation_ot_no_gate)
    CONFIG_PATH="$CONFIG_DIR/finetune_libero10hz_apple_corrected_55k_ablation_ot_no_gate.yaml"
    SESSION="rollflow_ablation_ot_no_gate_55k"
    PORT=29752
    ;;
  ablation_ot_gate)
    CONFIG_PATH="$CONFIG_DIR/finetune_libero10hz_apple_corrected_55k_ablation_ot_gate.yaml"
    SESSION="rollflow_ablation_ot_gate_55k"
    PORT=29753
    ;;
  ablation_ot_no_scaling_no_gate)
    CONFIG_PATH="$CONFIG_DIR/finetune_libero10hz_apple_corrected_55k_ablation_ot_no_scaling_no_gate.yaml"
    SESSION="rollflow_ablation_ot_no_scaling_no_gate_55k"
    PORT=29754
    ;;
  ablation_no_ot_no_gate_no_scaling)
    CONFIG_PATH="$CONFIG_DIR/finetune_libero10hz_apple_corrected_55k_ablation_no_ot_no_gate_no_scaling.yaml"
    SESSION="rollflow_ablation_no_ot_no_gate_no_scaling_55k"
    PORT=29755
    ;;
  ablation_ot_no_gate_no_scaling)
    CONFIG_PATH="$CONFIG_DIR/finetune_libero10hz_apple_corrected_55k_ablation_ot_no_gate_no_scaling.yaml"
    SESSION="rollflow_ablation_ot_no_gate_no_scaling_55k"
    PORT=29756
    ;;
  ablation_ot_gate_no_scaling)
    CONFIG_PATH="$CONFIG_DIR/finetune_libero10hz_apple_corrected_55k_ablation_ot_gate_no_scaling.yaml"
    SESSION="rollflow_ablation_ot_gate_no_scaling_55k"
    PORT=29757
    ;;
  from_scratch_g1)
    CONFIG_PATH="$CONFIG_DIR/finetune_agibot_g1_apple_corrected_from_scratch.yaml"
    SESSION="rollflow_agibot_g1_apple_from_scratch"
    PORT=29747
    ;;
  from_scratch_mixed)
    CONFIG_PATH="$CONFIG_DIR/finetune_libero10hz_apple_corrected_from_scratch.yaml"
    SESSION="rollflow_libero10hz_apple_from_scratch_mixed"
    PORT=29748
    ;;
  *)
    echo "EXPERIMENT must be standard, armweighted, raw_standard, raw_armweighted, corrected_adapters, mixed_55k, mixed_55k_lsd_gate, mixed_55k_lsd_target_gate, ablation_no_ot_no_gate, ablation_ot_no_gate, ablation_ot_gate, ablation_ot_no_scaling_no_gate, ablation_no_ot_no_gate_no_scaling, ablation_ot_no_gate_no_scaling, ablation_ot_gate_no_scaling, from_scratch_g1, or from_scratch_mixed" >&2
    exit 2
    ;;
esac

# Keep the durable log beside that run's checkpoints/configuration.  The run_id
# is read from the selected YAML so a new experiment cannot accidentally append
# to a different run's log.
RUN_ID="$(sed -n 's/^run_id:[[:space:]]*//p' "$CONFIG_PATH" | head -n 1)"
RUN_ID="${RUN_ID%\"}"
RUN_ID="${RUN_ID#\"}"
RUN_DIR="/data/tzq/starVLA_checkpoints/${RUN_ID}"
LOG_PATH="${LOG_PATH:-${RUN_DIR}/train.log}"

# pipe-pane starts asynchronously; create the run directory first so the
# first startup messages cannot be lost while train_starvla initializes it.
mkdir -p "$RUN_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Refusing to replace existing tmux session: $SESSION" >&2
  echo "Attach with: tmux attach -t $SESSION" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="disabled"

tmux new-session -d -s "$SESSION" bash -lc "
  set -euo pipefail
  export CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'
  export PYTORCH_CUDA_ALLOC_CONF='$PYTORCH_CUDA_ALLOC_CONF'
  export WANDB_MODE=disabled
  export WANDB_DISABLED=true
  source /data/miniconda3/etc/profile.d/conda.sh
  conda activate starVLA
  cd '$ROOT_DIR'
  if [[ '$EXPERIMENT' == corrected_adapters || '$EXPERIMENT' == mixed_55k || '$EXPERIMENT' == mixed_55k_lsd_gate || '$EXPERIMENT' == mixed_55k_lsd_target_gate || '$EXPERIMENT' == ablation_no_ot_no_gate || '$EXPERIMENT' == ablation_ot_no_gate || '$EXPERIMENT' == ablation_ot_gate || '$EXPERIMENT' == ablation_ot_no_scaling_no_gate || '$EXPERIMENT' == ablation_no_ot_no_gate_no_scaling || '$EXPERIMENT' == ablation_ot_no_gate_no_scaling || '$EXPERIMENT' == ablation_ot_gate_no_scaling || '$EXPERIMENT' == from_scratch_g1 || '$EXPERIMENT' == from_scratch_mixed ]]; then
    '$PYTHON_BIN' examples/realRobots/AgiBotG1/dataset_tools/audit_apple_mapping.py \
      --source '$DATA_ROOT/pick_up_the_apple' \
      --overlay '$DATA_ROOT/agibot-g1-apple-corrected'
  fi
  exec accelerate launch \
    --config_file '$ROOT_DIR/starVLA/config/deepseeds/deepspeed_zero2.yaml' \
    --main_process_port '$PORT' \
    --num_processes 4 \
    starVLA/training/train_starvla.py \
    --config_yaml '$CONFIG_PATH'
"
# tmux keeps the live view while pipe-pane appends the same bytes to disk.
# This is attached after creation so it also works for detached launches.
tmux pipe-pane -t "$SESSION:0" -o "cat >> '$LOG_PATH'"

echo "Started $EXPERIMENT in tmux session: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b then d"
echo "Log: $LOG_PATH"
