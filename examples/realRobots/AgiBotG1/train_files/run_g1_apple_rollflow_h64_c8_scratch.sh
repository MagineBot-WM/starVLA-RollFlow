#!/usr/bin/env bash
set -euo pipefail

# G1-only Apple RollFlow experiment.  The default invocation waits for the
# currently occupied GPUs, runs a short fresh-model preflight, then starts the
# formal run in a distinct directory.  Set WAIT_FOR_GPUS=0 to fail fast when
# the cards are busy, or PREFLIGHT_STEPS=0 to skip the smoke run explicitly.

SCRIPT_PATH="$(readlink -f "$0")"
ROOT_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
PYTHON_BIN="${STARVLA_PYTHON:-/data/miniconda3/envs/starVLA/bin/python}"
CONFIG_PATH="${CONFIG_YAML:-${ROOT_DIR}/examples/realRobots/AgiBotG1/train_files/agibot_g1_apple_rollflow_h64_c8_scratch.yaml}"
DATA_ROOT="${DATA_ROOT:-/data/tzq/datasets/starVLA/Datasets}"
DATA_MIX="${DATA_MIX:-agibot_g1_apple_corrected_h64}"
RUN_ROOT="${RUN_ROOT:-/data/tzq/starVLA_checkpoints}"
RUN_ID="${RUN_ID:-agibot_g1_apple_rollflow_h64_c8_scratch_v1}"
SESSION="${TMUX_SESSION:-${RUN_ID}}"
NUM_GPUS="${NUM_GPUS:-4}"
BATCH_PER_GPU="${BATCH_PER_GPU:-16}"
PORT="${MAIN_PROCESS_PORT:-29864}"
WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
PREFLIGHT_STEPS="${PREFLIGHT_STEPS:-4}"
GPU_MEMORY_LIMIT_MIB="${GPU_MEMORY_LIMIT_MIB:-12000}"
GPU_STABLE_POLLS="${GPU_STABLE_POLLS:-3}"
GPU_STABLE_INTERVAL_SEC="${GPU_STABLE_INTERVAL_SEC:-20}"

RUN_DIR="${RUN_ROOT}/${RUN_ID}"
LOG_PATH="${RUN_DIR}/train.log"
SMOKE_ID="${RUN_ID}_preflight"
SMOKE_DIR="${RUN_ROOT}/${SMOKE_ID}"
SMOKE_PORT="${PREFLIGHT_PORT:-29865}"

command -v tmux >/dev/null 2>&1 || { echo "tmux is required" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 1; }
[[ -f "$CONFIG_PATH" ]] || { echo "Missing config: $CONFIG_PATH" >&2; exit 1; }
[[ -d "$DATA_ROOT/agibot-g1-apple-corrected" ]] || {
  echo "Missing corrected Apple overlay under $DATA_ROOT" >&2
  exit 1
}
if [[ "${RUN_IN_TMUX:-0}" != 1 ]] && tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Refusing to replace existing tmux session: $SESSION" >&2
  echo "Attach with: tmux attach -t $SESSION" >&2
  exit 1
fi
if [[ -e "$RUN_DIR" && -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to reuse non-empty run directory: $RUN_DIR" >&2
  exit 1
fi
if [[ "$PREFLIGHT_STEPS" != 0 && -e "$SMOKE_DIR" && -n "$(find "$SMOKE_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to reuse non-empty preflight directory: $SMOKE_DIR" >&2
  exit 1
fi

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RUN_DIR"

cat <<INFO
Queued G1 Apple H64 RollFlow training
  config:    $CONFIG_PATH
  data mix:  $DATA_MIX
  horizon:   64, execution chunk: 8, inference steps: 8
  batch:     ${BATCH_PER_GPU}/GPU x ${NUM_GPUS} GPUs
  VLM:       frozen (qwen_vl_interface); action head: fresh
  run dir:   $RUN_DIR
  session:   $SESSION
INFO

if [[ "${RUN_IN_TMUX:-0}" != 1 ]]; then
  tmux new-session -d -s "$SESSION" env \
    RUN_IN_TMUX=1 \
    STARVLA_DIR="$ROOT_DIR" \
    STARVLA_PYTHON="$PYTHON_BIN" \
    CONFIG_YAML="$CONFIG_PATH" \
    DATA_ROOT="$DATA_ROOT" \
    DATA_MIX="$DATA_MIX" \
    RUN_ROOT="$RUN_ROOT" \
    RUN_ID="$RUN_ID" \
    TMUX_SESSION="$SESSION" \
    NUM_GPUS="$NUM_GPUS" \
    BATCH_PER_GPU="$BATCH_PER_GPU" \
    MAIN_PROCESS_PORT="$PORT" \
    WAIT_FOR_GPUS="$WAIT_FOR_GPUS" \
    PREFLIGHT_STEPS="$PREFLIGHT_STEPS" \
    GPU_MEMORY_LIMIT_MIB="$GPU_MEMORY_LIMIT_MIB" \
    GPU_STABLE_POLLS="$GPU_STABLE_POLLS" \
    GPU_STABLE_INTERVAL_SEC="$GPU_STABLE_INTERVAL_SEC" \
    PREFLIGHT_PORT="$SMOKE_PORT" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    "$SCRIPT_PATH"
  echo "Started queue: $SESSION"
  echo "Attach: tmux attach -t $SESSION"
  echo "Log/checkpoints: $RUN_DIR"
  exit 0
fi

# Child process inside tmux.  Keeping the queue logic here avoids fragile
# nested quoting and ensures failures are visible in the run log.
source /data/miniconda3/etc/profile.d/conda.sh
conda activate starVLA
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export STARVLA_PLAIN_LOGS=1
export NO_COLOR=1
exec > >(tee -a "$LOG_PATH") 2>&1

gpu_ready() {
  local used
  while read -r used; do
    used="${used//[[:space:]]/}"
    [[ "$used" =~ ^[0-9]+$ ]] || return 1
    (( used <= GPU_MEMORY_LIMIT_MIB )) || return 1
  done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
}

run_train() {
  local run_id="$1" port="$2" max_steps="$3" save_interval="$4" eval_interval="$5" log_frequency="$6"
  "$PYTHON_BIN" -u -m accelerate.commands.launch \
    --config_file "$ROOT_DIR/starVLA/config/deepseeds/deepspeed_zero2.yaml" \
    --main_process_port "$port" --num_processes "$NUM_GPUS" \
    starVLA/training/train_starvla.py \
    --config_yaml "$CONFIG_PATH" \
    --datasets.vla_data.data_root_dir "$DATA_ROOT" \
    --datasets.vla_data.data_mix "$DATA_MIX" \
    --datasets.vla_data.per_device_batch_size "$BATCH_PER_GPU" \
    --trainer.freeze_modules qwen_vl_interface \
    --trainer.max_train_steps "$max_steps" \
    --trainer.save_interval "$save_interval" \
    --trainer.eval_interval "$eval_interval" \
    --trainer.logging_frequency "$log_frequency" \
    --trainer.is_resume 0 \
    --run_root_dir "$RUN_ROOT" \
    --run_id "$run_id"
}

echo "[queue] $(date -Is) waiting for ${NUM_GPUS} GPUs below ${GPU_MEMORY_LIMIT_MIB} MiB each"
if [[ "$WAIT_FOR_GPUS" != 0 ]]; then
  until gpu_ready; do
    echo "[queue] $(date -Is) GPUs still occupied; retrying in 60s"
    sleep 60
  done
  stable_polls=1
  while (( stable_polls < GPU_STABLE_POLLS )); do
    sleep "$GPU_STABLE_INTERVAL_SEC"
    if gpu_ready; then
      stable_polls=$((stable_polls + 1))
      echo "[queue] $(date -Is) GPU precondition still satisfied (${stable_polls}/${GPU_STABLE_POLLS})"
    else
      stable_polls=0
      echo "[queue] $(date -Is) GPU usage changed; restarting stable-free check"
      until gpu_ready; do
        echo "[queue] $(date -Is) GPUs still occupied; retrying in 60s"
        sleep 60
      done
      stable_polls=1
    fi
  done
elif ! gpu_ready; then
  echo '[queue] GPUs are busy and WAIT_FOR_GPUS=0; refusing to start' >&2
  exit 2
fi
echo "[queue] $(date -Is) GPU precondition satisfied"

"$PYTHON_BIN" examples/realRobots/AgiBotG1/dataset_tools/audit_apple_mapping.py \
  --source "$DATA_ROOT/pick_up_the_apple" \
  --overlay "$DATA_ROOT/agibot-g1-apple-corrected"

if [[ "$PREFLIGHT_STEPS" != 0 ]]; then
  echo "[preflight] starting ${PREFLIGHT_STEPS}-step fresh-model smoke run"
  run_train "$SMOKE_ID" "$SMOKE_PORT" "$PREFLIGHT_STEPS" 1000000 1000000 1
  echo '[preflight] completed successfully; starting formal run'
fi

echo "[run] starting $RUN_ID: $DATA_MIX, RollFlow H64/C8, frozen VLM"
run_train "$RUN_ID" "$PORT" 20000 1000 1000 20
