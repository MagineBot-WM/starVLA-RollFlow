#!/usr/bin/env bash
# Standalone real-G1 fine-tuning, detached with tmux (no nohup).
# Attach: tmux attach -t agibot_g1_finetune
# Detach: Ctrl-b, then d. Stop training: attach, then Ctrl-c.
set -euo pipefail
script_path="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
repo_root="$(cd "$(dirname "$0")/../../../.." && pwd)"
python_bin="${STARVLA_PYTHON:-/data/miniconda3/envs/starVLA/bin/python}"
config="${CONFIG_YAML:-$(dirname "$0")/finetune_real.yaml}"
config="$(realpath "$config")"
session="${TMUX_SESSION:-agibot_g1_finetune}"
run_id="${RUN_ID:-libero_agibot_g1_stage2_55k_b32_c500_fm50}"
run_root="${RUN_ROOT:-/data/tzq/starVLA_checkpoints}"
run_dir="$run_root/$run_id"
port="${MAIN_PROCESS_PORT:-29511}"
max_steps="${MAX_TRAIN_STEPS:-10000}"
save_interval="${SAVE_INTERVAL:-1000}"
eval_interval="${EVAL_INTERVAL:-1000}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WANDB_MODE=disabled PYTHONUNBUFFERED=1 NO_ALBUMENTATIONS_UPDATE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"
cd "$repo_root"
if [[ "${1:-}" != "--worker" ]]; then
  [[ ! -e "$run_root/$run_id" ]] || { echo "Run exists: $run_root/$run_id; choose a new RUN_ID." >&2; exit 1; }
  ! tmux has-session -t "$session" 2>/dev/null || { echo "Session exists: $session" >&2; exit 1; }
  mkdir -p "$run_root"
  printf -v command 'mkdir -p %q && env STARVLA_PYTHON=%q CONFIG_YAML=%q RUN_ID=%q RUN_ROOT=%q MAIN_PROCESS_PORT=%q CUDA_VISIBLE_DEVICES=%q MAX_TRAIN_STEPS=%q SAVE_INTERVAL=%q EVAL_INTERVAL=%q bash %q --worker' \
    "$run_dir" "$python_bin" "$config" "$run_id" "$run_root" "$port" "$CUDA_VISIBLE_DEVICES" "$max_steps" "$save_interval" "$eval_interval" "$script_path"
  tmux new-session -d -s "$session" "$command"
  echo "Attach: tmux attach -t $session"
  echo "Run directory: $run_dir"
  echo "Log: $run_dir/train.log"
  exit 0
fi
# Show live output in tmux and retain the same log on disk. pipefail preserves
# a failed training exit status instead of reporting only tee's exit status.
"$python_bin" -m accelerate.commands.launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 --main_process_port "$port" \
  starVLA/training/train_starvla.py --config_yaml "$config" \
  --trainer.max_train_steps "$max_steps" --trainer.save_interval "$save_interval" \
  --trainer.eval_interval "$eval_interval" \
  --run_root_dir "$run_root" --run_id "$run_id" \
  2>&1 | tee -a "$run_dir/train.log"
