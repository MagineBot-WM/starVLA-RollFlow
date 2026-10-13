#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
python_bin="${STARVLA_MODEL_PYTHON:-python}"
config_yaml="${STARVLA_G2_TRAIN_CONFIG:-${repo_root}/examples/realRobots/AgiBotG2/train_files/starvla_agibot_g2_rollingflow_h64_b64_30k.yaml}"
run_id="${STARVLA_G2_RUN_ID:-agibot_g2_rollingflow_h64_c16_b64_mb4_ga4_scratch_30k_20261010}"
run_root="${STARVLA_RUN_ROOT:-${repo_root}/checkpoints}"
log_root="${STARVLA_LOG_ROOT:-${repo_root}/work_dirs/training_logs}"
log_file="${log_root}/${run_id}.log"
minimum_free_memory_mib=14000
required_memory_samples=3

cd "${repo_root}"
export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"
export WANDB_MODE="disabled"
mkdir -p "${log_root}"

if [[ ! -f "${config_yaml}" ]]; then
  echo "Missing G2 training config: ${config_yaml}" >&2
  exit 2
fi
if [[ -e "${run_root}/${run_id}" ]]; then
  echo "Refusing to overwrite existing run: ${run_root}/${run_id}" >&2
  exit 3
fi

gpu_has_room() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
    | awk -v min_free="${minimum_free_memory_mib}" '
      NF == 1 {
        gsub(/[[:space:]]/, "", $1)
        if (($1 + 0) < min_free) busy = 1
        seen++
      }
      END { exit (seen == 4 && !busy) ? 0 : 1 }
    '
}

memory_samples=0
echo "Waiting for at least ${minimum_free_memory_mib} MiB free on each of four GPUs; utilization may remain nonzero."
while (( memory_samples < required_memory_samples )); do
  if gpu_has_room; then
    ((memory_samples += 1))
    echo "GPU memory check ${memory_samples}/${required_memory_samples} passed."
  else
    memory_samples=0
    echo "Insufficient free memory on at least one GPU; no training process was started. Rechecking in 60 seconds."
  fi
  if (( memory_samples < required_memory_samples )); then sleep 60; fi
done

if [[ -e "${run_root}/${run_id}" ]]; then
  echo "Refusing to overwrite run directory created while waiting: ${run_root}/${run_id}" >&2
  exit 3
fi

exec "${python_bin}" -m accelerate.commands.launch \
  --config_file examples/realRobots/AgiBotG2/train_files/accelerate_zero2.yaml \
  --num_processes 4 \
  --gradient_accumulation_steps 4 \
  --main_process_port 29877 \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --run_root_dir "${run_root}" \
  --run_id "${run_id}" \
  2>&1 | tee "${log_file}"
