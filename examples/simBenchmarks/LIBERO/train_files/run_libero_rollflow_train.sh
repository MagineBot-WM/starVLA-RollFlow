#!/usr/bin/env bash
set -euo pipefail

repo_root="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
python_bin="${STARVLA_PYTHON:-/data/miniconda3/envs/starVLA/bin/python}"
config_yaml="${CONFIG_YAML:-${repo_root}/examples/simBenchmarks/LIBERO/train_files/starvla_qwengroot_rollflow_libero_all.yaml}"
data_root="${DATA_ROOT:-/data/tzq/datasets/starVLA/Datasets/libero}"
run_root="${RUN_ROOT:-/data/tzq/starVLA_checkpoints}"
run_id="${RUN_ID:-libero_qwengroot_rollflow_h32_c8_b128_unfrozen}"
num_gpus="${NUM_GPUS:-4}"
batch_per_gpu="${BATCH_PER_GPU:-32}"
max_train_steps="${MAX_TRAIN_STEPS:-80000}"
save_interval="${SAVE_INTERVAL:-5000}"
logging_frequency="${LOGGING_FREQUENCY:-20}"

output_dir="${run_root}/${run_id}"
if [[ -e "${output_dir}" ]]; then
  echo "Refusing to overwrite existing run directory: ${output_dir}" >&2
  echo "Set RUN_ID to a new value, or explicitly move the old run directory." >&2
  exit 2
fi

cd "${repo_root}"
export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

exec "${python_bin}" -m accelerate.commands.launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_gpus}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --datasets.vla_data.data_root_dir "${data_root}" \
  --datasets.vla_data.data_mix libero_all_rollflow \
  --datasets.vla_data.per_device_batch_size "${batch_per_gpu}" \
  --trainer.freeze_modules '' \
  --trainer.max_train_steps "${max_train_steps}" \
  --trainer.save_interval "${save_interval}" \
  --trainer.logging_frequency "${logging_frequency}" \
  --run_root_dir "${run_root}" \
  --run_id "${run_id}"
