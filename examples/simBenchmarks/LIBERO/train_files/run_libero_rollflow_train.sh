#!/usr/bin/env bash
set -euo pipefail

script_path="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
repo_root="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
python_bin="${STARVLA_PYTHON:-/data/miniconda3/envs/starVLA/bin/python}"
config_yaml="${CONFIG_YAML:-${repo_root}/examples/simBenchmarks/LIBERO/train_files/starvla_qwengroot_rollflow_libero_all.yaml}"
data_root="${DATA_ROOT:-/data/tzq/datasets/starVLA/Datasets/libero}"
run_root="${RUN_ROOT:-/data/tzq/starVLA_checkpoints}"
run_id="${RUN_ID:-libero_qwengroot_rollflow_h32_c8_b128_unfrozen}"
num_gpus="${NUM_GPUS:-4}"
main_process_port="${MAIN_PROCESS_PORT:-29501}"
batch_per_gpu="${BATCH_PER_GPU:-32}"
max_train_steps="${MAX_TRAIN_STEPS:-80000}"
save_interval="${SAVE_INTERVAL:-5000}"
logging_frequency="${LOGGING_FREQUENCY:-20}"
tmux_session="${TMUX_SESSION:-rollflow_libero}"
wait_for_gpu_free="${WAIT_FOR_GPU_FREE:-1}"

output_dir="${run_root}/${run_id}"
if [[ -e "${output_dir}" ]]; then
  echo "Refusing to overwrite existing run directory: ${output_dir}" >&2
  echo "Set RUN_ID to a new value, or explicitly move the old run directory." >&2
  exit 2
fi

if [[ "${1:-}" != "--worker" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is required but was not found." >&2
    exit 3
  fi
  if tmux has-session -t "${tmux_session}" 2>/dev/null; then
    echo "Refusing to replace existing tmux session: ${tmux_session}" >&2
    exit 4
  fi

  log_file="${run_root}/${run_id}.train.log"
  printf -v worker_cmd \
    'cd %q && exec env STARVLA_DIR=%q STARVLA_PYTHON=%q CONFIG_YAML=%q DATA_ROOT=%q RUN_ROOT=%q RUN_ID=%q NUM_GPUS=%q MAIN_PROCESS_PORT=%q BATCH_PER_GPU=%q MAX_TRAIN_STEPS=%q SAVE_INTERVAL=%q LOGGING_FREQUENCY=%q WAIT_FOR_GPU_FREE=%q WANDB_MODE=disabled %q --worker >> %q 2>&1' \
    "${repo_root}" "${repo_root}" "${python_bin}" "${config_yaml}" \
    "${data_root}" "${run_root}" "${run_id}" "${num_gpus}" \
    "${main_process_port}" "${batch_per_gpu}" "${max_train_steps}" "${save_interval}" \
    "${logging_frequency}" "${wait_for_gpu_free}" "${script_path}" "${log_file}"

  tmux new-session -d -s "${tmux_session}" "${worker_cmd}"
  echo "Started tmux session: ${tmux_session}"
  echo "Run directory: ${output_dir}"
  echo "Training log: ${log_file}"
  echo "Attach with: tmux attach -t ${tmux_session}"
  exit 0
fi
shift

if [[ "${wait_for_gpu_free}" == "1" ]]; then
  while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')" ]]; do
    echo "$(date '+%F %T') Waiting for all GPUs to become free..."
    sleep 30
  done
fi

cd "${repo_root}"
export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"
export WANDB_MODE=disabled

exec "${python_bin}" -m accelerate.commands.launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_gpus}" \
  --main_process_port "${main_process_port}" \
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
