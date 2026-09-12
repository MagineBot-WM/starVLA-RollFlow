#!/usr/bin/env bash
set -euo pipefail

script_path="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
repo_root="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
python_bin="${STARVLA_PYTHON:-/data/miniconda3/envs/starVLA/bin/python}"
config_yaml="${CONFIG_YAML:-${repo_root}/examples/simBenchmarks/LIBERO/train_files/starvla_qwengroot_rollflow_libero_all.yaml}"
data_root="${DATA_ROOT:-/data/tzq/datasets/starVLA/Datasets/libero}"
data_mix="${DATA_MIX:-libero_all_rollflow}"
run_root="${RUN_ROOT:-/data/tzq/starVLA_checkpoints}"
run_id="${RUN_ID:-libero_qwengroot_rollflow_h32_c8_b128_unfrozen}"
num_gpus="${NUM_GPUS:-4}"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
main_process_port="${MAIN_PROCESS_PORT:-29501}"
batch_per_gpu="${BATCH_PER_GPU:-16}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-2}"
max_train_steps="${MAX_TRAIN_STEPS:-80000}"
save_interval="${SAVE_INTERVAL:-5000}"
eval_interval="${EVAL_INTERVAL:-1000}"
logging_frequency="${LOGGING_FREQUENCY:-20}"
tmux_session="${TMUX_SESSION:-rollflow_libero}"
wait_for_gpu_free="${WAIT_FOR_GPU_FREE:-1}"
resume="${RESUME:-0}"
pytorch_cuda_alloc_conf="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

output_dir="${run_root}/${run_id}"

require_positive_integer() {
  local name="$1" value="$2"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive integer, got: ${value}" >&2
    exit 2
  fi
}

for pair in \
  "NUM_GPUS:${num_gpus}" \
  "MAIN_PROCESS_PORT:${main_process_port}" \
  "BATCH_PER_GPU:${batch_per_gpu}" \
  "GRADIENT_ACCUMULATION_STEPS:${gradient_accumulation_steps}" \
  "MAX_TRAIN_STEPS:${max_train_steps}" \
  "SAVE_INTERVAL:${save_interval}" \
  "EVAL_INTERVAL:${eval_interval}" \
  "LOGGING_FREQUENCY:${logging_frequency}"; do
  require_positive_integer "${pair%%:*}" "${pair#*:}"
done
if (( main_process_port > 65535 )); then
  echo "MAIN_PROCESS_PORT must be at most 65535, got: ${main_process_port}" >&2
  exit 2
fi
if [[ "${wait_for_gpu_free}" != "0" && "${wait_for_gpu_free}" != "1" ]]; then
  echo "WAIT_FOR_GPU_FREE must be 0 or 1, got: ${wait_for_gpu_free}" >&2
  exit 2
fi
if [[ "${resume}" != "0" && "${resume}" != "1" ]]; then
  echo "RESUME must be 0 or 1, got: ${resume}" >&2
  exit 2
fi
if [[ ! -x "${python_bin}" ]]; then
  echo "Python executable not found: ${python_bin}" >&2
  exit 2
fi
if [[ ! -f "${config_yaml}" ]]; then
  echo "Config file not found: ${config_yaml}" >&2
  exit 2
fi
if [[ ! -d "${data_root}" ]]; then
  echo "Dataset directory not found: ${data_root}" >&2
  exit 2
fi

IFS=',' read -r -a visible_devices <<< "${cuda_visible_devices}"
if (( num_gpus > ${#visible_devices[@]} )); then
  echo "NUM_GPUS=${num_gpus} exceeds CUDA_VISIBLE_DEVICES=${cuda_visible_devices}" >&2
  exit 2
fi

# The parent validates fresh-run collisions before creating the detached
# worker.  The worker itself may see the pre-created run directory because its
# log is stored next to the checkpoints.
if [[ "${1:-}" != "--worker" && "${resume}" == "0" && -e "${output_dir}" ]]; then
  echo "Refusing to overwrite existing run directory: ${output_dir}" >&2
  echo "Set RUN_ID to a new value, or set RESUME=1 to load its latest checkpoint." >&2
  exit 2
fi
if [[ "${resume}" == "1" && ! -d "${output_dir}" ]]; then
  echo "Cannot resume missing run directory: ${output_dir}" >&2
  exit 2
fi
if [[ "${resume}" == "1" ]] \
  && ! compgen -G "${output_dir}/checkpoints/steps_*_pytorch_model.pt" >/dev/null \
  && ! compgen -G "${output_dir}/checkpoints/steps_*_model.safetensors" >/dev/null; then
  echo "No resumable checkpoint found in: ${output_dir}/checkpoints" >&2
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

  mkdir -p "${run_root}"
  log_file="${output_dir}/train.log"
  printf -v worker_cmd \
    'cd %q && mkdir -p %q && exec env STARVLA_DIR=%q STARVLA_PYTHON=%q CONFIG_YAML=%q DATA_ROOT=%q DATA_MIX=%q RUN_ROOT=%q RUN_ID=%q NUM_GPUS=%q CUDA_VISIBLE_DEVICES=%q MAIN_PROCESS_PORT=%q BATCH_PER_GPU=%q GRADIENT_ACCUMULATION_STEPS=%q MAX_TRAIN_STEPS=%q SAVE_INTERVAL=%q EVAL_INTERVAL=%q LOGGING_FREQUENCY=%q WAIT_FOR_GPU_FREE=%q RESUME=%q PYTORCH_CUDA_ALLOC_CONF=%q PYTHONUNBUFFERED=1 NO_ALBUMENTATIONS_UPDATE=1 WANDB_MODE=disabled %q --worker >> %q 2>&1' \
    "${repo_root}" "${output_dir}" "${repo_root}" "${python_bin}" "${config_yaml}" \
    "${data_root}" "${data_mix}" "${run_root}" "${run_id}" "${num_gpus}" \
    "${cuda_visible_devices}" "${main_process_port}" "${batch_per_gpu}" \
    "${gradient_accumulation_steps}" "${max_train_steps}" "${save_interval}" "${eval_interval}" \
    "${logging_frequency}" "${wait_for_gpu_free}" "${resume}" \
    "${pytorch_cuda_alloc_conf}" "${script_path}" "${log_file}"

  tmux new-session -d -s "${tmux_session}" "${worker_cmd}"
  echo "Started tmux session: ${tmux_session}"
  echo "Run directory: ${output_dir}"
  echo "Training log: ${log_file}"
  echo "GPUs: ${cuda_visible_devices} (world size ${num_gpus})"
  echo "Effective batch: ${num_gpus} x ${batch_per_gpu} x ${gradient_accumulation_steps} = $((num_gpus * batch_per_gpu * gradient_accumulation_steps))"
  [[ "${resume}" == "1" ]] && echo "Resume: latest checkpoint in ${output_dir}/checkpoints"
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

if command -v ss >/dev/null 2>&1 && ss -H -ltn | awk '{print $4}' | grep -Eq "[:.]${main_process_port}$"; then
  echo "Distributed port ${main_process_port} is already in use." >&2
  exit 5
fi

cd "${repo_root}"
mkdir -p "${output_dir}"
cp "${script_path}" "${output_dir}/train_launcher.sh"
export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export NO_ALBUMENTATIONS_UPDATE=1
export PYTORCH_CUDA_ALLOC_CONF="${pytorch_cuda_alloc_conf}"

exec "${python_bin}" -m accelerate.commands.launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_gpus}" \
  --main_process_port "${main_process_port}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --datasets.vla_data.data_root_dir "${data_root}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.per_device_batch_size "${batch_per_gpu}" \
  --trainer.gradient_accumulation_steps "${gradient_accumulation_steps}" \
  --trainer.freeze_modules '' \
  --trainer.max_train_steps "${max_train_steps}" \
  --trainer.save_interval "${save_interval}" \
  --trainer.eval_interval "${eval_interval}" \
  --trainer.logging_frequency "${logging_frequency}" \
  --trainer.is_resume "${resume}" \
  --run_root_dir "${run_root}" \
  --run_id "${run_id}"
