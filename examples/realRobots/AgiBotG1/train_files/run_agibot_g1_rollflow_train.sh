#!/usr/bin/env bash
set -euo pipefail

repo_root="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
python_bin="${STARVLA_PYTHON:-/data/miniconda3/envs/starVLA/bin/python}"
config_yaml="${CONFIG_YAML:-${repo_root}/examples/realRobots/AgiBotG1/train_files/starvla_qwengroot_rollflow_agibot_g1.yaml}"
data_root="${DATA_ROOT:-/data/tzq/datasets/starVLA/Datasets}"
run_root="${RUN_ROOT:-/data/tzq/starVLA_checkpoints}"
run_id="${RUN_ID:-agibot_g1_qwengroot_rollflow_h32_c8_b128}"
num_gpus="${NUM_GPUS:-4}"
batch_per_gpu="${BATCH_PER_GPU:-32}"
data_mix="${DATA_MIX:-agibot_g1_all}"
max_train_steps="${MAX_TRAIN_STEPS:-100000}"
save_interval="${SAVE_INTERVAL:-5000}"

cd "${repo_root}"
export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"

"${python_bin}" examples/realRobots/AgiBotG1/dataset_tools/prepare_overlays.py \
  --datasets-root "${data_root}"
"${python_bin}" examples/realRobots/AgiBotG1/dataset_tools/audit_datasets.py \
  --datasets-root "${data_root}" \
  --output "${data_root}/AgiBot-G1-G2-StarVLA/g1/audit_agibot_g1.json"

"${python_bin}" -m accelerate.commands.launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_gpus}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --datasets.vla_data.data_root_dir "${data_root}/AgiBot-G1-G2-StarVLA" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.per_device_batch_size "${batch_per_gpu}" \
  --trainer.freeze_modules qwen_vl_interface \
  --trainer.max_train_steps "${max_train_steps}" \
  --trainer.save_interval "${save_interval}" \
  --run_root_dir "${run_root}" \
  --run_id "${run_id}"
