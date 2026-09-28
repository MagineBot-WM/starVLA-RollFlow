#!/usr/bin/env bash
set -euo pipefail

repo_root="/data/tzq/starVLA-RollFlow"
python_bin="/data/miniconda3/envs/starVLA/bin/python"
config_yaml="${repo_root}/examples/realRobots/0923G1_RollFlow/train_files/starvla_g1_0923_rollingflow_h128_30k.yaml"
run_id="g1_0923_pure_rolling_flow_h128_c32_scratch_30k_20260928"
run_root="/data/tzq/starVLA_checkpoints"
log_file="/data/tzq/tmp/${run_id}.log"

cd "${repo_root}"
export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"
export WANDB_MODE="disabled"

if [[ -e "${run_root}/${run_id}" ]]; then
  echo "Refusing to overwrite existing run: ${run_root}/${run_id}" >&2
  exit 3
fi

"${python_bin}" -m accelerate.commands.launch \
  --config_file starVLA/config/deepseeds/zero2.yaml \
  --num_processes 4 \
  --gradient_accumulation_steps 2 \
  --main_process_port 29661 \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name QwenGR00TRollingFlow \
  --trainer.max_train_steps 30000 \
  --trainer.num_warmup_steps 5000 \
  --trainer.lr_scheduler_type cosine_with_min_lr \
  --trainer.save_interval 1000 \
  --trainer.eval_interval 1000 \
  --trainer.logging_frequency 20 \
  --trainer.gradient_accumulation_steps 2 \
  --trainer.scheduler_specific_kwargs.min_lr 5.0e-07 \
  --datasets.vla_data.data_mix g1_0923_20d_all_h128 \
  --datasets.vla_data.data_root_dir /data/tzq/starVLA_datasets/0923_20s_20a_rollflow_h64 \
  --datasets.vla_data.per_device_batch_size 16 \
  --datasets.vla_data.num_workers 4 \
  --run_root_dir "${run_root}" \
  --run_id "${run_id}" \
  2>&1 | tee "${log_file}"
