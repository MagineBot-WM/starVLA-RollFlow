#!/usr/bin/env bash
set -euo pipefail

repo_root="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
python_bin="${STARVLA_PYTHON:-/data/miniconda3/envs/starVLA/bin/python}"
checkpoint="${CHECKPOINT:?Set CHECKPOINT to a trained .pt checkpoint or checkpoint directory}"
gpu_id="${GPU_ID:-0}"
port="${PORT:-5555}"

cd "${repo_root}"
export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES="${gpu_id}" "${python_bin}" \
  deployment/model_server/server_policy_gr00t_zmq.py \
  --ckpt_path "${checkpoint}" \
  --port "${port}" \
  --unnorm_key agibot_g1 \
  --use_bf16
