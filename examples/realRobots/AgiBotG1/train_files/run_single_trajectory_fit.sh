#!/usr/bin/env bash
set -euo pipefail

# Hardware smoke test: fit one Apple trajectory in memory and report whether
# the same clean 55K model retains its LIBERO loss.  No checkpoint or dataset
# is modified.  Override STEPS/WINDOWS/GPU_ID for a longer or smaller test.
repo_root="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
python_bin="${STARVLA_PYTHON:-/data/miniconda3/envs/starVLA/bin/python}"
gpu_id="${GPU_ID:-2}"
steps="${STEPS:-200}"
windows="${WINDOWS:-4}"
output="${OUTPUT:-/data/tzq/starVLA_checkpoints/agibot_g1_single_trajectory_fit.json}"

cd "${repo_root}"
export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CUDA_VISIBLE_DEVICES="${gpu_id}" "${python_bin}" \
  examples/realRobots/AgiBotG1/train_files/fit_single_trajectory.py \
  --steps "${steps}" \
  --windows "${windows}" \
  --output "${output}"
