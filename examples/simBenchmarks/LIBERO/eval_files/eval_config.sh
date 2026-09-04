# Shared defaults for LIBERO policy serving and evaluation.
#
# Normally this is the only line you need to change. All three launchers source
# this file, so the policy server and evaluator use exactly the same checkpoint.
# A one-off shell override also works: CKPT=/path/to/model.pt bash <script>.

CKPT="${CKPT:-/data/tzq/starVLA_checkpoints/libero_qwengroot_rollflow_h32_c8_b128_unfrozen_463a1f7/checkpoints/steps_10000_pytorch_model.pt}"

STARVLA_PYTHON="${STARVLA_PYTHON:-/data/miniconda3/envs/starVLA/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/data/miniconda3/envs/lerobot/bin/python}"
LIBERO_HOME="${LIBERO_HOME:-/home/taizun/tzq/LIBERO}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-6694}"
GPU_ID="${GPU_ID:-0}"
EVAL_GPU_ID="${EVAL_GPU_ID:-0}"
USE_BF16="${USE_BF16:-1}"

TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_goal}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
MAX_TASKS="${MAX_TASKS:--1}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
SEED="${SEED:-7}"
UNNORM_KEY="${UNNORM_KEY:-franka}"
