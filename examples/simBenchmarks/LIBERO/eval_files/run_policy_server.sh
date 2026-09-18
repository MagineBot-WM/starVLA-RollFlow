#!/usr/bin/env bash
# Start the StarVLA websocket policy server used by LIBERO evaluation.
#
# This script runs in the StarVLA environment.  In a second terminal, run
# eval_libero.sh from the LIBERO environment.  Keep CKPT, image/state inputs,
# and action normalization consistent with the training configuration.

set -euo pipefail

# ------------------------------ User configuration -------------------------
STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
CKPT="${CKPT:-${STARVLA_DIR}/playground/Checkpoints/libero_example/checkpoints/steps_50000_pytorch_model.pt}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-6694}"
USE_BF16="${USE_BF16:-1}"
IDLE_TIMEOUT="${IDLE_TIMEOUT:-1800}"

if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  echo "Set CKPT to a steps_*_pytorch_model.pt file." >&2
  exit 1
fi
if ! command -v "${STARVLA_PYTHON}" >/dev/null 2>&1; then
  echo "STARVLA_PYTHON was not found: ${STARVLA_PYTHON}" >&2
  exit 1
fi
if [[ ! "${PORT}" =~ ^[1-9][0-9]*$ ]] || (( PORT > 65535 )); then
  echo "PORT must be an integer in [1, 65535], got: ${PORT}" >&2
  exit 1
fi
if [[ -n "${USE_BF16}" && "${USE_BF16}" != "0" && "${USE_BF16}" != "1" ]]; then
  echo "USE_BF16 must be 0 or 1, got: ${USE_BF16}" >&2
  exit 1
fi
if [[ ! "${IDLE_TIMEOUT}" =~ ^-?[0-9]+$ ]]; then
  echo "IDLE_TIMEOUT must be an integer in seconds, got: ${IDLE_TIMEOUT}" >&2
  exit 1
fi

cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

# Build an array so paths and optional overrides remain shell-safe.
cmd=(
  "${STARVLA_PYTHON}" deployment/model_server/server_policy.py
  --ckpt_path "${CKPT}"
  --port "${PORT}"
  --idle_timeout "${IDLE_TIMEOUT}"
)
if [[ "${USE_BF16}" == "1" ]]; then
  cmd+=(--use_bf16)
fi

# Optional compatibility/config override, e.g. USE_CANONICAL_FORWARD=false.
if [[ -n "${USE_CANONICAL_FORWARD:-}" ]]; then
  if [[ "${USE_CANONICAL_FORWARD}" != "true" && "${USE_CANONICAL_FORWARD}" != "false" ]]; then
    echo "USE_CANONICAL_FORWARD must be 'true' or 'false'; got '${USE_CANONICAL_FORWARD}'." >&2
    exit 2
  fi
  override="framework.action_model.diffusion_model_cfg.use_canonical_forward=${USE_CANONICAL_FORWARD}"
  echo "Applying config override: ${override}"
  cmd+=(--config_override "${override}")
fi

echo "Starting policy server: ckpt=${CKPT} GPU=${GPU_ID} port=${PORT} bf16=${USE_BF16}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}"
