#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=eval_config.sh
source "${SCRIPT_DIR}/eval_config.sh"
STARVLA_DIR="${STARVLA_DIR:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"

[[ -n "${CKPT}" ]] || { echo "CKPT is empty; set it in eval_config.sh." >&2; exit 1; }

export PYTHONPATH="${STARVLA_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export NO_ALBUMENTATIONS_UPDATE=1
CMD=(
  "${STARVLA_PYTHON}" "${STARVLA_DIR}/deployment/model_server/server_policy.py"
  --ckpt_path "${CKPT}"
  --port "${PORT}"
)
[[ "${USE_BF16}" == "1" ]] && CMD+=(--use_bf16)

if [[ -n "${USE_CANONICAL_FORWARD:-}" ]]; then
  [[ "${USE_CANONICAL_FORWARD}" =~ ^(true|false)$ ]] || {
    echo "USE_CANONICAL_FORWARD must be 'true' or 'false'." >&2
    exit 1
  }
  OVERRIDE="framework.action_model.diffusion_model_cfg.use_canonical_forward=${USE_CANONICAL_FORWARD}"
  echo "Applying config override: ${OVERRIDE}"
  CMD+=(--config_override "${OVERRIDE}")
fi

echo "Serving ${CKPT} on GPU ${GPU_ID}, port ${PORT}"
cd "${STARVLA_DIR}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" DEBUG="${STARVLA_DEBUG:-}" "${CMD[@]}"
