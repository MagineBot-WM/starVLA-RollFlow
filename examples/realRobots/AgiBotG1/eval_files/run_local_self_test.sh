#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
python_bin="${STARVLA_PYTHON:-/data/miniconda3/envs/starVLA/bin/python}"
"${python_bin}" "${script_dir}/local_self_test.py"
