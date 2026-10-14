#!/usr/bin/env bash
# Export the committed EVA execution-side source without model weights or environments.
set -euo pipefail

eva_package_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
eva_package_revision="${1:-HEAD}"
cd "$eva_package_repo"
eva_package_commit="$(git rev-parse --verify "${eva_package_revision}^{commit}")"
eva_package_name="eva-starvla-client-${eva_package_commit:0:12}"
eva_package_output="${eva_package_repo}/dist/${eva_package_name}.zip"

if [[ -e "$eva_package_output" ]]; then
  echo "Package already exists: $eva_package_output" >&2
  exit 2
fi
mkdir -p "${eva_package_repo}/dist"
git archive --format=zip --prefix="${eva_package_name}/" \
  --output="$eva_package_output" "$eva_package_commit" -- \
  EVA-CLIENT examples/deployment examples/realRobots/EvaClient examples/realRobots/AgiBotG2 \
  README.md LICENSE INTEGRATION_README.md \
  docs/eva_deployment.md docs/releases/eva-deploy-20261013.md
printf 'Commit: %s\nPackage: %s\n' "$eva_package_commit" "$eva_package_output"
