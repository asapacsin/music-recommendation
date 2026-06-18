#!/usr/bin/env bash
# Run public-eval dataset downloaders (Jamendo / MTAT / OpenMIC).
# Prefer: bash scripts/run_public_eval_download.sh <mtat|openmic|all>
# Logs: data/log/public_eval_downloads/*.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${RAGWEB_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export RAGWEB_REPO="$REPO"

echo "=== public_eval backend driver $(date -u +"%Y-%m-%dT%H:%M:%SZ") ==="
echo "REPO=$REPO"

if [[ "${SKIP_JAMENDO:-0}" != "1" ]]; then
  bash "$SCRIPT_DIR/download_jamendo_five_tag_backend.sh"
else
  echo "SKIP_JAMENDO=1"
fi

if [[ "${SKIP_MTAT:-0}" != "1" ]]; then
  bash "$SCRIPT_DIR/download_mtat_backend.sh"
else
  echo "SKIP_MTAT=1"
fi

if [[ "${SKIP_OPENMIC:-0}" != "1" ]]; then
  bash "$SCRIPT_DIR/download_openmic_backend.sh"
else
  echo "SKIP_OPENMIC=1"
fi

echo "=== All requested backend downloads finished ==="
echo "Progress: bash scripts/refresh_progress.sh  → docs/PROGRESS.md (Public OOD section)"
