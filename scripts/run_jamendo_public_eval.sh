#!/usr/bin/env bash
# Jamendo-only public eval (wrapper). Prefer scripts/run_public_eval.sh for all datasets.
set -euo pipefail
export DATASETS="${DATASETS:-jamendo}"
exec "$(cd "$(dirname "$0")" && pwd)/run_public_eval.sh" "$@"
