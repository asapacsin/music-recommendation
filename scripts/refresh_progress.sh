#!/usr/bin/env bash
# Refresh docs/PROGRESS.md and data/eval/progress_snapshot.json from on-disk artifacts.
set -euo pipefail

REPO="${RAGWEB_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="${PYTHONPATH:-$REPO}"

python -m app.progress_monitor "$@"
echo ""
echo "Markdown: docs/PROGRESS.md"
echo "JSON:     data/eval/progress_snapshot.json"
