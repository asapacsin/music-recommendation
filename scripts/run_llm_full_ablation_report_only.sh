#!/usr/bin/env bash
# Report-only step for full LLM ablation (no eval re-run).
set -euo pipefail

REPO="${RAGWEB_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="${PYTHONPATH:-$REPO}"

ABLATION_DIR="${ABLATION_DIR:-$REPO/data/eval/llm_full_ablation}"
SEEDS="${SEEDS:-42 43 44}"
REPORT_TOP_K="${REPORT_TOP_K:-10}"
EVAL_CAPTION_INDEX="${EVAL_CAPTION_INDEX:-1}"

_SEEDS_CSV="${SEEDS// /,}"
_INDEX_KINDS="meta"
if [[ "$EVAL_CAPTION_INDEX" == "1" ]]; then
  _INDEX_KINDS="meta,caption"
fi

python -m app.data_handling.music_eval_llm_full_ablation_report \
  --ablation-dir "$ABLATION_DIR" \
  --seeds "$_SEEDS_CSV" \
  --top-k "$REPORT_TOP_K" \
  --index-kinds "$_INDEX_KINDS"

echo "Report: $ABLATION_DIR/REPORT.md"
