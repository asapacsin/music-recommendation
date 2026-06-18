#!/usr/bin/env bash
# Download Meta-Llama-3.1-8B-Instruct into model/llama3.1-8b-instruct (gitignored).
#
# Prerequisites:
#   1. Hugging Face account + accept license:
#      https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
#   2. Token: huggingface-cli login   OR   export HF_TOKEN=hf_...
#
# Usage:
#   cd ~/music-recommendation
#   bash scripts/download_llama31_8b.sh
#
# Override target directory:
#   RAGWEB_LLM_MODEL_DIR=my-llama bash scripts/download_llama31_8b.sh
#
# Override repo (e.g. smaller model for dev):
#   RAGWEB_LLM_HF_REPO_ID=meta-llama/Meta-Llama-3.1-8B-Instruct bash scripts/download_llama31_8b.sh

set -euo pipefail

REPO_ROOT="${RAGWEB_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
MODEL_DIR_NAME="${RAGWEB_LLM_MODEL_DIR:-llama3.1-8b-instruct}"
HF_REPO="${RAGWEB_LLM_HF_REPO_ID:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
LOCAL_DIR="$REPO_ROOT/model/$MODEL_DIR_NAME"

echo "=== Download local LLM ==="
echo "HF repo:    $HF_REPO"
echo "Local dir:  $LOCAL_DIR"
echo ""

if [[ -f "$LOCAL_DIR/config.json" ]]; then
  echo "Already present ($LOCAL_DIR/config.json). Skipping download."
  echo "Verify with: python -m app.llm_local --check-only"
  exit 0
fi

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "ERROR: huggingface-cli not found. Install: pip install -U huggingface_hub" >&2
  exit 1
fi

mkdir -p "$LOCAL_DIR"

# Prefer hf CLI name used by recent huggingface_hub
if huggingface-cli download --help 2>&1 | grep -q local-dir; then
  huggingface-cli download "$HF_REPO" --local-dir "$LOCAL_DIR"
else
  huggingface-cli download "$HF_REPO" --cache-dir "$LOCAL_DIR" --local-dir-use-symlinks False
fi

echo ""
echo "Done. Next:"
echo "  python -m app.llm_local --check-only"
echo "  python -m app.llm_local --smoke-test"
