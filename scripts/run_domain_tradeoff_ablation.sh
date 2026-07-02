#!/usr/bin/env bash
# Domain tradeoff — Question E (Grok captions on ACG, anime-only vs mixed FT):
#   - build clap_train_grok_mixed.jsonl (Grok anime + MTAT + OpenMIC)
#   - audio cache -> fine-tune thesis_grok_only + thesis_grok_mixed (seeds 42-44)
#   - in-domain gold + public OOD eval -> 2×2 report
#
# Usage:
#   sbatch scripts/sbatch_domain_tradeoff_ablation.sh
#
# Eval / report only (checkpoints exist):
#   SKIP_BUILD=1 SKIP_CACHE=1 SKIP_TRAIN=1 bash scripts/run_domain_tradeoff_ablation.sh

set -euo pipefail

REPO="${RAGWEB_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
export REPO
export PYTHONPATH="${PYTHONPATH:-$REPO}"

TRADE_DIR="${TRADE_DIR:-$REPO/data/eval/domain_tradeoff}"
INDEX_DIR="${INDEX_DIR:-$TRADE_DIR/index}"

ANIME_JSONL="${ANIME_JSONL:-$REPO/data/mapping/clap_train_15s.jsonl}"
MIXED_JSONL="${MIXED_JSONL:-$REPO/data/mapping/clap_train_grok_mixed.jsonl}"
HOLDOUT_TXT="${HOLDOUT_TXT:-$REPO/data/mapping/public_eval_holdout_paths.txt}"
TRAIN_PARAMS="${TRAIN_PARAMS:-$REPO/data/eval/llm_ablation/train_params.json}"
VAL_JSONL="${VAL_JSONL:-$REPO/data/mapping/clap_val_15s.jsonl}"
METADATA_JSON="${METADATA_JSON:-$REPO/data/mapping/music_metadata.json}"

RUN_ID_ANIME="${RUN_ID_ANIME:-thesis_grok_only}"
RUN_ID_MIXED="${RUN_ID_MIXED:-thesis_grok_mixed}"
SEEDS="${SEEDS:-42 43 44}"
TOP_K="${TOP_K:-10}"
MIN_CONFIDENCE="${MIN_CONFIDENCE:-0.35}"
MIX_RATIO="${MIX_RATIO:-0.5}"
PUBLIC_CLIP_TARGET="${PUBLIC_CLIP_TARGET:-0}"

SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_CACHE="${SKIP_CACHE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_GOLD_EVAL="${SKIP_GOLD_EVAL:-0}"
SKIP_PUBLIC_EVAL="${SKIP_PUBLIC_EVAL:-0}"
SKIP_REPORT="${SKIP_REPORT:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
NO_AUDIO_CACHE="${NO_AUDIO_CACHE:-0}"

DATASETS="${DATASETS:-jamendo mtat openmic}"
PUBLIC_ARMS="${PUBLIC_ARMS:-pretrained thesis_grok_only thesis_grok_mixed}"

mkdir -p "$TRADE_DIR" "$INDEX_DIR"

if [[ ! -f "$TRAIN_PARAMS" ]]; then
  echo "ERROR: train params missing: $TRAIN_PARAMS" >&2
  exit 1
fi
if [[ ! -f "$ANIME_JSONL" ]]; then
  echo "ERROR: anime train JSONL missing: $ANIME_JSONL" >&2
  exit 1
fi

_SEEDS_CSV="${SEEDS// /,}"
_MULTISEED_FLAGS=()
if [[ "$SKIP_EXISTING" == "0" ]]; then
  _MULTISEED_FLAGS+=(--no-skip-existing)
fi
if [[ "$NO_AUDIO_CACHE" == "1" ]]; then
  _MULTISEED_FLAGS+=(--no-audio-cache)
fi

_resolve_ckpt() {
  local run_id="$1"
  local seed="$2"
  local ckpt
  for ckpt in \
    "$REPO/model/clap/finetune/${run_id}/seed_${seed}/best_model.pt" \
    "$REPO/data/log/finetune_runs/${run_id}/seed_${seed}/best_model.pt"; do
    if [[ -f "$ckpt" ]]; then
      echo "$ckpt"
      return 0
    fi
  done
  echo "ERROR: checkpoint missing run_id=$run_id seed=$seed" >&2
  return 1
}

_gold_eval_arm() {
  local run_id="$1"
  local prefix="$2"
  echo "=== In-domain gold eval ($run_id) ==="
  for seed in $SEEDS; do
    ckpt="$(_resolve_ckpt "$run_id" "$seed")"
    export RAGWEB_CLAP_CHECKPOINT="$ckpt"
    meta_index="${INDEX_DIR}/${prefix}_meta_seed${seed}.faiss"
    meta_mapping="${INDEX_DIR}/${prefix}_meta_seed${seed}.mapping.json"
    echo "  seed $seed — build metadata FAISS"
    python -m app.data_handling.music_build_retrieval_faiss build-metadata \
      --metadata "$METADATA_JSON" \
      --out-index "$meta_index" \
      --out-mapping "$meta_mapping" \
      --min-confidence "$MIN_CONFIDENCE"
    python -m app.data_handling.music_eval_retrieval_vs_random \
      --top-k "$TOP_K" \
      --index "$meta_index" \
      --mapping "$meta_mapping" \
      --out-csv "$TRADE_DIR/${prefix}_gold_seed${seed}.csv" \
      --out-json "$TRADE_DIR/${prefix}_gold_seed${seed}.json"
  done
  unset RAGWEB_CLAP_CHECKPOINT || true
}

# --- 0. Build mixed JSONL (Grok anime rows) ---
if [[ "$SKIP_BUILD" != "1" ]]; then
  echo "=== Build Grok mixed-domain train JSONL ==="
  _BUILD_FLAGS=(
    --anime-jsonl "$ANIME_JSONL"
    --out-jsonl "$MIXED_JSONL"
    --holdout-txt "$HOLDOUT_TXT"
    --mix-ratio "$MIX_RATIO"
  )
  if [[ "$PUBLIC_CLIP_TARGET" -gt 0 ]]; then
    _BUILD_FLAGS+=(--public-clip-target "$PUBLIC_CLIP_TARGET")
  fi
  python -m app.data_handling.music_build_mixed_domain_train_jsonl "${_BUILD_FLAGS[@]}"
else
  echo "SKIP_BUILD=1"
  if [[ ! -f "$MIXED_JSONL" ]]; then
    echo "ERROR: mixed JSONL missing: $MIXED_JSONL" >&2
    exit 1
  fi
fi

# --- 1. Audio cache ---
if [[ "$SKIP_CACHE" != "1" ]]; then
  echo "=== Precompute CLAP audio cache (anime + mixed + val) ==="
  python -m app.data_handling.music_precompute_clap_audio_cache \
    --jsonl "$ANIME_JSONL" \
    --jsonl "$MIXED_JSONL" \
    --jsonl "$VAL_JSONL"
else
  echo "SKIP_CACHE=1"
fi

# --- 2. Train both arms ---
if [[ "$SKIP_TRAIN" != "1" ]]; then
  echo "=== Fine-tune $RUN_ID_ANIME (Grok anime-only) ==="
  _train_anime=(
    python -m app.train_clap_multiseed
    --run-id "$RUN_ID_ANIME"
    --seeds "$_SEEDS_CSV"
    --train-jsonl "$ANIME_JSONL"
    --params-json "$TRAIN_PARAMS"
  )
  if ((${#_MULTISEED_FLAGS[@]})); then
    _train_anime+=("${_MULTISEED_FLAGS[@]}")
  fi
  "${_train_anime[@]}"

  echo "=== Fine-tune $RUN_ID_MIXED (Grok anime + public) ==="
  _train_mixed=(
    python -m app.train_clap_multiseed
    --run-id "$RUN_ID_MIXED"
    --seeds "$_SEEDS_CSV"
    --train-jsonl "$MIXED_JSONL"
    --params-json "$TRAIN_PARAMS"
  )
  if ((${#_MULTISEED_FLAGS[@]})); then
    _train_mixed+=("${_MULTISEED_FLAGS[@]}")
  fi
  "${_train_mixed[@]}"
else
  echo "SKIP_TRAIN=1"
fi

# --- 3. In-domain gold eval (both arms) ---
if [[ "$SKIP_GOLD_EVAL" != "1" ]]; then
  _gold_eval_arm "$RUN_ID_ANIME" "anime_only"
  _gold_eval_arm "$RUN_ID_MIXED" "mixed"
else
  echo "SKIP_GOLD_EVAL=1"
fi

# --- 4. Public OOD eval ---
if [[ "$SKIP_PUBLIC_EVAL" != "1" ]]; then
  echo "=== Public OOD eval ==="
  ARMS="$PUBLIC_ARMS" \
    DATASETS="$DATASETS" \
    SEEDS="$SEEDS" \
    TOP_K="$TOP_K" \
    SKIP_EXISTING="$SKIP_EXISTING" \
    RUN_REPORT=0 \
    bash "$REPO/scripts/run_public_eval.sh"
else
  echo "SKIP_PUBLIC_EVAL=1"
fi

# --- 5. 2×2 report ---
if [[ "$SKIP_REPORT" != "1" ]]; then
  echo "=== Domain tradeoff 2×2 report ==="
  python -m app.data_handling.music_eval_domain_tradeoff_report \
    --trade-dir "$TRADE_DIR" \
    --eval-root "$REPO/data/eval" \
    --datasets "${DATASETS// /,}" \
    --seeds "$_SEEDS_CSV" \
    --top-k "$TOP_K" \
    --anime-arm "$RUN_ID_ANIME" \
    --mixed-arm "$RUN_ID_MIXED"
fi

echo "Done."
echo "  Report: $TRADE_DIR/REPORT.md"
echo "  Summary: $TRADE_DIR/summary.json"
