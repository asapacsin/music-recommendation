#!/usr/bin/env bash
# Tag-only vs tag→LLM training ablation:
#   - build tag train JSONL (gold join + fallback)
#   - song-level LLM expand tag strings
#   - audio cache -> fine-tune both arms -> symmetric eval -> report
#
# Usage:
#   python -m app.data_handling.music_build_tag_train_jsonl
#   sbatch scripts/sbatch_tag_llm_corpus_gen.sh
#   SKIP_LLM_GEN=1 sbatch scripts/sbatch_tag_llm_ablation.sh
#
# Eval only:
#   SKIP_BUILD=1 SKIP_LLM_GEN=1 SKIP_MERGE=1 SKIP_CACHE=1 SKIP_TRAIN=1 \
#     bash scripts/run_tag_llm_ablation.sh

set -euo pipefail

REPO="${RAGWEB_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
export REPO
export PYTHONPATH="${PYTHONPATH:-$REPO}"

ABLATION_DIR="${ABLATION_DIR:-$REPO/data/eval/tag_llm_ablation}"
INDEX_DIR="${INDEX_DIR:-$ABLATION_DIR/index}"
BASE_TRAIN_JSONL="${BASE_TRAIN_JSONL:-$REPO/data/mapping/clap_train_15s.jsonl}"
GOLD_JSONL="${GOLD_JSONL:-$REPO/data/eval/gold_merged.jsonl}"
TAG_TRAIN_JSONL="${TAG_TRAIN_JSONL:-$REPO/data/mapping/clap_train_tag.jsonl}"
TAG_LLM_SONGS_JSONL="${TAG_LLM_SONGS_JSONL:-$REPO/data/mapping/clap_train_tag_llm_songs.jsonl}"
TAG_LLM_TRAIN_JSONL="${TAG_LLM_TRAIN_JSONL:-$REPO/data/mapping/clap_train_tag_llm.jsonl}"
TRAIN_PARAMS="${TRAIN_PARAMS:-$REPO/data/eval/llm_ablation/train_params.json}"
VAL_JSONL="${VAL_JSONL:-$REPO/data/mapping/clap_val_15s.jsonl}"
METADATA_JSON="${METADATA_JSON:-$REPO/data/mapping/music_metadata.json}"
FALLBACK_TEXT="${FALLBACK_TEXT:-music}"

RUN_ID_TAG="${RUN_ID_TAG:-thesis_tag_only}"
RUN_ID_TAG_LLM="${RUN_ID_TAG_LLM:-thesis_tag_llm}"
SEEDS="${SEEDS:-42 43 44}"
TOP_K="${TOP_K:-10}"
REPORT_TOP_K="${REPORT_TOP_K:-10}"
MIN_CONFIDENCE="${MIN_CONFIDENCE:-0.35}"

SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_LLM_GEN="${SKIP_LLM_GEN:-0}"
SKIP_MERGE="${SKIP_MERGE:-0}"
SKIP_CACHE="${SKIP_CACHE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_TAG_TRAIN="${SKIP_TAG_TRAIN:-0}"
SKIP_TAG_LLM_TRAIN="${SKIP_TAG_LLM_TRAIN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
NO_AUDIO_CACHE="${NO_AUDIO_CACHE:-0}"
EVAL_CAPTION_INDEX="${EVAL_CAPTION_INDEX:-1}"
EVAL_METADATA_INDEX="${EVAL_METADATA_INDEX:-1}"
LLM_MAX_SONGS="${LLM_MAX_SONGS:-}"
SKIP_REPORT="${SKIP_REPORT:-0}"

export TAG_TRAIN_JSONL TAG_LLM_SONGS_JSONL TAG_LLM_TRAIN_JSONL VAL_JSONL METADATA_JSON
export ABLATION_DIR INDEX_DIR RUN_ID_TAG RUN_ID_TAG_LLM

BACKBONE="$REPO/model/clap/music_audioset_epoch_15_esc_90.14.pt"

mkdir -p "$ABLATION_DIR" "$INDEX_DIR"

if [[ ! -f "$BACKBONE" ]]; then
  echo "ERROR: CLAP backbone missing: $BACKBONE" >&2
  exit 1
fi

if [[ ! -f "$TRAIN_PARAMS" ]]; then
  echo "ERROR: train params missing: $TRAIN_PARAMS" >&2
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

_run_clap_multiseed() {
  local run_id="$1"
  local train_jsonl="$2"
  local -a cmd=(
    python -m app.train_clap_multiseed
    --run-id "$run_id"
    --seeds "$_SEEDS_CSV"
    --train-jsonl "$train_jsonl"
    --params-json "$TRAIN_PARAMS"
  )
  if ((${#_MULTISEED_FLAGS[@]})); then
    cmd+=("${_MULTISEED_FLAGS[@]}")
  fi
  "${cmd[@]}"
}

# --- 0. Build tag train JSONL ---
if [[ "$SKIP_BUILD" != "1" ]]; then
  echo "=== Build tag train JSONL ==="
  python -m app.data_handling.music_build_tag_train_jsonl \
    --train-jsonl "$BASE_TRAIN_JSONL" \
    --gold-jsonl "$GOLD_JSONL" \
    --out "$TAG_TRAIN_JSONL" \
    --fallback-text "$FALLBACK_TEXT"
else
  echo "SKIP_BUILD=1"
  if [[ ! -f "$TAG_TRAIN_JSONL" ]]; then
    echo "ERROR: tag train JSONL missing: $TAG_TRAIN_JSONL" >&2
    exit 1
  fi
fi

# --- 1. Tag → LLM song-level refine ---
if [[ "$SKIP_LLM_GEN" != "1" ]]; then
  echo "=== Tag → LLM refine (song-level, resumable) ==="
  _GEN_FLAGS=()
  if [[ -n "$LLM_MAX_SONGS" ]]; then
    _GEN_FLAGS+=(--max-songs "$LLM_MAX_SONGS")
  fi
  python -m app.data_handling.music_refine_tag_captions \
    --train-jsonl "$TAG_TRAIN_JSONL" \
    --progress-jsonl "$TAG_LLM_SONGS_JSONL" \
    --out-jsonl "$TAG_LLM_TRAIN_JSONL" \
    "${_GEN_FLAGS[@]}"
elif [[ "$SKIP_MERGE" != "1" ]]; then
  echo "SKIP_LLM_GEN=1 — merge-only into clip JSONL"
  python -m app.data_handling.music_refine_tag_captions \
    --train-jsonl "$TAG_TRAIN_JSONL" \
    --progress-jsonl "$TAG_LLM_SONGS_JSONL" \
    --out-jsonl "$TAG_LLM_TRAIN_JSONL" \
    --merge-only
else
  echo "SKIP_LLM_GEN=1 SKIP_MERGE=1"
  if [[ ! -f "$TAG_LLM_TRAIN_JSONL" ]]; then
    echo "ERROR: tag LLM train JSONL missing: $TAG_LLM_TRAIN_JSONL" >&2
    exit 1
  fi
fi

# --- 2. Audio backbone cache ---
if [[ "$SKIP_CACHE" != "1" ]] && [[ "$NO_AUDIO_CACHE" != "1" ]]; then
  echo "=== Ensure CLAP audio backbone cache ==="
  python -m app.data_handling.music_precompute_clap_audio_cache \
    --jsonl "$TAG_TRAIN_JSONL" \
    --jsonl "$VAL_JSONL"
else
  echo "SKIP_CACHE=1 or NO_AUDIO_CACHE=1"
fi

# --- 3. Fine-tune ---
if [[ "$SKIP_TRAIN" != "1" ]]; then
  if [[ "$SKIP_TAG_TRAIN" != "1" ]]; then
    echo "=== Fine-tune TAG-ONLY: $RUN_ID_TAG ==="
    _run_clap_multiseed "$RUN_ID_TAG" "$TAG_TRAIN_JSONL"
  else
    echo "SKIP_TAG_TRAIN=1"
  fi

  if [[ "$SKIP_TAG_LLM_TRAIN" != "1" ]]; then
    echo "=== Fine-tune TAG→LLM: $RUN_ID_TAG_LLM ==="
    if ! python -m app.data_handling.music_refine_tag_captions \
      --train-jsonl "$TAG_TRAIN_JSONL" \
      --progress-jsonl "$TAG_LLM_SONGS_JSONL" \
      --check-complete-only; then
      echo "ERROR: tag LLM refine incomplete; finish Job 1 first." >&2
      exit 1
    fi
    _run_clap_multiseed "$RUN_ID_TAG_LLM" "$TAG_LLM_TRAIN_JSONL"
  else
    echo "SKIP_TAG_LLM_TRAIN=1"
  fi
else
  echo "SKIP_TRAIN=1"
fi

_resolve_ckpt() {
  local run_id="$1"
  local seed="$2"
  local ckpt="$REPO/model/clap/finetune/${run_id}/seed_${seed}/best_model.pt"
  if [[ ! -f "$ckpt" ]]; then
    ckpt="$REPO/data/log/finetune_runs/${run_id}/seed_${seed}/best_model.pt"
  fi
  if [[ ! -f "$ckpt" ]]; then
    echo "ERROR: checkpoint missing for run_id=$run_id seed=$seed" >&2
    exit 1
  fi
  echo "$ckpt"
}

_index_paths() {
  local arm="$1"
  local kind="$2"
  local seed="$3"
  echo "${INDEX_DIR}/${kind}_${arm}_seed${seed}.faiss"
  echo "${INDEX_DIR}/${kind}_${arm}_seed${seed}.mapping.json"
}

_run_eval_arm() {
  local arm="$1"
  local run_id="$2"
  local caption_jsonl="$3"

  for seed in $SEEDS; do
    ckpt="$(_resolve_ckpt "$run_id" "$seed")"
    export RAGWEB_CLAP_CHECKPOINT="$ckpt"
    echo "=== Checkpoint $arm seed $seed ==="

    if [[ "$EVAL_METADATA_INDEX" == "1" ]]; then
      mapfile -t _meta_paths < <(_index_paths "$arm" "meta" "$seed")
      meta_index="${_meta_paths[0]}"
      meta_mapping="${_meta_paths[1]}"
      echo "  Build metadata FAISS ..."
      python -m app.data_handling.music_build_retrieval_faiss build-metadata \
        --metadata "$METADATA_JSON" \
        --out-index "$meta_index" \
        --out-mapping "$meta_mapping" \
        --min-confidence "$MIN_CONFIDENCE"
      echo "  Eval metadata index ..."
      python -m app.data_handling.music_eval_retrieval_vs_random \
        --top-k "$TOP_K" \
        --index "$meta_index" \
        --mapping "$meta_mapping" \
        --out-csv "$ABLATION_DIR/${arm}_meta_seed${seed}.csv" \
        --out-json "$ABLATION_DIR/${arm}_meta_seed${seed}.json"
    fi

    if [[ "$EVAL_CAPTION_INDEX" == "1" ]]; then
      mapfile -t _cap_paths < <(_index_paths "$arm" "caption" "$seed")
      cap_index="${_cap_paths[0]}"
      cap_mapping="${_cap_paths[1]}"
      echo "  Build caption FAISS from $caption_jsonl ..."
      python -m app.data_handling.music_build_retrieval_faiss build-caption \
        --jsonl "$caption_jsonl" \
        --out-index "$cap_index" \
        --out-mapping "$cap_mapping" \
        --min-confidence 0.0
      echo "  Eval caption index ..."
      python -m app.data_handling.music_eval_retrieval_vs_random \
        --top-k "$TOP_K" \
        --index "$cap_index" \
        --mapping "$cap_mapping" \
        --out-csv "$ABLATION_DIR/${arm}_caption_seed${seed}.csv" \
        --out-json "$ABLATION_DIR/${arm}_caption_seed${seed}.json"
    fi
  done
}

# --- 4. Eval ---
if [[ "$SKIP_EVAL" != "1" ]]; then
  echo "=== Symmetric retrieval eval (tag-only arm) ==="
  _run_eval_arm "tag" "$RUN_ID_TAG" "$TAG_TRAIN_JSONL"

  echo "=== Symmetric retrieval eval (tag→LLM arm) ==="
  _run_eval_arm "tag_llm" "$RUN_ID_TAG_LLM" "$TAG_LLM_TRAIN_JSONL"

  unset RAGWEB_CLAP_CHECKPOINT || true
else
  echo "SKIP_EVAL=1"
fi

# --- 5. Report ---
if [[ "$SKIP_REPORT" != "1" ]]; then
  echo "=== Tag vs tag→LLM ablation report ==="
  _INDEX_KINDS="meta"
  if [[ "$EVAL_CAPTION_INDEX" == "1" ]]; then
    _INDEX_KINDS="meta,caption"
  fi
  python -m app.data_handling.music_eval_tag_llm_ablation_report \
    --ablation-dir "$ABLATION_DIR" \
    --seeds "$_SEEDS_CSV" \
    --top-k "$REPORT_TOP_K" \
    --index-kinds "$_INDEX_KINDS"
fi

echo "Done."
echo "  Report: $ABLATION_DIR/REPORT.md"
echo "  Summary: $ABLATION_DIR/summary_primary.csv"
echo "  Indexes: $INDEX_DIR/"
