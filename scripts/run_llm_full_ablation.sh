#!/usr/bin/env bash
# Full-corpus LLM vs original caption ablation:
#   - song-level LLM rewrite -> clip JSONL
#   - audio cache (one-time) -> GPU-focused fine-tune
#   - per-checkpoint metadata + caption FAISS rebuild -> gold eval -> report
#
# Usage:
#   sbatch scripts/sbatch_llm_full_corpus_gen.sh          # Job 1: LLM captions
#   sbatch scripts/sbatch_llm_full_ablation.sh            # Job 2: cache+FT+eval
#
# Resume LLM gen:
#   sbatch scripts/sbatch_llm_full_corpus_gen.sh          # auto-skips done songs
#
# Skip LLM gen (JSONL ready):
#   SKIP_LLM_GEN=1 sbatch scripts/sbatch_llm_full_ablation.sh
#
# Eval only:
#   SKIP_LLM_GEN=1 SKIP_TRAIN=1 sbatch scripts/sbatch_llm_full_ablation.sh
#
# Reuse existing sparse-ablation orig checkpoints:
#   RUN_ID_ORIG=thesis_llm_ablation_orig SKIP_ORIG_TRAIN=1 ...

set -euo pipefail

REPO="${RAGWEB_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
export REPO
export PYTHONPATH="${PYTHONPATH:-$REPO}"

ABLATION_DIR="${ABLATION_DIR:-$REPO/data/eval/llm_full_ablation}"
INDEX_DIR="${INDEX_DIR:-$ABLATION_DIR/index}"
TRAIN_JSONL="${TRAIN_JSONL:-$REPO/data/mapping/clap_train_15s.jsonl}"
LLM_SONGS_JSONL="${LLM_SONGS_JSONL:-$REPO/data/mapping/clap_train_llm_full_songs.jsonl}"
LLM_TRAIN_JSONL="${LLM_TRAIN_JSONL:-$REPO/data/mapping/clap_train_llm_full.jsonl}"
TRAIN_PARAMS="${TRAIN_PARAMS:-$REPO/data/eval/llm_ablation/train_params.json}"
VAL_JSONL="${VAL_JSONL:-$REPO/data/mapping/clap_val_15s.jsonl}"
METADATA_JSON="${METADATA_JSON:-$REPO/data/mapping/music_metadata.json}"

RUN_ID_ORIG="${RUN_ID_ORIG:-thesis_llm_ablation_orig}"
RUN_ID_LLM="${RUN_ID_LLM:-thesis_llm_full_llm}"
SEEDS="${SEEDS:-42 43 44}"
_SEEDS_CSV="${SEEDS// /,}"
TOP_K="${TOP_K:-10}"
REPORT_TOP_K="${REPORT_TOP_K:-10}"
MIN_CONFIDENCE="${MIN_CONFIDENCE:-0.35}"

SKIP_LLM_GEN="${SKIP_LLM_GEN:-0}"
SKIP_MERGE="${SKIP_MERGE:-0}"
SKIP_CACHE="${SKIP_CACHE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_ORIG_TRAIN="${SKIP_ORIG_TRAIN:-1}"
SKIP_LLM_TRAIN="${SKIP_LLM_TRAIN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
NO_AUDIO_CACHE="${NO_AUDIO_CACHE:-0}"
EVAL_CAPTION_INDEX="${EVAL_CAPTION_INDEX:-1}"
EVAL_METADATA_INDEX="${EVAL_METADATA_INDEX:-1}"
LLM_MAX_SONGS="${LLM_MAX_SONGS:-}"

export TRAIN_JSONL LLM_SONGS_JSONL LLM_TRAIN_JSONL VAL_JSONL METADATA_JSON
export ABLATION_DIR INDEX_DIR RUN_ID_ORIG RUN_ID_LLM

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

# --- 1. Full-corpus LLM song captions + merge to clip JSONL ---
if [[ "$SKIP_LLM_GEN" != "1" ]]; then
  echo "=== Full-corpus LLM caption refine (song-level, resumable) ==="
  _GEN_FLAGS=()
  if [[ -n "$LLM_MAX_SONGS" ]]; then
    _GEN_FLAGS+=(--max-songs "$LLM_MAX_SONGS")
  fi
  python -m app.data_handling.music_refine_full_corpus_captions \
    --train-jsonl "$TRAIN_JSONL" \
    --progress-jsonl "$LLM_SONGS_JSONL" \
    --out-jsonl "$LLM_TRAIN_JSONL" \
    "${_GEN_FLAGS[@]}"
elif [[ "$SKIP_MERGE" != "1" ]]; then
  echo "SKIP_LLM_GEN=1 — merge-only into clip JSONL"
  python -m app.data_handling.music_refine_full_corpus_captions \
    --train-jsonl "$TRAIN_JSONL" \
    --progress-jsonl "$LLM_SONGS_JSONL" \
    --out-jsonl "$LLM_TRAIN_JSONL" \
    --merge-only
else
  echo "SKIP_LLM_GEN=1 SKIP_MERGE=1"
  if [[ ! -f "$LLM_TRAIN_JSONL" ]]; then
    echo "ERROR: LLM train JSONL missing: $LLM_TRAIN_JSONL" >&2
    exit 1
  fi
fi

# --- 2. One-time audio backbone cache (avoids MP3 decode each epoch) ---
if [[ "$SKIP_CACHE" != "1" ]] && [[ "$NO_AUDIO_CACHE" != "1" ]]; then
  echo "=== Ensure CLAP audio backbone cache ==="
  python -m app.data_handling.music_precompute_clap_audio_cache \
    --jsonl "$TRAIN_JSONL" \
    --jsonl "$VAL_JSONL"
else
  echo "SKIP_CACHE=1 or NO_AUDIO_CACHE=1"
fi

# --- 3. Fine-tune ---
if [[ "$SKIP_TRAIN" != "1" ]]; then
  if [[ "$SKIP_ORIG_TRAIN" != "1" ]]; then
    echo "=== Fine-tune ORIGINAL captions: $RUN_ID_ORIG ==="
    _run_clap_multiseed "$RUN_ID_ORIG" "$TRAIN_JSONL"
  else
    echo "SKIP_ORIG_TRAIN=1 (reuse checkpoints under $RUN_ID_ORIG)"
  fi

  if [[ "$SKIP_LLM_TRAIN" != "1" ]]; then
    echo "=== Fine-tune FULL LLM captions: $RUN_ID_LLM ==="
    if ! python -m app.data_handling.music_refine_full_corpus_captions \
      --train-jsonl "$TRAIN_JSONL" \
      --progress-jsonl "$LLM_SONGS_JSONL" \
      --check-complete-only; then
      echo "ERROR: LLM song refine incomplete; finish Job 1 first." >&2
      exit 1
    fi
    _run_clap_multiseed "$RUN_ID_LLM" "$LLM_TRAIN_JSONL"
  else
    echo "SKIP_LLM_TRAIN=1"
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
      echo "  Build metadata FAISS (checkpoint-aware) ..."
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

# --- 4. Per-checkpoint index rebuild + gold eval ---
if [[ "$SKIP_EVAL" != "1" ]]; then
  echo "=== Symmetric retrieval eval (orig arm) ==="
  _run_eval_arm "orig" "$RUN_ID_ORIG" "$TRAIN_JSONL"

  echo "=== Symmetric retrieval eval (LLM arm) ==="
  _run_eval_arm "llm" "$RUN_ID_LLM" "$LLM_TRAIN_JSONL"

  unset RAGWEB_CLAP_CHECKPOINT
else
  echo "SKIP_EVAL=1"
fi

# --- 5. Report ---
echo "=== Full LLM ablation report ==="
export ABLATION_DIR SEEDS REPORT_TOP_K EVAL_CAPTION_INDEX
bash scripts/run_llm_full_ablation_report_only.sh

echo "Done."
echo "  Report: $ABLATION_DIR/REPORT.md"
echo "  Summary: $ABLATION_DIR/summary_primary.csv"
echo "  Indexes: $INDEX_DIR/"
