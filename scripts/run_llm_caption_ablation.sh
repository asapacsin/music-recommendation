#!/usr/bin/env bash
# LLM vs original caption ablation: build JSONL, fine-tune (3 seeds × 2 arms), gold eval, report.
#
# Usage:
#   cd ~/music-recommendation && bash scripts/run_llm_caption_ablation.sh
#   SKIP_TRAIN=1 bash scripts/run_llm_caption_ablation.sh   # eval + report only
#   SKIP_BUILD=1 SKIP_TRAIN=1 bash scripts/run_llm_caption_ablation.sh
#
# Resume after Slurm timeout (skips finished seeds; auto audio cache):
#   SKIP_BUILD=1 sbatch scripts/sbatch_llm_caption_ablation.sh
#
# Disable audio cache (slow MP3 decode every batch):
#   NO_AUDIO_CACHE=1 sbatch scripts/sbatch_llm_caption_ablation.sh
#
# Train one arm only:
#   SKIP_BUILD=1 SKIP_LLM=1 sbatch ...    # orig only
#   SKIP_BUILD=1 SKIP_ORIG=1 sbatch ...   # LLM only
#
# Slurm: sbatch scripts/sbatch_llm_caption_ablation.sh

set -euo pipefail

REPO="${RAGWEB_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="${PYTHONPATH:-$REPO}"

ABLATION_DIR="${ABLATION_DIR:-$REPO/data/eval/llm_ablation}"
TRAIN_JSONL="${TRAIN_JSONL:-$REPO/data/mapping/clap_train_15s.jsonl}"
LLM_TRAIN_JSONL="${LLM_TRAIN_JSONL:-$REPO/data/mapping/clap_train_llm_gated_iter0.jsonl}"
REFINED_JSONL="${REFINED_JSONL:-$REPO/data/self_train/thesis_self_v2/iter_0/refined.jsonl}"
TRAIN_PARAMS="${TRAIN_PARAMS:-$REPO/data/eval/llm_ablation/train_params.json}"
RUN_ID_ORIG="${RUN_ID_ORIG:-thesis_llm_ablation_orig}"
RUN_ID_LLM="${RUN_ID_LLM:-thesis_llm_ablation_llm}"
SEEDS="${SEEDS:-42 43 44}"
TOP_K="${TOP_K:-10}"
REPORT_TOP_K="${REPORT_TOP_K:-10}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_ORIG="${SKIP_ORIG:-0}"
SKIP_LLM="${SKIP_LLM:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
NO_AUDIO_CACHE="${NO_AUDIO_CACHE:-0}"
VAL_JSONL="${VAL_JSONL:-$REPO/data/mapping/clap_val_15s.jsonl}"

INDEX="${INDEX:-$REPO/data/index/metadata_text_index.faiss}"
BACKBONE="$REPO/model/clap/music_audioset_epoch_15_esc_90.14.pt"

mkdir -p "$ABLATION_DIR"

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

# --- 1. Build LLM-swapped train manifest ---
if [[ "$SKIP_BUILD" != "1" ]]; then
  echo "=== Build LLM train JSONL ==="
  python -m app.data_handling.music_build_llm_train_jsonl \
    --train-jsonl "$TRAIN_JSONL" \
    --refined-jsonl "$REFINED_JSONL" \
    --out-jsonl "$LLM_TRAIN_JSONL"
else
  echo "SKIP_BUILD=1"
  if [[ ! -f "$LLM_TRAIN_JSONL" ]]; then
    echo "ERROR: LLM train JSONL missing: $LLM_TRAIN_JSONL" >&2
    exit 1
  fi
fi

# --- 2. Fine-tune both arms (3 seeds each) ---
if [[ "$SKIP_TRAIN" != "1" ]]; then
  if [[ "$SKIP_ORIG" != "1" ]]; then
    echo "=== Fine-tune ORIGINAL captions: $RUN_ID_ORIG ==="
    python -m app.train_clap_multiseed \
      --run-id "$RUN_ID_ORIG" \
      --seeds "$(echo "$SEEDS" | tr ' ' ',')" \
      --train-jsonl "$TRAIN_JSONL" \
      --params-json "$TRAIN_PARAMS" \
      "${_MULTISEED_FLAGS[@]}"
  else
    echo "SKIP_ORIG=1"
  fi

  if [[ "$SKIP_LLM" != "1" ]]; then
    echo "=== Fine-tune LLM-swapped captions: $RUN_ID_LLM ==="
    python -m app.train_clap_multiseed \
      --run-id "$RUN_ID_LLM" \
      --seeds "$(echo "$SEEDS" | tr ' ' ',')" \
      --train-jsonl "$LLM_TRAIN_JSONL" \
      --params-json "$TRAIN_PARAMS" \
      "${_MULTISEED_FLAGS[@]}"
  else
    echo "SKIP_LLM=1"
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

# --- 3. Gold retrieval matrices ---
if [[ "$SKIP_EVAL" != "1" ]]; then
  if [[ ! -f "$INDEX" ]]; then
    echo "Building metadata FAISS index (one-time) ..."
    python -m app.metadata_faiss build --min-confidence 0.35
  fi

  for seed in $SEEDS; do
    ckpt="$(_resolve_ckpt "$RUN_ID_ORIG" "$seed")"
    export RAGWEB_CLAP_CHECKPOINT="$ckpt"
    echo "=== Eval ORIG seed $seed ==="
    python -m app.data_handling.music_eval_retrieval_vs_random \
      --top-k "$TOP_K" \
      --out-csv "$ABLATION_DIR/orig_seed${seed}.csv" \
      --out-json "$ABLATION_DIR/orig_seed${seed}.json"
  done

  for seed in $SEEDS; do
    ckpt="$(_resolve_ckpt "$RUN_ID_LLM" "$seed")"
    export RAGWEB_CLAP_CHECKPOINT="$ckpt"
    echo "=== Eval LLM seed $seed ==="
    python -m app.data_handling.music_eval_retrieval_vs_random \
      --top-k "$TOP_K" \
      --out-csv "$ABLATION_DIR/llm_seed${seed}.csv" \
      --out-json "$ABLATION_DIR/llm_seed${seed}.json"
  done
  unset RAGWEB_CLAP_CHECKPOINT
else
  echo "SKIP_EVAL=1"
fi

# --- 4. Report ---
echo "=== LLM vs original caption ablation report ==="
python -m app.data_handling.music_eval_llm_caption_ablation_report \
  --ablation-dir "$ABLATION_DIR" \
  --seeds "$(echo "$SEEDS" | tr ' ' ',')" \
  --top-k "$REPORT_TOP_K"

echo "Done."
echo "  Report: $ABLATION_DIR/REPORT.md"
echo "  Summary: $ABLATION_DIR/summary_primary.csv"
echo "  Matrices: $ABLATION_DIR/orig_seed*.csv, $ABLATION_DIR/llm_seed*.csv"
