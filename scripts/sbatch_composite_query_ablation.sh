#!/bin/bash
#SBATCH --job-name=composite-query
#SBATCH --partition=h800_batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out

# Composite CLAP prompts: "piano" -> "piano vocal" -> "piano vocal relaxing"
# AND multihot relevance; pretrained + fine-tuned seeds; no " music" suffix.
#
# Submit:
#   cd ~/music-recommendation && sbatch scripts/sbatch_composite_query_ablation.sh
#
# Reports only (CSVs already under data/eval/ablation/composite/):
#   SKIP_EVAL=1 sbatch scripts/sbatch_composite_query_ablation.sh

set -euo pipefail

REPO="${RAGWEB_REPO:-$HOME/music-recommendation}"
MINICONDA="${MINICONDA:-$HOME/miniconda3}"

source "$MINICONDA/etc/profile.d/conda.sh"
conda activate ragweb

cd "$REPO"
export PYTHONPATH="$REPO/app:$REPO"

COMPOSITE_DIR="${COMPOSITE_DIR:-$REPO/data/eval/ablation/composite}"
RUN_ID="${RUN_ID:-thesis_ft_v1}"
SEEDS="${SEEDS:-42 43 44 45 46}"
TOP_K="${TOP_K:-10 20}"
REPORT_TOP_K="${REPORT_TOP_K:-10}"
SKIP_EVAL="${SKIP_EVAL:-0}"

mkdir -p "$COMPOSITE_DIR"

nvidia-smi || true

INDEX="$REPO/data/index/metadata_text_index.faiss"
BACKBONE="$REPO/model/clap/music_audioset_epoch_15_esc_90.14.pt"

if [[ ! -f "$BACKBONE" ]]; then
  echo "ERROR: CLAP backbone missing: $BACKBONE" >&2
  exit 1
fi

if [[ "$SKIP_EVAL" != "1" ]]; then
  if [[ ! -f "$INDEX" ]]; then
    echo "Building metadata FAISS index ..."
    python -m app.metadata_faiss build --min-confidence 0.35
  fi

  echo "=== Pretrained composite queries ==="
  unset RAGWEB_CLAP_CHECKPOINT
  python -m app.data_handling.music_eval_composite_query_ablation \
    --top-k $TOP_K \
    --model-label pretrained \
    --out-csv "$COMPOSITE_DIR/composite_pretrained.csv"

  for seed in $SEEDS; do
    ckpt="$REPO/model/clap/finetune/${RUN_ID}/seed_${seed}/best_model.pt"
    if [[ ! -f "$ckpt" ]]; then
      ckpt="$REPO/model/finetune/${RUN_ID}/seed_${seed}/best_model.pt"
    fi
    if [[ ! -f "$ckpt" ]]; then
      ckpt="$REPO/data/log/finetune_runs/${RUN_ID}/seed_${seed}/best_model.pt"
    fi
    if [[ ! -f "$ckpt" ]]; then
      echo "ERROR: checkpoint missing for seed $seed" >&2
      exit 1
    fi
    export RAGWEB_CLAP_CHECKPOINT="$ckpt"
    echo "=== Fine-tuned seed $seed composite queries ==="
    python -m app.data_handling.music_eval_composite_query_ablation \
      --top-k $TOP_K \
      --model-label "ft_seed${seed}" \
      --out-csv "$COMPOSITE_DIR/composite_ft_seed${seed}.csv"
  done
  unset RAGWEB_CLAP_CHECKPOINT
else
  echo "SKIP_EVAL=1 — report only"
fi

echo "=== Composite query report ==="
python -m app.data_handling.music_eval_composite_query_report \
  --composite-dir "$COMPOSITE_DIR" \
  --top-k "$REPORT_TOP_K"

echo "Done."
echo "  Report: $COMPOSITE_DIR/composite_query_report.md"
echo "          $COMPOSITE_DIR/composite_query_summary.csv"
