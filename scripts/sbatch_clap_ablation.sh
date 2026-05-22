#!/bin/bash
#SBATCH --job-name=clap-ablation
#SBATCH --partition=h800_batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

# Full ablation: pretrained + fine-tuned retrieval matrices (all style + tempo queries),
# then summary CSVs (primary 3 tags + all queries).
#
# Submit:
#   cd ~/music-recommendation && sbatch scripts/sbatch_clap_ablation.sh
#
# Overrides:
#   RUN_ID=thesis_ft_v1 SEEDS="42 43 44 45 46" TOP_K="10 20" sbatch scripts/sbatch_clap_ablation.sh
#   SKIP_EVAL=1 sbatch ...   # only rebuild report from existing CSVs under data/eval/ablation/

set -euo pipefail

module purge
# module load cuda/12.4

REPO="${RAGWEB_REPO:-$HOME/music-recommendation}"
MINICONDA="${MINICONDA:-$HOME/miniconda3}"

source "$MINICONDA/etc/profile.d/conda.sh"
conda activate ragweb

cd "$REPO"
export PYTHONPATH="$REPO/app:$REPO"

ABLATION_DIR="${ABLATION_DIR:-$REPO/data/eval/ablation}"
RUN_ID="${RUN_ID:-thesis_ft_v1}"
SEEDS="${SEEDS:-42 43 44 45 46}"
TOP_K="${TOP_K:-10 20}"
REPORT_TOP_K="${REPORT_TOP_K:-10}"
SKIP_EVAL="${SKIP_EVAL:-0}"

mkdir -p "$ABLATION_DIR"

nvidia-smi || true

BACKBONE="$REPO/model/clap/music_audioset_epoch_15_esc_90.14.pt"
INDEX="$REPO/data/index/metadata_text_index.faiss"

if [[ ! -f "$BACKBONE" ]]; then
  echo "ERROR: CLAP backbone missing: $BACKBONE" >&2
  exit 1
fi

if [[ "$SKIP_EVAL" != "1" ]]; then
  if [[ ! -f "$INDEX" ]]; then
    echo "Building metadata FAISS index (one-time) ..."
    python -m app.metadata_faiss build --min-confidence 0.35
  fi

  echo "=== Pretrained (no RAGWEB_CLAP_CHECKPOINT) ==="
  unset RAGWEB_CLAP_CHECKPOINT
  python -m app.data_handling.music_eval_retrieval_vs_random \
    --top-k $TOP_K \
    --out-csv "$ABLATION_DIR/pretrained.csv" \
    --out-json "$ABLATION_DIR/pretrained.json"

  for seed in $SEEDS; do
    ckpt="$REPO/model/clap/finetune/${RUN_ID}/seed_${seed}/best_model.pt"
    if [[ ! -f "$ckpt" ]]; then
      ckpt="$REPO/model/finetune/${RUN_ID}/seed_${seed}/best_model.pt"
    fi
    if [[ ! -f "$ckpt" ]]; then
      ckpt="$REPO/data/log/finetune_runs/${RUN_ID}/seed_${seed}/best_model.pt"
    fi
    if [[ ! -f "$ckpt" ]]; then
      echo "ERROR: checkpoint missing for seed $seed (checked model/clap/finetune, model/finetune, data/log)" >&2
      exit 1
    fi
    export RAGWEB_CLAP_CHECKPOINT="$ckpt"
    echo "=== Fine-tuned seed $seed ==="
    python -m app.data_handling.music_eval_retrieval_vs_random \
      --top-k $TOP_K \
      --out-csv "$ABLATION_DIR/ft_seed${seed}.csv" \
      --out-json "$ABLATION_DIR/ft_seed${seed}.json"
  done
  unset RAGWEB_CLAP_CHECKPOINT
else
  echo "SKIP_EVAL=1 — using existing matrices in $ABLATION_DIR (and legacy data/eval/ if needed)"
fi

echo "=== Ablation report ==="
python -m app.data_handling.music_eval_ablation_report \
  --ablation-dir "$ABLATION_DIR" \
  --pretrained-csv "$ABLATION_DIR/pretrained.csv" \
  --top-k "$REPORT_TOP_K" \
  --run-id "$RUN_ID"

echo "Done."
echo "  Matrices: $ABLATION_DIR/pretrained.csv, $ABLATION_DIR/ft_seed*.csv"
echo "  Report:   $ABLATION_DIR/summary_primary.csv (piano/vocal/relaxing @ K=$REPORT_TOP_K)"
echo "            $ABLATION_DIR/summary_all_queries.csv"
echo "            $ABLATION_DIR/summary.json"
