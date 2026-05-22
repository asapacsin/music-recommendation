#!/bin/bash
#SBATCH --job-name=clap-retrieval
#SBATCH --partition=h800_batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

# --- SICC: load CUDA if your site manual requires it (uncomment one line) ---
module purge
# module load cuda/12.4
# module load cudnn/...

# --- Repo + conda (adjust MINICONDA if not under $HOME/miniconda3) ---
REPO="${RAGWEB_REPO:-$HOME/music-recommendation}"
MINICONDA="${MINICONDA:-$HOME/miniconda3}"

source "$MINICONDA/etc/profile.d/conda.sh"
conda activate ragweb

cd "$REPO"
export PYTHONPATH="$REPO/app:$REPO"

nvidia-smi || true

BACKBONE="$REPO/model/clap/music_audioset_epoch_15_esc_90.14.pt"
INDEX="$REPO/data/index/metadata_text_index.faiss"

if [[ ! -f "$BACKBONE" ]]; then
  echo "ERROR: CLAP backbone missing: $BACKBONE" >&2
  exit 1
fi

if [[ ! -f "$INDEX" ]]; then
  echo "Building metadata FAISS index (one-time) ..."
  python -m app.metadata_faiss build --min-confidence 0.35
fi

RUN_ID="${RUN_ID:-thesis_ft_v1}"
SEEDS="${SEEDS:-42 43 44 45 46}"
TOP_K="${TOP_K:-10 20}"

for seed in $SEEDS; do
  ckpt="$REPO/model/clap/finetune/${RUN_ID}/seed_${seed}/best_model.pt"
  if [[ ! -f "$ckpt" ]]; then
    ckpt="$REPO/model/finetune/${RUN_ID}/seed_${seed}/best_model.pt"
  fi
  if [[ ! -f "$ckpt" ]]; then
    ckpt="$REPO/data/log/finetune_runs/${RUN_ID}/seed_${seed}/best_model.pt"
  fi
  if [[ ! -f "$ckpt" ]]; then
    echo "ERROR: checkpoint missing: $ckpt" >&2
    exit 1
  fi
  export RAGWEB_CLAP_CHECKPOINT="$ckpt"
  echo "=== seed $seed ($ckpt) ==="
  python -m app.data_handling.music_eval_retrieval_vs_random \
    --top-k $TOP_K \
    --out-csv "$REPO/data/eval/retrieval_matrix_seed${seed}.csv" \
    --out-json "$REPO/data/eval/retrieval_matrix_seed${seed}.json"
done
unset RAGWEB_CLAP_CHECKPOINT

echo "Done. Outputs: $REPO/data/eval/retrieval_matrix_seed*.csv"
