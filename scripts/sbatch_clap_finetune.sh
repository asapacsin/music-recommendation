#!/bin/bash
#SBATCH --job-name=clap-ft
#SBATCH --partition=h800_batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
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

# Optional: show GPU in log
nvidia-smi || true

# --- Multi-seed fine-tune (thesis: 5 seeds) ---
# Outputs: model/clap/finetune/<RUN_ID>/seed_*/best_model.pt; logs under data/log/finetune_runs/<RUN_ID>/
RUN_ID="${RUN_ID:-thesis_ft_v1}"
N_SEEDS="${N_SEEDS:-5}"
BASE_SEED="${BASE_SEED:-42}"

python -m app.train_clap_multiseed \
  --run-id "$RUN_ID" \
  --n-seeds "$N_SEEDS" \
  --base-seed "$BASE_SEED"

echo "Done. Checkpoints: $REPO/model/clap/finetune/$RUN_ID/seed_*/best_model.pt"
echo "Summary: $REPO/data/log/finetune_runs/$RUN_ID/summary.json"
