#!/bin/bash
#SBATCH --job-name=clap-self
#SBATCH --partition=h800_batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

REPO="${RAGWEB_REPO:-$HOME/music-recommendation}"
MINICONDA="${MINICONDA:-$HOME/miniconda3}"

source "$MINICONDA/etc/profile.d/conda.sh"
conda activate ragweb

cd "$REPO"
export PYTHONPATH="$REPO"

nvidia-smi || true

RUN_ID="${RUN_ID:-thesis_self_v2}"
N_ITERS="${N_ITERS:-2}"
HARD_FRAC="${HARD_FRAC:-0.2}"
SEED="${SEED:-42}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
REFINE="${REFINE:-1}"
RUN_GOLD_EVAL="${RUN_GOLD_EVAL:-1}"
export RAGWEB_LLM_4BIT="${RAGWEB_LLM_4BIT:-1}"

# Training RAM: batch 128 OOM'd at 64G and 128G; epoch-end train sim is batched in init_model.
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-2}"
MIN_EPOCHS="${MIN_EPOCHS:-5}"
MIN_EPOCHS_BEFORE_OUTER="${MIN_EPOCHS_BEFORE_OUTER:-5}"

EXTRA=()
if [[ -n "$MAX_SAMPLES" ]]; then
  EXTRA+=(--max-samples "$MAX_SAMPLES")
fi

REFINE_FLAG=(--no-refine)
if [[ "$REFINE" == "1" ]]; then
  REFINE_FLAG=(--refine)
fi

GOLD_FLAG=()
if [[ "$RUN_GOLD_EVAL" == "1" ]]; then
  GOLD_FLAG=(--run-gold-eval)
fi

python -m app.train_clap_self_loop \
  --run-id "$RUN_ID" \
  --n-iters "$N_ITERS" \
  --hard-frac "$HARD_FRAC" \
  --seed "$SEED" \
  --batch-size "$BATCH_SIZE" \
  --num-epochs "$NUM_EPOCHS" \
  --early-stop-patience "$EARLY_STOP_PATIENCE" \
  --min-epochs "$MIN_EPOCHS" \
  --min-epochs-before-outer "$MIN_EPOCHS_BEFORE_OUTER" \
  "${REFINE_FLAG[@]}" \
  "${GOLD_FLAG[@]}" \
  "${EXTRA[@]}"

echo "Done. Checkpoints: $REPO/model/clap/self_train/$RUN_ID/iter_*/best_model.pt"
echo "Summary: $REPO/data/log/self_train_runs/$RUN_ID/summary.json"
