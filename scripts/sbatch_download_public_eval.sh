#!/bin/bash
#SBATCH --job-name=public-dl
#SBATCH --partition=h800_batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

# Public-eval dataset download (no GPU). Submit via:
#   bash scripts/run_public_eval_download.sh mtat
#   bash scripts/run_public_eval_download.sh openmic
#   bash scripts/run_public_eval_download.sh all

set -euo pipefail

REPO="${RAGWEB_REPO:-$HOME/music-recommendation}"
MINICONDA="${MINICONDA:-$HOME/miniconda3}"

source "$MINICONDA/etc/profile.d/conda.sh"
conda activate ragweb

cd "$REPO"
export PYTHONPATH="$REPO"
export RAGWEB_REPO="$REPO"

echo "=== public-dl Slurm job $(date -u +"%Y-%m-%dT%H:%M:%SZ") ==="
echo "SKIP_JAMENDO=${SKIP_JAMENDO:-0} SKIP_MTAT=${SKIP_MTAT:-0} SKIP_OPENMIC=${SKIP_OPENMIC:-0}"
echo "Monitor: bash scripts/status_public_eval_download.sh"

echo "Monitor: bash scripts/status_public_eval_download.sh"

if bash scripts/download_public_eval_backend.sh; then
  echo "=== Slurm job finished OK ==="
else
  rc=$?
  echo ""
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  echo "  ERROR: public-dl Slurm job FAILED (exit $rc)"
  echo "  Check: bash scripts/status_public_eval_download.sh"
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  exit "$rc"
fi
echo "Next: bash scripts/status_public_eval_download.sh"
