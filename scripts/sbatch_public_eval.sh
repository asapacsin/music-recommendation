#!/bin/bash
#SBATCH --job-name=public-ood
#SBATCH --partition=h800_batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

REPO="${RAGWEB_REPO:-$HOME/music-recommendation}"
MINICONDA="${MINICONDA:-$HOME/miniconda3}"

source "$MINICONDA/etc/profile.d/conda.sh"
conda activate ragweb

cd "$REPO"
export PYTHONPATH="$REPO"

# Example:
#   BUILD_MANIFESTS=1 DATASETS="jamendo mtat openmic" sbatch scripts/sbatch_public_eval.sh
#   DATASETS=jamendo SKIP_EXISTING=1 sbatch ...

export DATASETS="${DATASETS:-jamendo mtat openmic}"
export ARMS="${ARMS:-pretrained thesis_ft_v1}"
export SEEDS="${SEEDS:-42 43 44}"
export BUILD_MANIFESTS="${BUILD_MANIFESTS:-0}"
export SKIP_EXISTING="${SKIP_EXISTING:-0}"
export RUN_REPORT="${RUN_REPORT:-1}"

bash scripts/run_public_eval.sh
