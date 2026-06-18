#!/bin/bash
#SBATCH --job-name=jamendo-pub
#SBATCH --partition=h800_batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

REPO="${RAGWEB_REPO:-$HOME/music-recommendation}"
MINICONDA="${MINICONDA:-$HOME/miniconda3}"

source "$MINICONDA/etc/profile.d/conda.sh"
conda activate ragweb

cd "$REPO"
export PYTHONPATH="$REPO"

# Eval only — does not train. Example:
#   ARMS="pretrained thesis_ft_v1 thesis_tag_only thesis_tag_llm" sbatch scripts/sbatch_jamendo_public_eval.sh
#   SKIP_EXISTING=1 sbatch ...   # resume partial run

export ARMS="${ARMS:-pretrained thesis_ft_v1}"
export SEEDS="${SEEDS:-42 43 44}"
export TOP_K="${TOP_K:-10}"
export SKIP_EXISTING="${SKIP_EXISTING:-0}"
export RUN_REPORT="${RUN_REPORT:-1}"

export DATASETS="${DATASETS:-jamendo}"
bash scripts/run_public_eval.sh
