#!/bin/bash
#SBATCH --job-name=llm-full-gen
#SBATCH --partition=h800_batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%j.out

# Job 1: full-corpus song-level LLM caption generation (resumable).
#
# Submit:
#   cd ~/music-recommendation && sbatch scripts/sbatch_llm_full_corpus_gen.sh
#
# Smoke (10 songs):
#   LLM_MAX_SONGS=10 sbatch scripts/sbatch_llm_full_corpus_gen.sh

set -euo pipefail

module purge
# module load cuda/12.4

REPO="${RAGWEB_REPO:-$HOME/music-recommendation}"
MINICONDA="${MINICONDA:-$HOME/miniconda3}"

source "$MINICONDA/etc/profile.d/conda.sh"
conda activate ragweb

cd "$REPO"
export PYTHONPATH="$REPO"

nvidia-smi || true

export SKIP_CACHE=1
export SKIP_TRAIN=1
export SKIP_EVAL=1
export SKIP_MERGE=0

bash scripts/run_llm_full_ablation.sh

echo "LLM gen job finished. Next: SKIP_LLM_GEN=1 sbatch scripts/sbatch_llm_full_ablation.sh"
