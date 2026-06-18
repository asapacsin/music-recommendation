#!/bin/bash
#SBATCH --job-name=tag-llm-gen
#SBATCH --partition=h800_batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%j.out

# Job 1: build tag JSONL + song-level LLM expand (resumable).
#
# Submit:
#   cd ~/music-recommendation && sbatch scripts/sbatch_tag_llm_corpus_gen.sh
#
# Smoke:
#   LLM_MAX_SONGS=10 sbatch scripts/sbatch_tag_llm_corpus_gen.sh

set -euo pipefail

module purge

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
export SKIP_REPORT=1
export SKIP_LLM_GEN=0
export SKIP_MERGE=0

bash scripts/run_tag_llm_ablation.sh

echo "Tag LLM gen finished. Next: SKIP_LLM_GEN=1 sbatch scripts/sbatch_tag_llm_ablation.sh"
