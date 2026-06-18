#!/bin/bash
#SBATCH --job-name=tag-llm-ablation
#SBATCH --partition=h800_batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%j.out

# Job 2: tag JSONL (if needed) + cache + FT + eval + report.
#
# Prerequisites: Job 1 finished OR clap_train_tag_llm.jsonl exists.
#
# Submit:
#   SKIP_LLM_GEN=1 sbatch scripts/sbatch_tag_llm_ablation.sh
#
# Eval only:
#   SKIP_BUILD=1 SKIP_LLM_GEN=1 SKIP_MERGE=1 SKIP_CACHE=1 SKIP_TRAIN=1 \
#     sbatch scripts/sbatch_tag_llm_ablation.sh

set -euo pipefail

module purge

REPO="${RAGWEB_REPO:-$HOME/music-recommendation}"
MINICONDA="${MINICONDA:-$HOME/miniconda3}"

source "$MINICONDA/etc/profile.d/conda.sh"
conda activate ragweb

cd "$REPO"
export PYTHONPATH="$REPO"

if [[ "${STAGE_AUDIO:-1}" == "1" ]] && [[ -n "${SLURM_TMPDIR:-}" ]]; then
  STAGE_DEST="${SLURM_TMPDIR}/music_db_15s"
  if [[ ! -d "$STAGE_DEST" ]] || [[ -z "$(ls -A "$STAGE_DEST" 2>/dev/null || true)" ]]; then
    echo "Staging audio to $STAGE_DEST ..."
    mkdir -p "$STAGE_DEST"
    rsync -a "${REPO}/data/music_db_15s/" "$STAGE_DEST/"
  else
    echo "Using existing staged audio at $STAGE_DEST"
  fi
  export RAGWEB_AUDIO_15S_ROOT="$STAGE_DEST"
fi

nvidia-smi || true

export SKIP_LLM_GEN="${SKIP_LLM_GEN:-1}"

bash scripts/run_tag_llm_ablation.sh
