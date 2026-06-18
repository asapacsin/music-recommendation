#!/bin/bash
#SBATCH --job-name=llm-cap-ablation
#SBATCH --partition=h800_batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%j.out

# LLM vs original caption ablation (3 seeds × 2 arms + gold eval + report).
#
# Submit:
#   cd ~/music-recommendation && sbatch scripts/sbatch_llm_caption_ablation.sh
#
# Resume after timeout (skips completed seeds; auto-builds audio cache if missing):
#   SKIP_BUILD=1 sbatch scripts/sbatch_llm_caption_ablation.sh
#
# After training completes (eval + report only):
#   SKIP_BUILD=1 SKIP_TRAIN=1 sbatch scripts/sbatch_llm_caption_ablation.sh
#
# One arm only:
#   SKIP_BUILD=1 SKIP_LLM=1 sbatch ...   # finish orig arm
#   SKIP_BUILD=1 SKIP_ORIG=1 sbatch ...  # LLM arm only
#
# Overrides: SEEDS="43 44" SKIP_BUILD=0 SKIP_TRAIN=0 SKIP_EVAL=0 STAGE_AUDIO=0 NO_AUDIO_CACHE=1

set -euo pipefail

module purge
# module load cuda/12.4

REPO="${RAGWEB_REPO:-$HOME/music-recommendation}"
MINICONDA="${MINICONDA:-$HOME/miniconda3}"

source "$MINICONDA/etc/profile.d/conda.sh"
conda activate ragweb

cd "$REPO"
export PYTHONPATH="$REPO"

# Stage 15s clips to node-local disk (faster MP3 reads during training).
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

bash scripts/run_llm_caption_ablation.sh
