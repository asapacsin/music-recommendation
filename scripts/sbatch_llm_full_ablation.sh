#!/bin/bash
#SBATCH --job-name=llm-full-ablation
#SBATCH --partition=h800_batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%j.out

# Job 2: audio cache + full-LLM fine-tune + per-checkpoint FAISS + eval + report.
#
# Prerequisites: Job 1 finished OR clap_train_llm_full.jsonl exists.
#
# Submit:
#   SKIP_LLM_GEN=1 sbatch scripts/sbatch_llm_full_ablation.sh
#
# Resume after timeout (skips done seeds / uses existing cache):
#   SKIP_LLM_GEN=1 SKIP_MERGE=1 sbatch scripts/sbatch_llm_full_ablation.sh
#
# Eval only:
#   SKIP_LLM_GEN=1 SKIP_MERGE=1 SKIP_TRAIN=1 sbatch scripts/sbatch_llm_full_ablation.sh
#
# Reuse orig checkpoints from sparse ablation (default SKIP_ORIG_TRAIN=1):
#   RUN_ID_ORIG=thesis_llm_ablation_orig ...

set -euo pipefail

module purge
# module load cuda/12.4

REPO="${RAGWEB_REPO:-$HOME/music-recommendation}"
MINICONDA="${MINICONDA:-$HOME/miniconda3}"

source "$MINICONDA/etc/profile.d/conda.sh"
conda activate ragweb

cd "$REPO"
export PYTHONPATH="$REPO"

# Stage 15s clips to node-local disk (helps cache precompute if cache missing).
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
export SKIP_ORIG_TRAIN="${SKIP_ORIG_TRAIN:-1}"

bash scripts/run_llm_full_ablation.sh
