#!/bin/bash
#SBATCH --job-name=domain-tradeoff
#SBATCH --partition=h800_batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%j.out

# Question E — Grok-caption domain tradeoff (anime-only vs mixed FT):
#   build clap_train_grok_mixed.jsonl -> cache -> FT both arms -> gold + public eval -> 2×2 report
#
# Submit:
#   sbatch scripts/sbatch_domain_tradeoff_ablation.sh
#
# Eval + report only:
#   SKIP_BUILD=1 SKIP_CACHE=1 SKIP_TRAIN=1 sbatch scripts/sbatch_domain_tradeoff_ablation.sh

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
    echo "Staging anime 15s audio to $STAGE_DEST ..."
    mkdir -p "$STAGE_DEST"
    rsync -a "${REPO}/data/music_db_15s/" "$STAGE_DEST/"
  fi
  export RAGWEB_AUDIO_15S_ROOT="$STAGE_DEST"
fi

nvidia-smi || true

bash scripts/run_domain_tradeoff_ablation.sh
