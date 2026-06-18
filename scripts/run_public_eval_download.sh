#!/usr/bin/env bash
# Submit public-eval dataset download to Slurm (no screen, no attach).
#
# Usage (from repo root):
#   bash scripts/run_public_eval_download.sh mtat
#   bash scripts/run_public_eval_download.sh openmic
#   bash scripts/run_public_eval_download.sh jamendo
#   bash scripts/run_public_eval_download.sh all
#
# Check progress anytime:
#   bash scripts/status_public_eval_download.sh

set -euo pipefail

REPO="${RAGWEB_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"

WHAT="${1:-}"
if [[ -z "$WHAT" ]]; then
  cat <<'EOF'
Usage: bash scripts/run_public_eval_download.sh <target>

  mtat     — MagnaTagATune zip + extract + manifest
  openmic  — OpenMIC tarball + extract + manifest
  jamendo  — Jamendo five-tag MP3s + manifest (usually already done)
  all      — all three

After submit, disconnect freely. Check:
  bash scripts/status_public_eval_download.sh
EOF
  exit 1
fi

SKIP_JAMENDO=0
SKIP_MTAT=0
SKIP_OPENMIC=0

case "$WHAT" in
  mtat)
    SKIP_JAMENDO=1
    SKIP_OPENMIC=1
    if [[ "${EXTRACT_ONLY:-0}" == "1" ]]; then
      SKIP_DOWNLOAD=1
    fi
    ;;
  openmic)
    SKIP_JAMENDO=1
    SKIP_MTAT=1
    if [[ "${EXTRACT_ONLY:-0}" == "1" ]]; then
      SKIP_DOWNLOAD=1
    fi
    ;;
  jamendo)
    SKIP_MTAT=1
    SKIP_OPENMIC=1
    ;;
  all) ;;
  *)
    echo "Unknown target: $WHAT (use mtat | openmic | jamendo | all)" >&2
    exit 1
    ;;
esac

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. On cluster run: module load slurm or use login node." >&2
  exit 1
fi

JOB_ID="$(
  sbatch --parsable \
    --job-name="pub-dl-${WHAT}" \
    --export=ALL,SKIP_JAMENDO="${SKIP_JAMENDO}",SKIP_MTAT="${SKIP_MTAT}",SKIP_OPENMIC="${SKIP_OPENMIC}",SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}" \
    scripts/sbatch_download_public_eval.sh
)"

echo ""
echo "Submitted Slurm job ${JOB_ID}  (target=${WHAT})"
echo ""
echo "  Status:  bash scripts/status_public_eval_download.sh"
echo "  Slurm:   tail -f ${REPO}/slurm-${JOB_ID}.out"
echo "  MTAT:    tail -f ${REPO}/data/log/public_eval_downloads/mtat_backend.log"
echo "  OpenMIC: tail -f ${REPO}/data/log/public_eval_downloads/openmic_backend.log"
echo ""
echo "You can close this terminal now."
