#!/usr/bin/env bash
# Backend Jamendo five-tag download.
# Submit: bash scripts/run_public_eval_download.sh jamendo

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/public_eval_download_common.sh
source "$SCRIPT_DIR/lib/public_eval_download_common.sh"

_public_eval_log_init "jamendo_five_tag_backend"
_public_eval_setup_traps
_public_eval_status_running "jamendo" "Jamendo five-tag download started"
_public_eval_activate_conda

MAX_PER_TAG="${MAX_PER_TAG:-60}"
JAMENDO_REPO="$REPO/data/public_eval/jamendo/mtg-jamendo-dataset"
JAMENDO_TARGET_MP3=297
JAMENDO_MIN_MP3=250

if [[ ! -d "$JAMENDO_REPO/.git" ]]; then
  _public_eval_log "Cloning MTG-Jamendo annotations repo"
  git clone https://github.com/MTG/mtg-jamendo-dataset.git "$JAMENDO_REPO"
else
  _public_eval_log "Jamendo annotations repo present"
fi

_public_eval_status_running "jamendo" "Downloading five-tag audio (max_per_tag=$MAX_PER_TAG)"
_public_eval_log "Starting five-tag audio download (max_per_tag=$MAX_PER_TAG)"
python -m app.data_handling.music_eval_jamendo_five_tag_download --max-per-tag "$MAX_PER_TAG"

_n_mp3="$(find "$REPO/data/public_eval/jamendo/audio_five_tag" -name '*.mp3' 2>/dev/null | wc -l | tr -d ' ')"
_public_eval_log "Jamendo five-tag MP3 count: ${_n_mp3:-0} (target ~$JAMENDO_TARGET_MP3)"
if [[ "${_n_mp3:-0}" -lt "$JAMENDO_MIN_MP3" ]]; then
  _public_eval_fail "jamendo" "Only ${_n_mp3:-0} mp3 files (need >= $JAMENDO_MIN_MP3). Re-run: bash scripts/run_public_eval_download.sh jamendo"
fi

read -r _manifest_ok _manifest_n <<< "$(_public_eval_manifest_ready_pair jamendo_five_tag_manifest.jsonl)"
_public_eval_manifest_audio_counts
_public_eval_refresh_snapshots

if [[ "${_manifest_ok:-0}" -lt 1 ]] || [[ "${_manifest_ok:-0}" -ne "${_manifest_n:-0}" ]]; then
  _public_eval_fail "jamendo" "Manifest not ready (${_manifest_ok:-0}/${_manifest_n:-0} rows with audio). See log: $LOG_FILE"
fi

_public_eval_status_complete "jamendo" "${_n_mp3} mp3 files; manifest ${_manifest_ok}/${_manifest_n} audio ready"
