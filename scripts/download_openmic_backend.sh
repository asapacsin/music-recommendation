#!/usr/bin/env bash
# Backend OpenMIC download + extract + manifest build.
# Submit: bash scripts/run_public_eval_download.sh openmic

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/public_eval_download_common.sh
source "$SCRIPT_DIR/lib/public_eval_download_common.sh"

_public_eval_log_init "openmic_backend"
_public_eval_setup_traps
_public_eval_status_running "openmic" "OpenMIC download/extract started"

OPENMIC_DIR="${OPENMIC_DIR:-$REPO/data/public_eval/openmic}"
OPENMIC_TGZ="$OPENMIC_DIR/openmic-2018-v1.0.0.tgz"
OPENMIC_URL="${OPENMIC_URL:-https://zenodo.org/record/1432913/files/openmic-2018-v1.0.0.tgz}"
BUILD_MANIFEST="${BUILD_MANIFEST:-1}"
MAX_PER_TAG="${MAX_PER_TAG:-60}"
OPENMIC_MIN_OGG=1000
OPENMIC_MIN_TGZ_BYTES="${OPENMIC_MIN_TGZ_BYTES:-2500000000}"

mkdir -p "$OPENMIC_DIR"
cd "$OPENMIC_DIR"

_openmic_ogg_count() {
  local audio_dir="$OPENMIC_DIR/openmic-2018/audio"
  if [[ ! -d "$audio_dir" ]]; then
    echo 0
    return
  fi
  find "$audio_dir" -name '*.ogg' 2>/dev/null | wc -l | tr -d ' '
}

_openmic_labels_present() {
  [[ -f openmic-2018/openmic-2018-aggregated-labels.csv ]]
}

_openmic_extracted_ok() {
  local n
  n="$(_openmic_ogg_count)"
  _openmic_labels_present && [[ "${n:-0}" -ge "$OPENMIC_MIN_OGG" ]]
}

_openmic_extract_tarball() {
  if [[ ! -f "$OPENMIC_TGZ" ]]; then
    _public_eval_fail "openmic" "Missing tarball $OPENMIC_TGZ — run without SKIP_DOWNLOAD"
  fi
  got_tgz="$(_public_eval_file_size "$OPENMIC_TGZ")"
  if [[ "$got_tgz" -lt "$OPENMIC_MIN_TGZ_BYTES" ]]; then
    _public_eval_fail "openmic" "Tarball too small ($got_tgz bytes, need >= $OPENMIC_MIN_TGZ_BYTES)"
  fi
  _public_eval_status_running "openmic" "Extracting tarball"
  _public_eval_log "Extracting $OPENMIC_TGZ"
  tar -xzf "$OPENMIC_TGZ" -C "$OPENMIC_DIR" || _public_eval_fail "openmic" "tar extract failed for $OPENMIC_TGZ"
}

_n_ogg="$(_openmic_ogg_count)"

if _openmic_extracted_ok; then
  _public_eval_log "OpenMIC already extracted (${_n_ogg} ogg files)"
elif [[ "${SKIP_DOWNLOAD:-0}" == "1" ]]; then
  if _openmic_labels_present && [[ "${_n_ogg:-0}" -lt "$OPENMIC_MIN_OGG" ]]; then
    _public_eval_log "Incomplete extract (${_n_ogg} ogg) — removing openmic-2018/ before extract"
    rm -rf openmic-2018
  fi
  _public_eval_log "SKIP_DOWNLOAD=1 — extract only"
  _openmic_extract_tarball
elif [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
  if _openmic_labels_present && [[ "${_n_ogg:-0}" -lt "$OPENMIC_MIN_OGG" ]]; then
    _public_eval_log "Incomplete extract (${_n_ogg} ogg) — removing openmic-2018/ and re-downloading"
    rm -rf openmic-2018
  fi
  if [[ ! -f "$OPENMIC_TGZ" ]]; then
    _public_eval_status_running "openmic" "Downloading OpenMIC tarball"
    _public_eval_log "Downloading OpenMIC tarball"
    wget -c -O "$OPENMIC_TGZ" "$OPENMIC_URL"
  else
    _public_eval_status_running "openmic" "Resuming OpenMIC tarball download"
    _public_eval_log "Resuming OpenMIC tarball (wget -c)"
    wget -c -O "$OPENMIC_TGZ" "$OPENMIC_URL"
  fi
  _openmic_extract_tarball
else
  _public_eval_fail "openmic" "OpenMIC not extracted — run: bash scripts/run_public_eval_download.sh openmic"
fi

_n_ogg="$(_openmic_ogg_count)"
_public_eval_log "OpenMIC ogg count: ${_n_ogg:-0}"
if [[ "${_n_ogg:-0}" -lt "$OPENMIC_MIN_OGG" ]]; then
  _public_eval_fail "openmic" "Only ${_n_ogg:-0} ogg files (need >= $OPENMIC_MIN_OGG). Re-run: bash scripts/run_public_eval_download.sh openmic"
fi

if [[ "$BUILD_MANIFEST" == "1" ]]; then
  _public_eval_status_running "openmic" "Building OpenMIC manifest"
  _public_eval_activate_conda
  _public_eval_log "Building OpenMIC manifest"
  python -m app.data_handling.music_build_openmic_manifest --max-per-tag "$MAX_PER_TAG"
fi

read -r _manifest_ok _manifest_n <<< "$(_public_eval_manifest_ready_pair openmic_manifest.jsonl)"
_public_eval_manifest_audio_counts
_public_eval_refresh_snapshots

if [[ "${_manifest_ok:-0}" -lt 1 ]] || [[ "${_manifest_ok:-0}" -ne "${_manifest_n:-0}" ]]; then
  _public_eval_fail "openmic" "Manifest not ready (${_manifest_ok:-0}/${_manifest_n:-0} rows with audio). See log: $LOG_FILE"
fi

_public_eval_status_complete "openmic" "${_n_ogg} ogg files; manifest ${_manifest_ok}/${_manifest_n} audio ready"
