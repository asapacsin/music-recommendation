#!/usr/bin/env bash
# Backend MTAT download + split-zip extract + manifest build.
# Submit: bash scripts/run_public_eval_download.sh mtat

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/public_eval_download_common.sh
source "$SCRIPT_DIR/lib/public_eval_download_common.sh"

_public_eval_log_init "mtat_backend"
_public_eval_setup_traps
_public_eval_status_running "mtat" "MTAT download/extract started"

MTAT_BASE="${MTAT_BASE:-https://mirg.city.ac.uk/datasets/magnatagatune}"
MTAT_DIR="${MTAT_DIR:-$REPO/data/public_eval/magnatagatune}"
MTAT_ZIP_001_BYTES=1100000000
MTAT_ZIP_002_BYTES=1100000000
MTAT_ZIP_003_BYTES=772769864
BUILD_MANIFEST="${BUILD_MANIFEST:-1}"
MAX_PER_TAG="${MAX_PER_TAG:-60}"
MTAT_MIN_MP3=1000

mkdir -p "$MTAT_DIR"
cd "$MTAT_DIR"

_mtat_mp3_count() {
  find "$MTAT_DIR" -name '*.mp3' 2>/dev/null | wc -l | tr -d ' '
}

_mtat_part_size_ok() {
  local part="$1" expected="$2"
  [[ -f "$part" ]] || return 1
  [[ "$(_public_eval_file_size "$part")" -eq "$expected" ]]
}

_mtat_zip_parts_ready() {
  _mtat_part_size_ok mp3.zip.001 "$MTAT_ZIP_001_BYTES" \
    && _mtat_part_size_ok mp3.zip.002 "$MTAT_ZIP_002_BYTES" \
    && _mtat_part_size_ok mp3.zip.003 "$MTAT_ZIP_003_BYTES"
}

_ensure_mtat_part() {
  local part="$1" expected="$2"
  local url="$MTAT_BASE/$part"
  local got=0

  if [[ -f "$part" ]]; then
    got="$(_public_eval_file_size "$part")"
    if [[ "$got" -eq "$expected" ]]; then
      _public_eval_log "OK $part ($got bytes)"
      return 0
    fi
    if [[ "$got" -gt "$expected" ]]; then
      _public_eval_log "Corrupt $part ($got > $expected) — removing and re-downloading"
      rm -f "$part"
      got=0
    else
      _public_eval_log "Resuming $part ($got / $expected bytes)"
      _public_eval_status_running "mtat" "Downloading $part ($got / $expected bytes)"
    fi
  else
    _public_eval_log "Downloading $part (expected $expected bytes)"
    _public_eval_status_running "mtat" "Downloading $part"
  fi

  if [[ -f "$part" ]]; then
    wget -c "$url" -O "$part"
  else
    wget "$url" -O "$part"
  fi

  got="$(_public_eval_file_size "$part")"
  if [[ "$got" -ne "$expected" ]]; then
    if [[ "$got" -gt "$expected" ]]; then
      _public_eval_log "Post-wget size mismatch ($got > $expected) — retrying fresh download"
      rm -f "$part"
      wget "$url" -O "$part"
      got="$(_public_eval_file_size "$part")"
    fi
    if [[ "$got" -ne "$expected" ]]; then
      _public_eval_fail "mtat" "$part size $got != expected $expected — delete $part and re-run"
    fi
  fi
  _public_eval_log "Verified $part ($got bytes)"
}

_mtat_join_and_extract() {
  if ! _mtat_zip_parts_ready; then
    _public_eval_fail "mtat" "Zip parts missing or wrong size — cannot join split archive"
  fi

  _public_eval_status_running "mtat" "Joining split zip and extracting MP3s"
  rm -f mp3_all.zip

  _public_eval_log "Joining verified split zip with: cat mp3.zip.001 mp3.zip.002 mp3.zip.003 > mp3_all.zip"
  cat mp3.zip.001 mp3.zip.002 mp3.zip.003 > mp3_all.zip

  if [[ ! -f mp3_all.zip ]]; then
    _public_eval_fail "mtat" "cat join did not produce mp3_all.zip"
  fi

  _public_eval_log "Testing mp3_all.zip (unzip -t)"
  if ! unzip -t mp3_all.zip >/dev/null; then
    rm -f mp3_all.zip
    _public_eval_fail "mtat" "mp3_all.zip failed unzip -t — zip part sizes may be wrong; delete mp3.zip.* and re-run"
  fi

  _public_eval_log "Extracting mp3_all.zip"
  unzip -q -o mp3_all.zip || _public_eval_fail "mtat" "unzip mp3_all.zip failed"
}

_n_mp3="$(_mtat_mp3_count)"
if [[ "${_n_mp3:-0}" -lt "$MTAT_MIN_MP3" ]]; then
  if [[ "${SKIP_DOWNLOAD:-0}" == "1" ]]; then
    if ! _mtat_zip_parts_ready; then
      _public_eval_fail "mtat" "SKIP_DOWNLOAD=1 but zip parts missing or wrong size — run without SKIP_DOWNLOAD"
    fi
    _public_eval_log "SKIP_DOWNLOAD=1 — zip parts OK, extract only"
    _mtat_join_and_extract
  else
    _public_eval_log "--- MTAT CSVs ---"
    for f in annotations_final.csv clip_info_final.csv comparisons_final.csv; do
      if [[ ! -f "$f" ]]; then
        wget -c "$MTAT_BASE/$f"
      else
        _public_eval_log "have $f"
      fi
    done

    if _mtat_zip_parts_ready; then
      _public_eval_log "--- MTAT zip parts already verified ---"
    else
      _public_eval_log "--- MTAT zip parts ---"
      _ensure_mtat_part mp3.zip.001 "$MTAT_ZIP_001_BYTES"
      _ensure_mtat_part mp3.zip.002 "$MTAT_ZIP_002_BYTES"
      _ensure_mtat_part mp3.zip.003 "$MTAT_ZIP_003_BYTES"
    fi
    _mtat_join_and_extract
  fi
else
  _public_eval_log "Skipping download/extract (already have ${_n_mp3} mp3 files)"
fi

_n_mp3="$(_mtat_mp3_count)"
_public_eval_log "MTAT mp3 count: ${_n_mp3:-0}"
if [[ "${_n_mp3:-0}" -lt "$MTAT_MIN_MP3" ]]; then
  _public_eval_fail "mtat" "Only ${_n_mp3:-0} mp3 files (need >= $MTAT_MIN_MP3). Re-run: bash scripts/run_public_eval_download.sh mtat"
fi

if [[ "$BUILD_MANIFEST" == "1" ]]; then
  _public_eval_status_running "mtat" "Building MTAT manifest"
  _public_eval_activate_conda
  _public_eval_log "Building MTAT manifest"
  python -m app.data_handling.music_build_mtat_manifest --max-per-tag "$MAX_PER_TAG"
fi

read -r _manifest_ok _manifest_n <<< "$(_public_eval_manifest_ready_pair mtat_manifest.jsonl)"
_public_eval_manifest_audio_counts
_public_eval_refresh_snapshots

if [[ "${_manifest_ok:-0}" -lt 1 ]] || [[ "${_manifest_ok:-0}" -ne "${_manifest_n:-0}" ]]; then
  _public_eval_fail "mtat" "Manifest not ready (${_manifest_ok:-0}/${_manifest_n:-0} rows with audio). See log: $LOG_FILE"
fi

_public_eval_status_complete "mtat" "${_n_mp3} mp3 files; manifest ${_manifest_ok}/${_manifest_n} audio ready"
