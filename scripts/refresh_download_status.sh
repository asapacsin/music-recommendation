#!/usr/bin/env bash
# Refresh data/eval/download_status_snapshot.json and print download progress.
set -euo pipefail

REPO="${RAGWEB_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"

count() {
  local n
  n=$(eval "$1" 2>/dev/null | wc -l | tr -d ' ')
  echo "${n:-0}"
}

JAMENDO_MP3=$(count "find data/public_eval/jamendo/audio_five_tag -name '*.mp3'")
MANIFEST_PATH_LINES=$(grep -c audio_path data/eval/jamendo_five_tag_manifest.jsonl 2>/dev/null | head -1 || true)
MANIFEST_PATH_LINES=${MANIFEST_PATH_LINES:-0}
MTAT_MP3=$(count "find data/public_eval/magnatagatune -name '*.mp3'")
OPENMIC_OGG=$(count "find data/public_eval/openmic -name '*.ogg'")
LLAMA_SHARDS=$(count "ls model/llama3.1-8b-instruct/model-*.safetensors")

MTAT_ZIP001_BYTES=0
if [[ -f data/public_eval/magnatagatune/mp3.zip.001 ]]; then
  MTAT_ZIP001_BYTES=$(stat -c%s data/public_eval/magnatagatune/mp3.zip.001 2>/dev/null || stat -f%z data/public_eval/magnatagatune/mp3.zip.001)
fi

JAMENDO_JOB=""
if pgrep -f 'music_eval_jamendo_five_tag_download' >/dev/null 2>&1; then
  JAMENDO_JOB="running"
fi

UPDATED_UTC=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

OUT="data/eval/download_status_snapshot.json"
mkdir -p data/eval

export UPDATED_UTC JAMENDO_MP3 MANIFEST_PATH_LINES JAMENDO_JOB MTAT_MP3 MTAT_ZIP001_BYTES OPENMIC_OGG LLAMA_SHARDS OUT
python3 <<'PY'
import json
import os
from pathlib import Path

job = os.environ.get("JAMENDO_JOB", "").strip()
snap = {
    "updated_utc": os.environ["UPDATED_UTC"],
    "jamendo_five_tag_mp3": int(os.environ.get("JAMENDO_MP3", 0)),
    "jamendo_five_tag_target": 297,
    "jamendo_manifest_audio_path_lines": int(os.environ.get("MANIFEST_PATH_LINES", 0)),
    "jamendo_five_tag_job": job if job else None,
    "mtat_mp3": int(os.environ.get("MTAT_MP3", 0)),
    "mtat_zip001_bytes": int(os.environ.get("MTAT_ZIP001_BYTES", 0)),
    "openmic_ogg": int(os.environ.get("OPENMIC_OGG", 0)),
    "llama_safetensor_shards": int(os.environ.get("LLAMA_SHARDS", 0)),
    "llama_safetensor_shards_expected": 4,
}
path = Path(os.environ["OUT"])
path.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
print(json.dumps(snap, indent=2))
PY

echo ""
echo "Doc: docs/DOWNLOAD_STATUS.md"
