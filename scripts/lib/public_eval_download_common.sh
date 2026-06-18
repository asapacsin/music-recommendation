# Shared helpers for public-eval backend download scripts.
# Source from scripts/download_*_backend.sh — do not execute directly.

_public_eval_repo_root() {
  if [[ -n "${RAGWEB_REPO:-}" ]]; then
    echo "$RAGWEB_REPO"
  else
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
    echo "$(cd "$script_dir/.." && pwd)"
  fi
}

_public_eval_log_init() {
  local name="$1"
  REPO="$(_public_eval_repo_root)"
  LOG_DIR="${PUBLIC_EVAL_LOG_DIR:-$REPO/data/log/public_eval_downloads}"
  mkdir -p "$LOG_DIR"
  LOG_FILE="${LOG_DIR}/${name}.log"
  export REPO LOG_DIR LOG_FILE PUBLIC_EVAL_BACKEND="$name"
  exec > >(tee -a "$LOG_FILE") 2>&1
  echo "=== $(date -u +"%Y-%m-%dT%H:%M:%SZ") $name start ==="
  echo "REPO=$REPO"
  echo "LOG_FILE=$LOG_FILE"
}

_public_eval_log() {
  echo "[$(date -u +"%H:%M:%S")] $*"
}

_public_eval_file_size() {
  stat -c%s "$1" 2>/dev/null || stat -f%z "$1"
}

_public_eval_dataset_from_backend() {
  case "${PUBLIC_EVAL_BACKEND:-}" in
    mtat_backend) echo "mtat" ;;
    openmic_backend) echo "openmic" ;;
    jamendo_five_tag_backend) echo "jamendo" ;;
    *) echo "unknown" ;;
  esac
}

_public_eval_status_file() {
  local dataset="$1"
  echo "${LOG_DIR}/${dataset}.status.json"
}

_public_eval_status_set() {
  local dataset="$1"
  local state="$2"
  local message="$3"
  local extra_json="${4:-{}}"
  local path ts extra_file
  path="$(_public_eval_status_file "$dataset")"
  ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  extra_file="$(mktemp)"
  printf '%s' "$extra_json" >"$extra_file"
  _PES_DATASET="$dataset" _PES_STATE="$state" _PES_MSG="$message" \
    _PES_PATH="$path" _PES_TS="$ts" _PES_LOG="$LOG_FILE" _PES_EXTRA_FILE="$extra_file" \
    python3 - <<'PY'
import json, os
from pathlib import Path
extra_path = Path(os.environ["_PES_EXTRA_FILE"])
raw = extra_path.read_text(encoding="utf-8").strip()
extra_path.unlink(missing_ok=True)
try:
    extra = json.loads(raw) if raw else {}
except json.JSONDecodeError:
    extra = {}
doc = {
    "dataset": os.environ["_PES_DATASET"],
    "state": os.environ["_PES_STATE"],
    "updated_utc": os.environ["_PES_TS"],
    "message": os.environ["_PES_MSG"],
    "log_file": os.environ["_PES_LOG"],
}
doc.update(extra)
Path(os.environ["_PES_PATH"]).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
PY
}

_public_eval_banner() {
  local state="$1"
  local dataset="$2"
  local message="$3"
  echo ""
  echo "================================================================"
  echo "  PUBLIC EVAL DOWNLOAD — ${dataset^^}: ${state}"
  echo "================================================================"
  echo "  ${message}"
  echo "================================================================"
  echo ""
}

_public_eval_status_running() {
  local dataset="$1"
  local message="${2:-Job started}"
  _public_eval_status_set "$dataset" "RUNNING" "$message"
  _public_eval_banner "RUNNING" "$dataset" "$message"
}

_public_eval_status_complete() {
  local dataset="$1"
  local message="$2"
  local extra_json="${3:-{}}"
  _public_eval_status_set "$dataset" "COMPLETED" "$message" "$extra_json"
  _public_eval_banner "COMPLETED" "$dataset" "$message"
  _public_eval_log "STATUS=COMPLETED"
}

_public_eval_fail() {
  local dataset="$1"
  local message="$2"
  _public_eval_status_set "$dataset" "FAILED" "$message"
  _public_eval_banner "FAILED" "$dataset" "$message"
  echo ""
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  echo "  ERROR: ${dataset^^} download FAILED"
  echo "  ${message}"
  echo "  Log: ${LOG_FILE}"
  echo "  Check: bash scripts/status_public_eval_download.sh"
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  echo ""
  exit 1
}

_public_eval_on_err() {
  trap - ERR
  local dataset line
  dataset="$(_public_eval_dataset_from_backend)"
  line="$(grep -E 'ERROR:|zip error|wget:|tar:|unzip:' "$LOG_FILE" 2>/dev/null | tail -1 || true)"
  if [[ "$dataset" != "unknown" ]]; then
    if [[ -n "$line" ]]; then
      _public_eval_fail "$dataset" "${line#*] }"
    fi
    _public_eval_fail "$dataset" "Script crashed — see log above for the first error"
  fi
  exit 1
}

_public_eval_setup_traps() {
  trap _public_eval_on_err ERR
}

_public_eval_refresh_snapshots() {
  _public_eval_log "Refreshing download + progress snapshots"
  bash "$REPO/scripts/refresh_download_status.sh" || true
  bash "$REPO/scripts/refresh_progress.sh" || true
}

_public_eval_activate_conda() {
  if [[ -f "${MINICONDA:-$HOME/miniconda3}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${MINICONDA:-$HOME/miniconda3}/etc/profile.d/conda.sh"
    conda activate ragweb 2>/dev/null || true
  fi
  export PYTHONPATH="${PYTHONPATH:-$REPO}"
  cd "$REPO"
}

_public_eval_manifest_audio_counts() {
  _public_eval_activate_conda
  python - <<'PY'
import json
from pathlib import Path
repo = Path(__import__("os").environ["REPO"])
for name in ["jamendo_five_tag_manifest.jsonl", "mtat_manifest.jsonl", "openmic_manifest.jsonl"]:
    p = repo / "data/eval" / name
    if not p.is_file():
        print(f"{name}: MISSING manifest")
        continue
    ok = n = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        n += 1
        ap = json.loads(line).get("audio_path")
        if ap and Path(ap).is_file():
            ok += 1
    print(f"{name}: {ok}/{n} rows with local audio")
PY
}

_public_eval_manifest_ready_pair() {
  local manifest_file="$1"
  _public_eval_activate_conda
  python3 - <<PY
import json
from pathlib import Path
p = Path("$REPO/data/eval/$manifest_file")
ok = n = 0
if p.is_file():
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        n += 1
        ap = json.loads(line).get("audio_path")
        if ap and Path(ap).is_file():
            ok += 1
print(f"{ok} {n}")
PY
}
