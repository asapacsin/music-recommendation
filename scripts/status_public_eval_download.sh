#!/usr/bin/env bash
# One-page status for public-eval downloads (COMPLETED / RUNNING / FAILED).
#
# Usage: bash scripts/status_public_eval_download.sh
# Exit: 0=all COMPLETED, 1=any FAILED, 2=any RUNNING (none failed)

set -uo pipefail

REPO="${RAGWEB_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
LOG_DIR="$REPO/data/log/public_eval_downloads"

_validate_state() {
  local id="$1" state="$2"
  case "$id" in
    jamendo)
      local n ok n_m
      n="$(find "$REPO/data/public_eval/jamendo/audio_five_tag" -name '*.mp3' 2>/dev/null | wc -l | tr -d ' ')"
      read -r ok n_m <<< "$(python3 - <<PY
import json
from pathlib import Path
p = Path("$REPO/data/eval/jamendo_five_tag_manifest.jsonl")
ok = n = 0
if p.is_file():
  for line in p.read_text(encoding="utf-8").splitlines():
    if not line.strip(): continue
    n += 1
    ap = json.loads(line).get("audio_path")
    if ap and Path(ap).is_file(): ok += 1
print(ok, n)
PY
)"
      if [[ "${n:-0}" -ge 250 ]] && [[ "${ok:-0}" -eq "${n_m:-0}" ]] && [[ "${ok:-0}" -gt 0 ]]; then
        echo "COMPLETED"; return
      fi
      ;;
    mtat)
      local n ok n_m
      n="$(find "$REPO/data/public_eval/magnatagatune" -name '*.mp3' 2>/dev/null | wc -l | tr -d ' ')"
      read -r ok n_m <<< "$(python3 - <<PY
import json
from pathlib import Path
p = Path("$REPO/data/eval/mtat_manifest.jsonl")
ok = n = 0
if p.is_file():
  for line in p.read_text(encoding="utf-8").splitlines():
    if not line.strip(): continue
    n += 1
    ap = json.loads(line).get("audio_path")
    if ap and Path(ap).is_file(): ok += 1
print(ok, n)
PY
)"
      if [[ "${n:-0}" -ge 1000 ]] && [[ "${ok:-0}" -eq "${n_m:-0}" ]] && [[ "${ok:-0}" -gt 0 ]]; then
        echo "COMPLETED"; return
      fi
      ;;
    openmic)
      local n ok n_m
      n="$(find "$REPO/data/public_eval/openmic" -path '*/openmic-2018/audio/*.ogg' 2>/dev/null | wc -l | tr -d ' ')"
      read -r ok n_m <<< "$(python3 - <<PY
import json
from pathlib import Path
p = Path("$REPO/data/eval/openmic_manifest.jsonl")
ok = n = 0
if p.is_file():
  for line in p.read_text(encoding="utf-8").splitlines():
    if not line.strip(): continue
    n += 1
    ap = json.loads(line).get("audio_path")
    if ap and Path(ap).is_file(): ok += 1
print(ok, n)
PY
)"
      if [[ "${n:-0}" -ge 1000 ]] && [[ "${ok:-0}" -eq "${n_m:-0}" ]] && [[ "${ok:-0}" -gt 0 ]]; then
        echo "COMPLETED"; return
      fi
      ;;
  esac
  if [[ "$state" != "COMPLETED" ]]; then
    echo "$state"
    return
  fi
  echo "FAILED"
}

_slurm_running_for() {
  local needle="$1"
  if ! command -v squeue >/dev/null 2>&1; then
    return 1
  fi
  squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -qE "^pub-dl-${needle}$|^pub-dl-all$|^public-dl$" 2>/dev/null
}

_read_status_json() {
  local id="$1"
  local sf="$LOG_DIR/${id}.status.json"
  if [[ ! -f "$sf" ]]; then
    echo ""
    return
  fi
  python3 - <<PY
import json
from pathlib import Path
p = Path("$sf")
try:
    d = json.loads(p.read_text(encoding="utf-8"))
    print(d.get("state", ""))
    print(d.get("message", ""))
    print(d.get("updated_utc", ""))
except Exception:
    pass
PY
}

_infer_disk_state() {
  local id="$1"
  case "$id" in
    jamendo)
      local n
      n="$(find "$REPO/data/public_eval/jamendo/audio_five_tag" -name '*.mp3' 2>/dev/null | wc -l | tr -d ' ')"
      if [[ "${n:-0}" -ge 250 ]]; then
        echo "COMPLETED|~${n} mp3 files on disk (no status file)"
      else
        echo "NOT_STARTED|${n:-0}/297 mp3 files"
      fi
      ;;
    mtat)
      local n
      n="$(find "$REPO/data/public_eval/magnatagatune" -name '*.mp3' 2>/dev/null | wc -l | tr -d ' ')"
      if [[ "${n:-0}" -ge 1000 ]]; then
        echo "COMPLETED|${n} mp3 files on disk (no status file)"
      elif _slurm_running_for mtat; then
        echo "RUNNING|Slurm job active; ${n:-0} mp3 extracted so far"
      else
        local mtat_dir="$REPO/data/public_eval/magnatagatune"
        local exp1=1100000000 exp2=1100000000 exp3=772769864
        for part_spec in "mp3.zip.001:$exp1" "mp3.zip.002:$exp2" "mp3.zip.003:$exp3"; do
          local part="${part_spec%%:*}" expected="${part_spec##*:}"
          if [[ -f "$mtat_dir/$part" ]]; then
            local sz
            sz="$(stat -c%s "$mtat_dir/$part" 2>/dev/null || stat -f%z "$mtat_dir/$part")"
            if [[ "$sz" -gt "$expected" ]]; then
              echo "FAILED|Corrupt $part ($sz bytes, expected $expected) — rm $part and re-run"
              return
            fi
            if [[ "$sz" -lt "$expected" ]]; then
              echo "RUNNING|$part ${sz}/${expected} bytes (download incomplete)"
              return
            fi
          fi
        done
        if [[ -f "$mtat_dir/mp3.zip.003" ]] && [[ "${n:-0}" -eq 0 ]]; then
          echo "FAILED|Zip parts OK but 0 mp3 — extract failed; re-run: bash scripts/run_public_eval_download.sh mtat"
        elif [[ -f "$mtat_dir/mp3.zip.001" ]]; then
          echo "FAILED|Only ${n:-0} mp3 files — re-run: bash scripts/run_public_eval_download.sh mtat"
        else
          echo "NOT_STARTED|MTAT not downloaded"
        fi
      fi
      ;;
    openmic)
      local n
      n="$(find "$REPO/data/public_eval/openmic" -path '*/openmic-2018/audio/*.ogg' 2>/dev/null | wc -l | tr -d ' ')"
      if [[ "${n:-0}" -ge 1000 ]]; then
        echo "COMPLETED|${n} ogg files on disk (no status file)"
      elif _slurm_running_for openmic; then
        echo "RUNNING|Slurm job active; ${n:-0} ogg extracted so far"
      elif [[ -d "$REPO/data/public_eval/openmic/openmic-2018" ]] && [[ "${n:-0}" -lt 1000 ]]; then
        echo "FAILED|Extract dir exists but only ${n:-0} ogg — re-run: bash scripts/run_public_eval_download.sh openmic"
      else
        echo "NOT_STARTED|OpenMIC not downloaded"
      fi
      ;;
  esac
}

_dataset_line() {
  local id="$1" label="$2"
  local state="" message="" updated=""
  local json_out
  json_out="$(_read_status_json "$id")"
  if [[ -n "$json_out" ]]; then
    local orig_state
    orig_state="$(echo "$json_out" | sed -n '1p')"
    state="$orig_state"
    message="$(echo "$json_out" | sed -n '2p')"
    updated="$(echo "$json_out" | sed -n '3p')"
    state="$(_validate_state "$id" "$state")"
    if [[ "$orig_state" == "COMPLETED" ]] && [[ "$state" == "FAILED" ]]; then
      message="Status file said COMPLETED but disk check failed — re-run: bash scripts/run_public_eval_download.sh $id"
    elif [[ "$state" == "FAILED" ]] && [[ "$message" == "Script crashed"* ]]; then
      local log_hint=""
      case "$id" in
        mtat) log_hint="$LOG_DIR/mtat_backend.log" ;;
        openmic) log_hint="$LOG_DIR/openmic_backend.log" ;;
        jamendo) log_hint="$LOG_DIR/jamendo_five_tag_backend.log" ;;
      esac
      if [[ -f "$log_hint" ]]; then
        local last_err
        last_err="$(grep -E 'ERROR:|zip error|_public_eval_fail|FAILED' "$log_hint" 2>/dev/null | tail -1 || true)"
        if [[ -n "$last_err" ]]; then
          message="${last_err#*] }"
        fi
      fi
    fi
  fi
  if [[ -z "$state" ]]; then
    local inferred
    inferred="$(_infer_disk_state "$id")"
    state="${inferred%%|*}"
    message="${inferred#*|}"
  fi
  if [[ "$state" == "RUNNING" ]] && ! _slurm_running_for "$id" && [[ "$id" != "jamendo" ]]; then
    local log=""
    case "$id" in
      mtat) log="$LOG_DIR/mtat_backend.log" ;;
      openmic) log="$LOG_DIR/openmic_backend.log" ;;
      jamendo) log="$LOG_DIR/jamendo_five_tag_backend.log" ;;
    esac
    if [[ -f "$log" ]]; then
      local session_tail last_start
      last_start="$(grep -nE 'mtat_backend start|openmic_backend start|jamendo_five_tag_backend start' "$log" 2>/dev/null | tail -1 | cut -d: -f1 || true)"
      if [[ -n "$last_start" ]]; then
        session_tail="$(tail -n +"$last_start" "$log" 2>/dev/null | tail -20 || true)"
      else
        session_tail="$(tail -20 "$log" 2>/dev/null || true)"
      fi
      if echo "$session_tail" | grep -q 'STATUS=COMPLETED'; then
        state="COMPLETED"
        message="Last backend session completed — see $log"
      elif echo "$session_tail" | grep -qiE 'ERROR: .* FAILED|_public_eval_fail|zip error:|Script crashed'; then
        state="FAILED"
        message="$(echo "$session_tail" | grep -E 'ERROR:|zip error:|FAILED' | tail -1 | sed 's/^[[:space:]]*//')"
        [[ -z "$message" ]] && message="Last session failed — tail $log"
      fi
    fi
  fi
  printf '%s|%s|%s|%s' "$label" "$state" "$message" "$updated"
}

_print_banner() {
  local state="$1" label="$2" message="$3"
  case "$state" in
    COMPLETED)
      echo "================================================================"
      echo "  ${label}: COMPLETED"
      echo "  ${message}"
      echo "================================================================"
      ;;
    RUNNING)
      echo "----------------------------------------------------------------"
      echo "  ${label}: RUNNING"
      echo "  ${message}"
      echo "----------------------------------------------------------------"
      ;;
    FAILED)
      echo ""
      echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
      echo "  ${label}: FAILED"
      echo "  ${message}"
      echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
      echo ""
      ;;
    NOT_STARTED)
      echo "----------------------------------------------------------------"
      echo "  ${label}: NOT STARTED"
      echo "  ${message}"
      echo "  Start: bash scripts/run_public_eval_download.sh $id"
      echo "----------------------------------------------------------------"
      ;;
    *)
      echo "----------------------------------------------------------------"
      echo "  ${label}: ${state}"
      echo "  ${message}"
      echo "----------------------------------------------------------------"
      ;;
  esac
}

echo ""
echo "PUBLIC EVAL DOWNLOAD STATUS  ($(date -u +"%Y-%m-%dT%H:%M:%SZ") UTC)"
echo ""

any_failed=0
any_running=0
all_completed=1

for pair in "jamendo:JAMENDO" "mtat:MTAT" "openmic:OPENMIC"; do
  id="${pair%%:*}"
  label="${pair#*:}"
  line="$(_dataset_line "$id" "$label")"
  IFS='|' read -r _label state message updated <<< "$line"
  _print_banner "$state" "$label" "$message"
  [[ -n "$updated" ]] && echo "  (updated $updated)"
  echo ""
  case "$state" in
    FAILED) any_failed=1; all_completed=0 ;;
    RUNNING) any_running=1; all_completed=0 ;;
    COMPLETED) ;;
    *) all_completed=0 ;;
  esac
done

if command -v squeue >/dev/null 2>&1; then
  echo "--- Active Slurm jobs ---"
  squeue -u "$USER" -n pub-dl-mtat,pub-dl-openmic,pub-dl-jamendo,pub-dl-all,public-dl 2>/dev/null \
    || squeue -u "$USER" 2>/dev/null | grep -E 'pub-dl|public-dl|JOBID' || echo "(none)"
  echo ""
fi

LATEST_SLURM="$(ls -t slurm-*.out 2>/dev/null | head -1 || true)"
if [[ -n "$LATEST_SLURM" ]]; then
  echo "--- Latest slurm log ($LATEST_SLURM, last 5 lines) ---"
  tail -5 "$LATEST_SLURM" 2>/dev/null || true
  if grep -qE 'FAILED|ERROR:|Script crashed' "$LATEST_SLURM" 2>/dev/null; then
    echo ""
    echo ">>> Slurm log contains FAILED/ERROR — read: less $LATEST_SLURM"
  elif grep -q 'Slurm job finished OK' "$LATEST_SLURM" 2>/dev/null; then
    echo ""
    echo ">>> Slurm job exited OK (check dataset banners above for real outcome)"
  fi
  echo ""
fi

bash "$REPO/scripts/refresh_download_status.sh" >/dev/null 2>&1 || true
bash "$REPO/scripts/refresh_progress.sh" >/dev/null 2>&1 || true

echo "--- PROGRESS.md (Public OOD) ---"
grep -A18 "## Public OOD pipeline" "$REPO/docs/PROGRESS.md" 2>/dev/null | head -20 || echo "(run refresh_progress.sh)"
echo ""

if [[ "$any_failed" -eq 1 ]]; then
  echo "OVERALL: FAILED — fix errors above, then re-run the dataset download script."
  exit 1
elif [[ "$any_running" -eq 1 ]]; then
  echo "OVERALL: RUNNING — check again: bash scripts/status_public_eval_download.sh"
  exit 2
elif [[ "$all_completed" -eq 1 ]]; then
  echo "OVERALL: COMPLETED — all three datasets ready for public OOD eval."
  exit 0
else
  echo "OVERALL: INCOMPLETE — start missing datasets with bash scripts/run_public_eval_download.sh <name>"
  exit 2
fi
