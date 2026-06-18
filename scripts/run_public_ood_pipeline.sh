#!/usr/bin/env bash
# End-to-end Public OOD pipeline: refresh progress → download (if needed) → eval (when ready).
#
# Usage:
#   bash scripts/run_public_ood_pipeline.sh              # auto: one next action
#   bash scripts/run_public_ood_pipeline.sh status       # refresh + print plan only
#   bash scripts/run_public_ood_pipeline.sh eval         # submit eval for prep-ready datasets
#   bash scripts/run_public_ood_pipeline.sh download openmic   # force download target
#   DRY_RUN=1 bash scripts/run_public_ood_pipeline.sh    # print actions, no submit
#
# Progress: bash scripts/refresh_progress.sh  → docs/PROGRESS.md § Public OOD pipeline

set -euo pipefail

REPO="${RAGWEB_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="${PYTHONPATH:-$REPO}"
export RAGWEB_REPO="$REPO"

MODE="${1:-auto}"
FORCE_DS="${2:-}"
DRY_RUN="${DRY_RUN:-0}"

echo "=== Public OOD pipeline ($MODE) $(date -u +"%Y-%m-%dT%H:%M:%SZ") ==="

bash "$REPO/scripts/refresh_download_status.sh" >/dev/null 2>&1 || true
bash "$REPO/scripts/refresh_progress.sh" >/dev/null 2>&1 || true

_run_download() {
  local ds="$1"
  echo ""
  echo ">> Submit download: $ds"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN: bash scripts/run_public_eval_download.sh $ds"
    return 0
  fi
  bash "$REPO/scripts/run_public_eval_download.sh" "$ds"
}

_run_eval() {
  local datasets="$1"
  echo ""
  echo ">> Submit public eval: DATASETS=$datasets"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN: DATASETS=\"$datasets\" ARMS=\"pretrained thesis_tag_only thesis_tag_llm\" SKIP_EXISTING=1 sbatch scripts/sbatch_public_eval.sh"
    return 0
  fi
  DATASETS="$datasets" ARMS="pretrained thesis_tag_only thesis_tag_llm" SKIP_EXISTING=1 \
    sbatch "$REPO/scripts/sbatch_public_eval.sh"
  echo "Monitor: bash scripts/refresh_progress.sh  (Public OOD pipeline section)"
}

case "$MODE" in
  status)
    bash "$REPO/scripts/status_public_eval_download.sh" || true
    python -m app.progress_monitor --ood-plan-json | python3 - <<'PY'
import json, sys
plan = json.load(sys.stdin)
print("\nPipeline units:")
for u in plan.get("pipeline_units", []):
    print(f"  [{u.get('unit')}] {u.get('label')}: {u.get('state')}")
print("\nNext commands:")
for c in plan.get("next_commands", []):
    print(f"  {c}")
print("\nPlanned actions:", json.dumps(plan.get("actions", []), indent=2))
PY
    exit 0
    ;;
  download)
    if [[ -z "$FORCE_DS" ]]; then
      echo "Usage: bash scripts/run_public_ood_pipeline.sh download <jamendo|mtat|openmic>" >&2
      exit 1
    fi
    _run_download "$FORCE_DS"
    exit 0
    ;;
  eval)
    READY="$(python3 - <<PY
import json
from app.progress_monitor import public_ood_pipeline_actions
plan = public_ood_pipeline_actions()
print(" ".join(plan.get("datasets_ready") or []))
PY
)"
    if [[ -z "$READY" ]]; then
      echo "ERROR: no datasets prep-ready for eval. Run downloads first." >&2
      bash "$REPO/scripts/status_public_eval_download.sh" || true
      exit 1
    fi
    _run_eval "$READY"
    exit 0
    ;;
  auto)
    PLAN_JSON="$(python -m app.progress_monitor --ood-plan-json)"
    PLAN_FILE="$(mktemp)"
    printf '%s' "$PLAN_JSON" >"$PLAN_FILE"
    python3 - "$PLAN_FILE" <<'PY'
import json, sys
plan = json.loads(open(sys.argv[1], encoding="utf-8").read())
print("Overall:", plan.get("status"))
print("Ready datasets:", " ".join(plan.get("datasets_ready") or []) or "(none)")
for u in plan.get("pipeline_units", []):
    print(f"  unit {u.get('unit')}: {u.get('state')} — {u.get('label')}")
PY
    rm -f "$PLAN_FILE"

    ACTION_TYPE="$(printf '%s' "$PLAN_JSON" | python3 -c "import json,sys; a=json.load(sys.stdin).get('actions',[]); print(a[0]['type'] if a else '')")"
    if [[ -z "$ACTION_TYPE" ]]; then
      echo ""
      echo "No automatic action (pipeline complete or waiting on running jobs)."
      bash "$REPO/scripts/status_public_eval_download.sh" || true
      exit 0
    fi

    if [[ "$ACTION_TYPE" == "download" ]]; then
      DS="$(printf '%s' "$PLAN_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['actions'][0]['dataset'])")"
      _run_download "$DS"
    elif [[ "$ACTION_TYPE" == "eval" ]]; then
      DS_LIST="$(printf '%s' "$PLAN_JSON" | python3 -c "import json,sys; print(' '.join(json.load(sys.stdin)['actions'][0]['datasets']))")"
      _run_eval "$DS_LIST"
    fi
    echo ""
    echo "After job finishes: bash scripts/refresh_progress.sh"
    ;;
  *)
    echo "Unknown mode: $MODE (use auto | status | eval | download <ds>)" >&2
    exit 1
    ;;
esac
