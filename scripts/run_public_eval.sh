#!/usr/bin/env bash
# Post-train public OOD retrieval: Jamendo + MTAT + OpenMIC (no training).
set -euo pipefail

REPO="${RAGWEB_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="${PYTHONPATH:-$REPO}"

EVAL_ROOT="${EVAL_ROOT:-$REPO/data/eval}"
DATASETS="${DATASETS:-jamendo mtat openmic}"
SEEDS="${SEEDS:-42 43 44}"
TOP_K="${TOP_K:-10}"
AUDIO_BATCH="${AUDIO_BATCH:-16}"
ARMS="${ARMS:-pretrained thesis_ft_v1}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
RUN_REPORT="${RUN_REPORT:-1}"
BUILD_MANIFESTS="${BUILD_MANIFESTS:-0}"
MAX_PER_TAG="${MAX_PER_TAG:-60}"

declare -A MANIFEST_PATH=(
  [jamendo]="$REPO/data/eval/jamendo_five_tag_manifest.jsonl"
  [mtat]="$REPO/data/eval/mtat_manifest.jsonl"
  [openmic]="$REPO/data/eval/openmic_manifest.jsonl"
)

declare -A MANIFEST_BUILDER=(
  [jamendo]="python -m app.data_handling.music_eval_jamendo_five_tag_download --max-per-tag $MAX_PER_TAG"
  [mtat]="python -m app.data_handling.music_build_mtat_manifest --max-per-tag $MAX_PER_TAG"
  [openmic]="python -m app.data_handling.music_build_openmic_manifest --max-per-tag $MAX_PER_TAG"
)

declare -A ARM_RUN_ID=(
  [thesis_ft_v1]="thesis_ft_v1"
  [thesis_llm_full_llm]="thesis_llm_full_llm"
  [thesis_llm_ablation_orig]="thesis_llm_ablation_orig"
  [thesis_llm_ablation_llm]="thesis_llm_ablation_llm"
  [thesis_tag_only]="thesis_tag_only"
  [thesis_tag_llm]="thesis_tag_llm"
  [thesis_tag_mixed]="thesis_tag_mixed"
)

_resolve_ckpt() {
  local run_id="$1"
  local seed="$2"
  local ckpt
  for ckpt in \
    "$REPO/model/clap/finetune/${run_id}/seed_${seed}/best_model.pt" \
    "$REPO/model/finetune/${run_id}/seed_${seed}/best_model.pt" \
    "$REPO/data/log/finetune_runs/${run_id}/seed_${seed}/best_model.pt"; do
    if [[ -f "$ckpt" ]]; then
      echo "$ckpt"
      return 0
    fi
  done
  return 1
}

if [[ "$BUILD_MANIFESTS" == "1" ]]; then
  for ds in $DATASETS; do
    echo "=== Build manifest: $ds ==="
    ${MANIFEST_BUILDER[$ds]:-}
  done
fi

for ds in $DATASETS; do
  MANIFEST="${MANIFEST_PATH[$ds]:-}"
  EVAL_DIR="$EVAL_ROOT/${ds}_public"
  mkdir -p "$EVAL_DIR"

  if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: manifest missing for $ds: $MANIFEST" >&2
    echo "Set BUILD_MANIFESTS=1 or run: ${MANIFEST_BUILDER[$ds]:-}" >&2
    exit 1
  fi

  _n_ready="$(python - "$MANIFEST" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
ok = n = 0
for line in p.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    n += 1
    r = json.loads(line)
    ap = r.get("audio_path")
    if ap and Path(ap).is_file():
        ok += 1
print(ok, n)
PY
)"
  read -r _ok _total <<< "$_n_ready"
  echo "=== $ds manifest: ${_ok}/${_total} rows with local audio ==="
  if [[ "$_ok" -lt 1 ]]; then
    echo "ERROR: no audio for dataset $ds; download/extract public data first." >&2
    exit 1
  fi
  if [[ "$_ok" -lt 10 ]]; then
    echo "WARNING: small pool for $ds; metrics will be noisy." >&2
  fi

  for arm in $ARMS; do
    if [[ "$arm" == "pretrained" ]]; then
      for seed in $SEEDS; do
        out_csv="$EVAL_DIR/pretrained_seed${seed}.csv"
        if [[ "$SKIP_EXISTING" == "1" && -f "$out_csv" ]]; then
          echo "skip $out_csv"
          continue
        fi
        unset RAGWEB_CLAP_CHECKPOINT || true
        echo "=== $ds pretrained seed $seed ==="
        python -m app.data_handling.music_eval_public_retrieval \
          --dataset "$ds" \
          --manifest "$MANIFEST" \
          --out-csv "$out_csv" \
          --top-k "$TOP_K" \
          --seed "$seed" \
          --audio-batch-size "$AUDIO_BATCH" \
          --arm pretrained
      done
      continue
    fi

    run_id="${ARM_RUN_ID[$arm]:-$arm}"
    for seed in $SEEDS; do
      out_csv="$EVAL_DIR/${arm}_seed${seed}.csv"
      if [[ "$SKIP_EXISTING" == "1" && -f "$out_csv" ]]; then
        echo "skip $out_csv"
        continue
      fi
      ckpt="$(_resolve_ckpt "$run_id" "$seed")" || {
        echo "ERROR: checkpoint missing arm=$arm run_id=$run_id seed=$seed" >&2
        exit 1
      }
      export RAGWEB_CLAP_CHECKPOINT="$ckpt"
      echo "=== $ds $arm seed $seed ==="
      python -m app.data_handling.music_eval_public_retrieval \
        --dataset "$ds" \
        --manifest "$MANIFEST" \
        --out-csv "$out_csv" \
        --top-k "$TOP_K" \
        --seed "$seed" \
        --audio-batch-size "$AUDIO_BATCH" \
        --arm "$arm"
    done
  done
done
unset RAGWEB_CLAP_CHECKPOINT || true

if [[ "$RUN_REPORT" == "1" ]]; then
  _SEEDS_CSV="${SEEDS// /,}"
  _ARMS_CSV="${ARMS// /,}"
  _DS_CSV="${DATASETS// /,}"
  python -m app.data_handling.music_eval_public_report \
    --eval-root "$EVAL_ROOT" \
    --datasets "$_DS_CSV" \
    --arms "$_ARMS_CSV" \
    --seeds "$_SEEDS_CSV" \
    --top-k "$TOP_K"
  echo "Combined report: $EVAL_ROOT/REPORT.md"
fi

echo "Done. Per-dataset outputs: $EVAL_ROOT/{jamendo,mtat,openmic}_public/"
