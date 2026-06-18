#!/bin/bash
# Wait for Phase A (full 15s manifests), verify summary, then submit B4 self-train.
set -euo pipefail

REPO="${RAGWEB_REPO:-$HOME/music-recommendation}"
MARKER="$REPO/docs/agent_runs/20260526_self_train_v2/phase_a.done"
SUMMARY="$REPO/data/mapping/clap_split_summary.json"
MIN_GROUPS="${MIN_GROUPS:-1000}"
POLL_SEC="${POLL_SEC:-120}"

cd "$REPO"
echo "[wait_phase_a] Waiting for $MARKER ..."
while [[ ! -f "$MARKER" ]]; do
  nseg=$(find "$REPO/data/music_db_15s" -name '*.mp3' 2>/dev/null | wc -l || true)
  echo "[wait_phase_a] $(date -Is) segments=$nseg (no marker yet)"
  sleep "$POLL_SEC"
done

groups=$(python -c "import json; print(json.load(open('$SUMMARY'))['group_count_sources'])")
echo "[wait_phase_a] phase_a.done; group_count_sources=$groups"
if [[ "$groups" -lt "$MIN_GROUPS" ]]; then
  echo "[wait_phase_a] ERROR: expected group_count_sources >= $MIN_GROUPS" >&2
  exit 1
fi

RUN_ID="${RUN_ID:-thesis_self_v2}"
REFINE="${REFINE:-1}"
RUN_GOLD_EVAL="${RUN_GOLD_EVAL:-1}"
export RUN_ID REFINE RUN_GOLD_EVAL RAGWEB_LLM_4BIT="${RAGWEB_LLM_4BIT:-1}"

jid=$(sbatch --export=ALL,RUN_ID,REFINE,RUN_GOLD_EVAL,RAGWEB_LLM_4BIT \
  "$REPO/scripts/sbatch_clap_self_train.sh" | awk '{print $NF}')
echo "[wait_phase_a] Submitted Slurm job $jid (RUN_ID=$RUN_ID REFINE=$REFINE)"
echo "$jid" >> "$REPO/docs/agent_runs/20260526_self_train_v2/slurm_b4_jobid.txt"
