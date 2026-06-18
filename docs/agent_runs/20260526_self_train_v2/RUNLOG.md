# RUNLOG — `20260526_self_train_v2`

## Batch 1 — Plan + code (v2)

- Created agent run folder and implemented gate, LlmRefiner, val early stop, driver flags.
- Handoff snapshot: `STATE.md` (update when Phase A / B4 status changes).

## Batch 2 — Phase A full 15s split

- Started detached screen `clap_full_15s` (2026-05-26 05:21): `music_split_to_15s` → `music_build_train_val_from_15s` → `phase_a.done`
- Progress checkpoints:
  - ~06:42 UTC: ~15.7k segments; still `music_split_to_15s`
  - ~07:29 UTC: ~23.5k segments; `clap_split_summary.json` still pilot (44 / 665)
- Split saves incrementally: MP3s on disk + `music_15s_map.json` every 25 source files (resume without `--overwrite`)

## Batch 3 — v2 code smoke

- `smoke_self_v2_norefine`: exit 0; val monitor `best_val_similarity` 0.194
- `smoke_self_v2_refine`: exit 0; 2 LLM calls, 0 gate accepts (drift); train completed
- `manifest.py`: empty `refined.jsonl` no longer treats all hard rows as refined

## Batch 4 — B4 orchestration

- `screen clap_b4_wait` → `scripts/wait_phase_a_then_self_train.sh` (polls `phase_a.done`, then sbatch)
- **Not submitted yet** (waiting Phase A); log: `wait_b4.log`

## Batch 5 — Full thesis run

(Pending Phase A completion — see `STATE.md`.)
