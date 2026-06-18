# REVIEW — `20260526_self_train_v2`

## Checklist

- [x] `app/self_train/gate.py` — CLAP sim + text drift gate
- [x] `LlmRefiner` in `app/self_train/refine.py` — Llama + gate; `refiner.close()` frees VRAM
- [x] `model_creation` — `val_jsonl`, best-val checkpoint, patience early stop
- [x] `train_clap_self_loop` — `--refine`, outer val plateau, gate CLI flags
- [x] `scripts/sbatch_clap_self_train.sh` — REFINE=1, RUN_GOLD_EVAL=1, RAGWEB_LLM_4BIT
- [x] Smoke `--no-refine` (8 samples) — PASS
- [x] Smoke `--refine` (2 hard) — pipeline PASS (0/2 accepted by gate; drift reject)
- [ ] Phase A full 15s split — **in progress** (`screen clap_full_15s`; ~23.6k segments 2026-05-26; summary still 44 groups)
- [ ] Full Slurm B4 on full JSONL — waiter `clap_b4_wait` queued; pending `phase_a.done`

## Smoke notes

- Val monitor logged: `TrainSim` / `ValSim` / `checkpoint_metric` in `metrics.jsonl`
- Refine smoke: all captions rejected by `min_text_cos=0.85` (expected on tiny sample); manifest no longer duplicates unrefined hard rows when `refined.jsonl` empty
- `llm_local`: falls back to float16 if `bitsandbytes` missing

## Verdict

**PASS** (code + smoke). **FAIL** thesis-scale run until `clap_split_summary.json` shows full library.

## Next

1. Wait for `phase_a.done` + `group_count_sources` >> 44 (see `STATE.md`)
2. Confirm `slurm_b4_jobid.txt` from waiter, or manual sbatch (same env as `sbatch_clap_self_train.sh`)

**Handoff:** [`STATE.md`](STATE.md) — main points for the next session.
