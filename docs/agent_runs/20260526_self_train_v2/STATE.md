# STATE — `20260526_self_train_v2` (session handoff)

**Last updated:** 2026-05-27

## One-line status

**Phase A: DONE** (~3876 groups, 65k train). **B4 Slurm jobs 120645/120650 FAILED** (train OOM). **Fix applied:** batched epoch train-sim, sbatch 256G + batch 32 + min_epochs 5. **Resubmit B4.**

## Slurm history

| Job | Mem | Result |
|-----|-----|--------|
| 120645 | 64G | OOM at train (batch 128) |
| 120650 | 128G | OOM at train; mine/refine/mix OK (refined 1851/13009) |

## Resubmit

```bash
cd ~/music-recommendation
RUN_ID=thesis_self_v2 REFINE=1 RUN_GOLD_EVAL=1 sbatch scripts/sbatch_clap_self_train.sh
```

Optional overrides: `BATCH_SIZE=16`, `MIN_EPOCHS=8`, `NUM_EPOCHS=30`.

## iter_0 artifacts (reused until overwritten)

- `hard_mined.jsonl` 13009, `refined.jsonl` 1851, `train_mixed.jsonl` 66892
- No `best_model.pt` yet

## Code changes (2026-05-27)

- `init_model.mean_diagonal_similarity_batched` — fixes per-epoch 66k embed OOM
- `--min-epochs` (inner), `--min-epochs-before-outer` (outer)
- `sbatch_clap_self_train.sh`: 256G, batch 32, min_epochs 5
