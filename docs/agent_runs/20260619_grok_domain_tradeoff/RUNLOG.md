# RUNLOG — `20260619_grok_domain_tradeoff`

Append-only. Newest entries at the **bottom**.

---

## Entry — 2026-06-19 (implementation)

- **Actor**: Cursor agent
- **Command**:

```bash
cd ~/music-recommendation
python -m pytest tests/test_build_mixed_domain_train_jsonl.py -q
```

- **Exit code**: 0
- **Outputs created / updated**: (none — test only)
- **Notes**: 4 passed including `test_build_grok_domain_tradeoff_report` (2×2 schema for Grok arms).

---

## Entry — 2026-06-19 (holdout path fix)

- **Actor**: Cursor agent
- **Command**: Fixed `HOLDOUT_TXT` default in `run_grok_domain_tradeoff_ablation.sh` → `public_eval_holdout_paths.txt` (grok-specific path did not exist).

- **Exit code**: 0
- **Notes**: Prevents build failure on missing holdout file.

---

## Entry — 2026-06-19 (Slurm submit)

- **Actor**: Cursor agent
- **Command**:

```bash
cd ~/music-recommendation
sbatch scripts/sbatch_grok_domain_tradeoff_ablation.sh
```

- **Exit code**: 0
- **Outputs created / updated**: Slurm job **122294** (`slurm-122294.out`)
- **Notes**: Full pipeline — build, cache, FT both arms, gold + public eval, 2×2 report. Partition `h800_batch`, 48h budget.

---

## Entry — 2026-06-20 (job 122295 complete)

- **Command**: `sbatch scripts/sbatch_domain_tradeoff_ablation.sh` (job **122295**)
- **Exit code**: 0
- **Outputs**: `data/eval/domain_tradeoff/REPORT.md`, `summary.json`; checkpoints `thesis_grok_{only,mixed}/seed_{42,43,44}`; log `slurm-122295.out`
- **Notes**: Grok E complete. README + OPUS_FEED rewritten for specialization–generalization focus.

---

## Entry — 2026-06-19 (canonical path merge)

- **Actor**: Cursor agent
- **Change**: Merged Grok E pipeline into `run_domain_tradeoff_ablation.sh`; single output dir `data/eval/domain_tradeoff/`. Tag-only E pipeline removed. `run_grok_*` / `sbatch_grok_*` are deprecated aliases.
- **Job 122294**: If started with old `domain_tradeoff_grok/` path, after completion run:
  `rsync -a data/eval/domain_tradeoff_grok/ data/eval/domain_tradeoff/` then eval-only resume if needed.

---
