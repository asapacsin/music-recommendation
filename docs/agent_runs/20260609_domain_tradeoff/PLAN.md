# PLAN — Domain tradeoff ablation (Question E)

**Run ID:** `20260609_domain_tradeoff`  
**Goal:** 2×2 comparison (anime-only vs mixed FT × in-domain gold vs public OOD) with 3 seeds to test forgetting vs specialization.

---

## Hypothesis

Anime-only tag-only FT may hurt public OOD via **catastrophic forgetting**. Mixed training (anime + MTAT + OpenMIC, Jamendo held out) should recover OOD if forgetting dominates.

---

## Commands

```bash
# Build mixed JSONL
python -m app.data_handling.music_build_mixed_domain_train_jsonl

# Full pipeline
sbatch scripts/sbatch_domain_tradeoff_ablation.sh

# Report only
SKIP_BUILD=1 SKIP_CACHE=1 SKIP_TRAIN=1 SKIP_PUBLIC_EVAL=0 \
  bash scripts/run_domain_tradeoff_ablation.sh
```

---

## Expected outputs

| Path | Description |
|------|-------------|
| `data/mapping/clap_train_tag_mixed.jsonl` | Mixed train corpus |
| `model/clap/finetune/thesis_tag_mixed/seed_{42,43,44}/best_model.pt` | Mixed checkpoints |
| `data/eval/domain_tradeoff/REPORT.md` | 2×2 report |
| `data/eval/domain_tradeoff/summary.json` | Machine-readable summary |

---

## Risks

- Large mixed JSONL (~130k rows) → long cache + train time
- MTAT/OpenMIC full-track vs anime 15s clip length mismatch
- Anime-only gold CSVs missing if tag ablation eval not run → copy step warns

---

## Success criteria

- `tests/test_build_mixed_domain_train_jsonl.py` pass
- `tests/test_domain_tradeoff_report.py` pass
- Holdout paths excluded from mixed JSONL
- Report classifies each tag (forgetting / specialization / none)
