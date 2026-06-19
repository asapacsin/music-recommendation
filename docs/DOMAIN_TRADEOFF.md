# Domain tradeoff: forgetting vs specialization (Question E)

**Goal:** Distinguish **catastrophic forgetting** from **specialization** when fine-tuning on the anime/game catalog.

**2×2 design** (tag-only training text, 3 seeds 42–44):

|  | In-domain gold | Public OOD |
|--|----------------|------------|
| **Anime-only** (`thesis_tag_only`) | Existing eval | Existing `data/eval/REPORT.md` |
| **Mixed** (`thesis_tag_mixed`) | New eval | New eval |

Mixed training = anime `clap_train_tag.jsonl` + MTAT + OpenMIC (eval holdouts excluded). **Jamendo never in training** (strict OOD).

---

## Hypothesis

| Outcome | Gold Δ (mixed − anime) | OOD Δ (mixed − anime) | Interpretation |
|---------|--------------------------|------------------------|----------------|
| Forgetting | ≈ 0 | **+large** (esp. vocal) | Anime-only FT erased pretrained alignment |
| Specialization | **+large** | negative | In-domain gain trades off OOD |
| No effect | ≈ 0 | ≈ 0 | Other factors (index, queries) |

---

## Commands

### Full pipeline (Slurm)

```bash
cd ~/music-recommendation
sbatch scripts/sbatch_domain_tradeoff_ablation.sh
```

### Step-by-step

```bash
# 1. Build mixed JSONL (50/50 anime/public rows by default)
python -m app.data_handling.music_build_mixed_domain_train_jsonl

# 2. Full orchestrator (build → cache → train → eval → report)
bash scripts/run_domain_tradeoff_ablation.sh

# 3. Eval + report only (after checkpoints exist)
SKIP_BUILD=1 SKIP_CACHE=1 SKIP_TRAIN=1 \
  bash scripts/run_domain_tradeoff_ablation.sh
```

### Environment overrides

| Variable | Default | Meaning |
|----------|---------|---------|
| `MIX_RATIO` | `0.5` | Public fraction when `PUBLIC_CLIP_TARGET=0` |
| `PUBLIC_CLIP_TARGET` | `0` | Explicit public row count (overrides ratio) |
| `RUN_ID_MIXED` | `thesis_tag_mixed` | Checkpoint folder |
| `RUN_ID_ANIME` | `thesis_tag_only` | Reference arm (no retrain) |
| `SEEDS` | `42 43 44` | Training + eval seeds |

---

## Outputs

| Artifact | Path |
|----------|------|
| Mixed train JSONL | `data/mapping/clap_train_tag_mixed.jsonl` |
| Holdout audit list | `data/mapping/public_eval_holdout_paths.txt` |
| Checkpoints | `model/clap/finetune/thesis_tag_mixed/seed_*/best_model.pt` |
| Gold CSVs | `data/eval/domain_tradeoff/{anime_only,mixed}_gold_seed*.csv` |
| Public CSVs | `data/eval/{jamendo,mtat,openmic}_public/thesis_tag_mixed_seed*.csv` |
| **2×2 report** | **`data/eval/domain_tradeoff/REPORT.md`** |
| Summary JSON | `data/eval/domain_tradeoff/summary.json` |

---

## Leakage rules

1. **Jamendo:** never included in mixed training; always OOD.
2. **MTAT / OpenMIC:** all paths in eval manifests are in `public_eval_holdout_paths.txt` and excluded from training.
3. Do not add eval manifest clips to mixed JSONL without updating holdouts.

---

## Prerequisites

- `data/mapping/clap_train_tag.jsonl` (Question D tag-only corpus)
- `thesis_tag_only` checkpoints + `data/eval/tag_llm_ablation/tag_meta_seed*.csv` (anime-only gold column)
- Public downloads COMPLETED: `bash scripts/status_public_eval_download.sh`
- Same `train_params.json` as Question D

---

## Related

- Question D: [`docs/THESIS_QUESTIONS.md`](THESIS_QUESTIONS.md)
- Public OOD: [`docs/PUBLIC_OOD_EVAL.md`](PUBLIC_OOD_EVAL.md)
- Agent run: [`docs/agent_runs/20260609_domain_tradeoff/PLAN.md`](agent_runs/20260609_domain_tradeoff/PLAN.md)
