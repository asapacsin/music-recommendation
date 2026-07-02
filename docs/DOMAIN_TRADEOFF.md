# Domain tradeoff: forgetting vs specialization (Question E)

**Goal:** Distinguish **catastrophic forgetting** from **specialization** when fine-tuning on the anime/game catalog with **Grok/metadata captions** on all ACG clips.

**2×2 design** (3 seeds 42–44):

|  | In-domain gold | Public OOD |
|--|----------------|------------|
| **Anime-only** (`thesis_grok_only`) | Eval | Eval |
| **Mixed** (`thesis_grok_mixed`) | Eval | Eval |

- **Anime-only train:** `clap_train_15s.jsonl` (Grok caption per clip)
- **Mixed train:** `clap_train_grok_mixed.jsonl` (Grok anime + MTAT/OpenMIC tag strings; eval holdouts excluded)
- **Jamendo never in training** (strict OOD)
- Both arms retrained with matched hyperparams (`data/eval/domain_tradeoff/train_params.json`)

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
# Build mixed JSONL
python -m app.data_handling.music_build_mixed_domain_train_jsonl \
  --anime-jsonl data/mapping/clap_train_15s.jsonl \
  --out-jsonl data/mapping/clap_train_grok_mixed.jsonl \
  --holdout-txt data/mapping/public_eval_holdout_paths.txt \
  --mix-ratio 0.5

# Full orchestrator
bash scripts/run_domain_tradeoff_ablation.sh

# Eval + report only (checkpoints exist)
SKIP_BUILD=1 SKIP_CACHE=1 SKIP_TRAIN=1 \
  bash scripts/run_domain_tradeoff_ablation.sh
```

### Environment overrides

| Variable | Default | Meaning |
|----------|---------|---------|
| `ANIME_JSONL` | `clap_train_15s.jsonl` | Grok anime train |
| `MIXED_JSONL` | `clap_train_grok_mixed.jsonl` | Mixed train output |
| `RUN_ID_ANIME` | `thesis_grok_only` | Anime-only checkpoints |
| `RUN_ID_MIXED` | `thesis_grok_mixed` | Mixed checkpoints |
| `TRADE_DIR` | `data/eval/domain_tradeoff` | Report + gold CSVs |
| `MIX_RATIO` | `0.5` | Public fraction when `PUBLIC_CLIP_TARGET=0` |
| `SEEDS` | `42 43 44` | Training + eval seeds |

Resume flags: `SKIP_BUILD`, `SKIP_CACHE`, `SKIP_TRAIN`, `SKIP_GOLD_EVAL`, `SKIP_PUBLIC_EVAL`, `SKIP_REPORT`.

---

## Outputs

| Artifact | Path |
|----------|------|
| Mixed train JSONL | `data/mapping/clap_train_grok_mixed.jsonl` |
| Holdout audit list | `data/mapping/public_eval_holdout_paths.txt` |
| Checkpoints | `model/clap/finetune/thesis_grok_{only,mixed}/seed_*/best_model.pt` |
| Gold CSVs | `data/eval/domain_tradeoff/{anime_only,mixed}_gold_seed*.csv` |
| Public CSVs | `data/eval/{jamendo,mtat,openmic}_public/thesis_grok_*_seed*.csv` |
| **2×2 report** | **`data/eval/domain_tradeoff/REPORT.md`** |
| Summary JSON | `data/eval/domain_tradeoff/summary.json` |

---

## Leakage rules

1. **Jamendo:** never included in mixed training; always OOD.
2. **MTAT / OpenMIC:** eval manifest paths are in `public_eval_holdout_paths.txt` and excluded from training.
3. Do not add eval manifest clips to mixed JSONL without updating holdouts.

---

## Prerequisites

- `data/mapping/clap_train_15s.jsonl` and `clap_val_15s.jsonl` (Grok captions)
- Public downloads COMPLETED: `bash scripts/status_public_eval_download.sh`
- Same `train_params.json` as LLM/tag ablation runs (batch 32, val early-stop)

---

## Related

- Question D (tag vs tag→LLM): [`docs/THESIS_QUESTIONS.md`](THESIS_QUESTIONS.md) — separate from Question E
- Public OOD: [`docs/PUBLIC_OOD_EVAL.md`](PUBLIC_OOD_EVAL.md)
- Agent run: [`docs/agent_runs/20260619_grok_domain_tradeoff/`](agent_runs/20260619_grok_domain_tradeoff/)

**Note:** Legacy tag-only / LLM ablation pipelines were removed from this repo; Question E uses Grok captions only.
