# PLAN — Grok-caption domain tradeoff (Question E v2)

**Run ID:** `20260619_grok_domain_tradeoff`  
**Goal:** Redo Question E with Grok/metadata captions on all ~65k ACG clips. Train **both** arms (`thesis_grok_only`, `thesis_grok_mixed`) with matched hyperparams (seeds 42–44) and produce the 2×2 report under **`data/eval/domain_tradeoff/`** (single canonical path).

---

## Context

- Tag-only domain tradeoff (`thesis_tag_only` / `thesis_tag_mixed`) is **superseded** — not used for Question E.
- Orchestrator: `scripts/run_domain_tradeoff_ablation.sh`
- Slurm: `scripts/sbatch_domain_tradeoff_ablation.sh`

---

## Inputs

| Path | Role |
|------|------|
| `data/mapping/clap_train_15s.jsonl` | Anime-only train (Grok captions) |
| `data/mapping/clap_val_15s.jsonl` | Val early-stop (Grok) |
| `data/eval/domain_tradeoff/train_params.json` | Matched FT hyperparams (batch 32, max 20 epochs) |
| `data/mapping/public_eval_holdout_paths.txt` | MTAT/OpenMIC eval holdouts |
| Public corpora (Jamendo, MTAT, OpenMIC) | OOD eval — Jamendo never in train |

---

## Commands (ordered)

```bash
cd ~/music-recommendation

# Step 1 — build mixed JSONL (Grok anime + MTAT/OpenMIC tag strings)
python -m app.data_handling.music_build_mixed_domain_train_jsonl \
  --anime-jsonl data/mapping/clap_train_15s.jsonl \
  --out-jsonl data/mapping/clap_train_grok_mixed.jsonl \
  --holdout-txt data/mapping/public_eval_holdout_paths.txt \
  --mix-ratio 0.5

# Full pipeline (build → cache → train both → gold + public → report)
sbatch scripts/sbatch_domain_tradeoff_ablation.sh

# Eval + report only (checkpoints exist)
SKIP_BUILD=1 SKIP_CACHE=1 SKIP_TRAIN=1 bash scripts/run_domain_tradeoff_ablation.sh
```

---

## Expected outputs

| Output path | What should be true |
|-------------|---------------------|
| `data/mapping/clap_train_grok_mixed.jsonl` | ~72k rows; anime rows retain Grok `text` |
| `data/mapping/clap_train_grok_mixed.jsonl.summary.json` | Build stats |
| `model/clap/finetune/thesis_grok_only/seed_{42,43,44}/best_model.pt` | Anime-only checkpoints |
| `model/clap/finetune/thesis_grok_mixed/seed_{42,43,44}/best_model.pt` | Mixed checkpoints |
| `data/eval/domain_tradeoff/anime_only_gold_seed*.csv` | In-domain gold (anime arm) |
| `data/eval/domain_tradeoff/mixed_gold_seed*.csv` | In-domain gold (mixed arm) |
| `data/eval/{jamendo,mtat,openmic}_public/thesis_grok_*_seed*.csv` | Public OOD per arm |
| **`data/eval/domain_tradeoff/REPORT.md`** | **2×2 P@10 + interpretation** |
| `data/eval/domain_tradeoff/summary.json` | Machine-readable summary |

---

## Risks / rollback

| Risk | Mitigation |
|------|------------|
| Long GPU time (2× FT) | 48h Slurm; audio cache reuse on overlapping anime paths |
| Overwrites tag E | Separate `TRADE_DIR` and run IDs |
| Public clips still short tags | Documented; only anime gets Grok |
| Job failure mid-pipeline | Resume flags: `SKIP_BUILD`, `SKIP_CACHE`, `SKIP_TRAIN`, etc. |

---

## Definition of done

- [ ] `tests/test_build_mixed_domain_train_jsonl.py` pass (Grok text preserved)
- [ ] Both arms complete seeds 42–44
- [ ] `domain_tradeoff/REPORT.md` has full 2×2 P@10 + OOD-by-dataset
- [ ] Thesis docs updated (THESIS_QUESTIONS, DOMAIN_TRADEOFF, AGENTS, OPUS_FEED)
