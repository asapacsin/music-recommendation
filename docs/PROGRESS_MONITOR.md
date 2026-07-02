# Progress monitor (human guide)

Auto-generated status: **[`PROGRESS.md`](PROGRESS.md)** — refresh with:

```bash
bash scripts/refresh_progress.sh
```

Thesis scope: **Question E only** (Grok domain tradeoff). See [`THESIS_QUESTIONS.md`](THESIS_QUESTIONS.md).

---

## What the snapshot tracks

| Section | Meaning |
|---------|---------|
| **Thesis questions** | Question E status + report path |
| **Question E pipeline** | Units 0–5: JSONLs → cache → FT both arms → report |
| **Question E training recipe** | Hyperparams, arms, early-stop rule |
| **Fine-tune seeds** | Per-seed checkpoint + best val_similarity |
| **Public OOD pipeline** | Jamendo / MTAT / OpenMIC download + eval matrix |
| **Recent Slurm** | Tail of latest `slurm-*.out` |

---

## Question E pipeline units

| Unit | Step | Done when |
|------|------|-----------|
| 0 | Anime train JSONL | `clap_train_15s.jsonl` ~65k lines |
| 1 | Mixed train JSONL | `clap_train_grok_mixed.jsonl` exists |
| 2 | Audio cache | backbone cache `index.json` |
| 3 | FT `thesis_grok_only` | 3/3 seeds complete |
| 4 | FT `thesis_grok_mixed` | 3/3 seeds complete |
| 5 | Eval + report | `data/eval/domain_tradeoff/REPORT.md` |

**Params:** `data/eval/domain_tradeoff/train_params.json` (batch 32, max 20 epochs, val early-stop).

---

## Public OOD (Question E)

Default eval arms: `pretrained`, `thesis_grok_only`, `thesis_grok_mixed`.

```bash
ARMS="pretrained thesis_grok_only thesis_grok_mixed" bash scripts/run_public_eval.sh
```

Guide: [`PUBLIC_OOD_EVAL.md`](PUBLIC_OOD_EVAL.md).

---

## Useful checks

```bash
ls model/clap/finetune/thesis_grok_only/seed_*/best_model.pt
ls model/clap/finetune/thesis_grok_mixed/seed_*/best_model.pt
cat data/eval/domain_tradeoff/summary.json
```

Full pipeline: `sbatch scripts/sbatch_domain_tradeoff_ablation.sh` — see [`DOMAIN_TRADEOFF.md`](DOMAIN_TRADEOFF.md).
