# Thesis experiment — Question E

**Research question:** When fine-tuning CLAP on anime music (Grok captions) vs adding public-domain clips (MTAT, OpenMIC), is the out-of-domain retrieval drop **forgetting** or **specialization**?

**Human summary:** [`README.md`](../README.md). **Agent context:** [`AGENTS.md`](../AGENTS.md). **Progress:** [`PROGRESS.md`](PROGRESS.md) (`bash scripts/refresh_progress.sh`).

**Primary evaluation tags:** `inst_piano`, `inst_vocal`, `mood_relaxing`.

---

## Three eval layers (do not mix)

| Layer | Metric | Data |
|-------|--------|------|
| **Training** | `val_similarity` on held-out clips | `clap_val_15s.jsonl` |
| **In-domain (gold)** | P@K, nDCG@K vs random | `gold_merged.jsonl` ∩ metadata FAISS |
| **Public OOD** | Same metrics on external audio | Jamendo / MTAT / OpenMIC manifests |

---

## Compared arms

| Column | Arm | Run ID | Training text |
|--------|-----|--------|---------------|
| Baseline | Pretrained CLAP | — | AudioSet backbone, no FT |
| Anime-only | Grok captions, ACG only | `thesis_grok_only` | `clap_train_15s.jsonl` |
| Mixed | Grok ACG + public clips | `thesis_grok_mixed` | `clap_train_grok_mixed.jsonl` |

Public clips in the mixed arm use **short tag strings** (dataset labels); ACG clips keep **Grok/metadata captions**.

---

## Status: **Done** (Slurm job 122295)

| Artifact | Path |
|----------|------|
| 2×2 report | [`data/eval/domain_tradeoff/REPORT.md`](../data/eval/domain_tradeoff/REPORT.md) |
| Summary JSON | `data/eval/domain_tradeoff/summary.json` |
| Checkpoints | `model/clap/finetune/thesis_grok_{only,mixed}/seed_*/best_model.pt` |
| FT logs | `data/log/finetune_runs/thesis_grok_*/` |
| Opus writing pack | [`docs/opus_tradeoff_bundle/OPUS_FEED.md`](opus_tradeoff_bundle/OPUS_FEED.md) |

---

## Reproduce

```bash
# Full pipeline
sbatch scripts/sbatch_domain_tradeoff_ablation.sh

# Eval only (checkpoints exist)
SKIP_BUILD=1 SKIP_CACHE=1 SKIP_TRAIN=1 bash scripts/run_domain_tradeoff_ablation.sh
```

Step-by-step: [`DOMAIN_TRADEOFF.md`](DOMAIN_TRADEOFF.md). Agent trace: [`agent_runs/20260619_grok_domain_tradeoff/`](agent_runs/20260619_grok_domain_tradeoff/).

---

## Headline results (job 122295)

**Gold P@10** (pretrained / anime-only / mixed): piano 0.20 / 0.40 / 0.40; vocal 0.70 / 0.90 / 0.80; relaxing 0.50 / 0.50 / 0.50.

**OOD macro P@10:** piano 0.98 / 0.83 / 0.84; vocal 0.76 / 0.33 / 0.60; relaxing 0.53 / 0.25 / 0.37.

**Interpretation:** Anime-only specializes on gold vocal; mixed partially recovers OOD vocal at modest in-domain cost — see report for 2×2 cells.

---

## Public OOD (post-train)

Does not train models. Default arms for this repo:

```bash
ARMS="pretrained thesis_grok_only thesis_grok_mixed" bash scripts/run_public_eval.sh
```

Guide: [`PUBLIC_OOD_EVAL.md`](PUBLIC_OOD_EVAL.md).
