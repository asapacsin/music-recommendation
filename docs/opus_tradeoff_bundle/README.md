# Opus bundle — Question E (specialization–generalization tradeoff)

Copy of all thesis numbers and configs for **anime-only vs mixed** CLAP fine-tuning.  
Feed this entire folder to Opus when revising the tradeoff-focused thesis.

**Scope:** `thesis_tag_only` vs `thesis_tag_mixed` · gold (200 songs) + public OOD (Jamendo, MTAT, OpenMIC) · seeds 42–44.

---

## Start here (priority order)

| # | File | Use for |
|---|------|---------|
| 1 | `eval/domain_tradeoff/REPORT.md` | Main 2×2 table + OOD-by-dataset |
| 2 | `eval/domain_tradeoff/summary.json` | Same numbers in JSON (machine-readable) |
| 3 | `eval/public_OOD_REPORT.md` | Combined public OOD summary |
| 4 | `corpus/corpus_stats.json` | Mixed corpus line counts (not full 32MB JSONL) |
| 5 | `hyperparams/thesis_tag_only/seed_*_params.json` | Training config (anime-only arm) |
| 6 | `hyperparams/thesis_tag_mixed/seed_*_params.json` | Training config (mixed arm) |
| 7 | `eval/gold/retrieval_vs_random_matrix.csv` | Gold tag prevalence (29 / 139 / 76 positives) |
| 8 | `docs/DOMAIN_TRADEOFF.md` | Experiment design prose |
| 9 | `REVISION_BRIEF.md` | What to change in the thesis draft |

---

## Folder layout

```
opus_tradeoff_bundle/
├── README.md                          ← this file
├── REVISION_BRIEF.md                  ← edit checklist for Opus
├── corpus/
│   ├── corpus_stats.json              ← 71,651 mixed = 65,041 anime + 4,576 MTAT + 2,034 OpenMIC
│   ├── clap_train_tag_SAMPLE.jsonl    ← 5 example rows (anime-only train)
│   ├── clap_train_tag_mixed_SAMPLE.jsonl
│   └── clap_val_15s_SAMPLE.jsonl
├── docs/
│   ├── DOMAIN_TRADEOFF.md
│   └── THESIS_QUESTIONS.md            ← full A–E map (main thesis = Question E only)
├── eval/
│   ├── domain_tradeoff/
│   │   ├── REPORT.md
│   │   └── summary.json
│   ├── public_OOD_REPORT.md
│   ├── gold/
│   │   ├── retrieval_vs_random_matrix.csv
│   │   ├── anime_only_gold_seed{42,43,44}.csv
│   │   └── mixed_gold_seed{42,43,44}.csv
│   └── public_ood/
│       ├── jamendo/   pretrained + thesis_tag_only + thesis_tag_mixed × 3 seeds
│       ├── mtat/
│       └── openmic/
└── hyperparams/
    ├── thesis_tag_only/seed_{42,43,44}_params.json
    └── thesis_tag_mixed/seed_{42,43,44}_params.json
```

---

## Headline numbers (mean over seeds 42–44, P@10)

From `eval/domain_tradeoff/summary.json`:

| Tag | Anime-only Gold | Mixed Gold | Δ Gold | Anime-only OOD | Mixed OOD | Δ OOD | Pretrained OOD |
|-----|-----------------|------------|--------|----------------|-----------|-------|----------------|
| inst_piano | 0.20 | 0.30 | +0.10 | 0.70 | 0.69 | −0.01 | 0.98 |
| inst_vocal | 1.00 | 0.90 | −0.10 | 0.37 | 0.53 | +0.17 | 0.76 |
| mood_relaxing | 0.50 | 0.50 | 0.00 | 0.28 | 0.40 | +0.12 | 0.53 |

**Arms:** tag-only training text · backbone frozen · projection/transform fine-tuned · val early-stop on `clap_val_15s.jsonl`.

**Not in this bundle:** `thesis_ft_v1` (Grok-caption FT, Question A) — different arm; do not conflate with E.

---

## Full JSONL manifests (not copied — too large)

| File | Lines | Repo path |
|------|-------|-----------|
| Mixed train | 71,651 | `data/mapping/clap_train_tag_mixed.jsonl` |
| Anime-only train | 65,041 | `data/mapping/clap_train_tag.jsonl` |
| Val | 7,246 | `data/mapping/clap_val_15s.jsonl` |

Use `corpus/corpus_stats.json` + samples for Methods; do not open full JSONL in the editor.

---

## Regenerate this bundle

From repo root:

```bash
# Re-run the copy script or ask the agent to refresh docs/opus_tradeoff_bundle/
```

Source reports live under `data/eval/domain_tradeoff/` and `data/eval/*_public/`.
