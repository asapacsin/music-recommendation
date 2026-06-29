# Thesis revision brief (tradeoff-only) — for Opus

Revise the thesis to focus **only on Question E**: specialization–generalization tradeoff.

## Single research question

Does fine-tuning CLAP on ACG music trade in-domain gold retrieval for public OOD performance, and does **anime-only vs mixed** training corpus rebalance that tradeoff per tag (piano, vocal, relaxing)?

## Experimental arms (only these)

| Arm | Run ID | Training corpus | Training text |
|-----|--------|-----------------|----------------|
| Pretrained | — | — | — (OOD reference only) |
| Anime-only FT | `thesis_tag_only` | ACG catalog | Tag-only |
| Mixed FT | `thesis_tag_mixed` | ACG + MTAT + OpenMIC | Tag-only |

**Do not** use `thesis_ft_v1` (Grok captions) as the main comparison — that is Question A.

## Must add to thesis

1. **Table:** Main 2×2 from `eval/domain_tradeoff/REPORT.md` (P@10 + nDCG@10).
2. **Table:** OOD per dataset (Jamendo / MTAT / OpenMIC) from same report.
3. **Table:** Gold tag prevalence — piano 29, vocal 139, relaxing 76 (from `eval/gold/retrieval_vs_random_matrix.csv`).
4. **Table:** Hyperparameters from `hyperparams/*/seed_42_params.json`.
5. **Corpus stats:** 71,651 mixed clips = 65,041 anime + 4,576 MTAT + 2,034 OpenMIC; Jamendo never in train (`corpus/corpus_stats.json`).
6. **Arm mapping paragraph** — explain tag-only FT arms explicitly.

## Must remove or appendix

- LLM caption refinement (Question B)
- Iterative self-training (Question C)
- Tag vs tag→LLM (Question D)
- Drift filtering / hard-pair mining as primary method (Ch 4–5)
- Bootstrap CI claims unless you compute them
- Triple-duplicated §3.6.1 paragraph
- Contradictory pro–self-training sentences in §5.3 and §7.2

## Preprocessing (accurate wording)

- 15s clips; 48 kHz at CLAP load; stereo → mono downmix (librosa default)
- **No** loudness normalization
- FAISS IndexFlatIP on L2-normalized embeddings; metadata text index

## Instruction for Opus

Use only files in this bundle for numbers. Do not invent metrics. If a number is not in the bundle, say so.
