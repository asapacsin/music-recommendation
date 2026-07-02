# Thesis data pack — Question E only (paste this entire file into Opus)

**Purpose:** Revise the thesis to focus on the **specialization–generalization tradeoff** only.  
**Rule for Opus:** Use only numbers in this document. Do not invent metrics.

**Status:** Complete — Slurm job **122295** (2026). Source: `data/eval/domain_tradeoff/summary.json`.

---

## 1. What Opus should do

Rewrite the thesis around **one question**:

> When CLAP is fine-tuned on ACG music with **Grok/metadata captions**, how much **in-domain gold retrieval** is gained versus **public out-of-domain (OOD) retrieval** lost — and does **anime-only vs mixed** training corpus rebalance that tradeoff per tag?

**Remove from main thesis:** LLM captions (B), self-training (C), tag→LLM (D), drift filtering, hard-pair mining, tag-only domain tradeoff.

**Do not use** `thesis_ft_v1` (older Grok FT, batch 128 / 5 epochs) as the main E comparison.

---

## 2. Experimental arms

| Arm | Run ID | Training corpus | Training text |
|-----|--------|-----------------|---------------|
| Pretrained CLAP | — | None | — |
| Anime-only fine-tune | `thesis_grok_only` | ACG catalog only (~65k clips) | Grok/metadata per clip (`clap_train_15s.jsonl`) |
| Mixed fine-tune | `thesis_grok_mixed` | ACG + MTAT + OpenMIC (~72k clips) | Grok on anime; mapped tag strings on MTAT/OpenMIC |

Both fine-tuned arms share **identical** optimizer, early-stopping, backbone freeze, and seeds (42, 43, 44). Only **corpus composition** differs.

**Report:** `data/eval/domain_tradeoff/REPORT.md`

---

## 3. Datasets and evaluation

| Item | Detail |
|------|--------|
| Private catalog | ~4,011 source tracks → 15s clips |
| Train clips (anime-only) | 65,041 |
| Train clips (mixed) | ~71,651 = 65,041 anime + 4,576 MTAT + 2,034 OpenMIC |
| Val clips | 7,246 (`clap_val_15s.jsonl`, Grok captions, early-stop both arms) |
| Gold eval | 200 human-labeled songs, 3 binary tags |
| Gold queries | `piano`, `vocal`, `relaxing` |
| OOD eval | Jamendo, MagnaTagATune (MTAT), OpenMIC |
| Jamendo in training | **Never** (strict OOD) |
| Index | Metadata text FAISS, IndexFlatIP, L2-normalized embeddings |
| Metric | Precision@10 and nDCG@10 (report means over seeds 42–44) |

### Gold tag prevalence (200 songs)

| Tag | Query | Positives | Prevalence |
|-----|-------|-----------|------------|
| inst_piano | piano | 29 | 14.5% |
| inst_vocal | vocal | 139 | 69.5% |
| mood_relaxing | relaxing | 76 | 38.0% |

---

## 4. Training configuration (both E arms)

| Setting | Value |
|---------|-------|
| Backbone | `music_audioset_epoch_15_esc_90.14.pt` (frozen) |
| Unfrozen | audio_projection, audio_transform, text_projection, text_transform |
| Learning rate | 0.0001 |
| Batch size | 32 |
| Max epochs | 20 |
| Early stopping | val_similarity, patience 2, min_epochs 5 |
| Temperature (contrastive) | 100 |
| Audio load | 48 kHz, mono downmix (librosa default) |
| Loudness normalization | **None** |
| Seeds | 42, 43, 44 |

Anime-only train JSONL: `clap_train_15s.jsonl`  
Mixed train JSONL: `clap_train_grok_mixed.jsonl`  
Val JSONL: `clap_val_15s.jsonl`

---

## 5. Main result — Gold set (P@10, in-domain)

Pretrained: [`retrieval_vs_random_matrix.csv`](../../data/eval/retrieval_vs_random_matrix.csv) (same gold pool). Fine-tuned: mean seeds 42–44.

| Tag | Pretrained | Anime-only FT | Mixed FT |
|-----|------------|---------------|----------|
| inst_piano | 0.20 | 0.40 | 0.40 |
| inst_vocal | 0.70 | 0.90 | 0.80 |
| mood_relaxing | 0.50 | 0.50 | 0.50 |

---

## 6. Main result — Public OOD (P@10 macro)

Macro mean over Jamendo, MTAT, OpenMIC. Fine-tuned: mean seeds 42–44.

| Tag | Pretrained | Anime-only FT | Mixed FT |
|-----|------------|---------------|----------|
| inst_piano | 0.98 | 0.83 | 0.84 |
| inst_vocal | 0.76 | 0.33 | 0.60 |
| mood_relaxing | 0.53 | 0.25 | 0.37 |

**Δ (mixed − anime-only) on OOD:** piano +0.01, vocal +0.27, relaxing +0.12.

---

## 7. OOD by dataset (P@10, mean seeds 42–44)

### Jamendo

| Tag | Anime-only | Mixed |
|-----|------------|-------|
| inst_piano | 0.63 | 0.63 |
| inst_vocal | 0.27 | 0.27 |
| mood_relaxing | 0.20 | 0.30 |

### MagnaTagATune (MTAT)

| Tag | Anime-only | Mixed |
|-----|------------|-------|
| inst_piano | 0.87 | 0.90 |
| inst_vocal | 0.33 | 0.73 |
| mood_relaxing | 0.30 | 0.43 |

### OpenMIC

| Tag | Anime-only | Mixed |
|-----|------------|-------|
| inst_piano | 1.00 | 1.00 |
| inst_vocal | 0.40 | 0.80 |
| mood_relaxing | n/a | n/a |

(OpenMIC has no mood labels.)

---

## 8. Abstract-ready findings (copy-paste)

Fine-tuning CLAP on an ACG catalog with Grok/metadata captions improves in-domain gold retrieval (e.g. vocal P@10 0.90 anime-only vs ~0.70 pretrained gold baseline) but reduces public OOD retrieval versus the pretrained model (vocal OOD macro 0.33–0.60 vs 0.76 pretrained). Comparing anime-only training (`thesis_grok_only`) to mixed training (`thesis_grok_mixed`: Grok ACG plus MTAT and OpenMIC, Jamendo held out) shows a **tag-dependent** tradeoff: mixed training improves OOD vocal (+0.27 P@10) and relaxing (+0.12) versus anime-only, with a modest in-domain vocal cost (−0.10). Piano gold is unchanged across corpus mix. **Corpus composition** materially affects the specialization–generalization balance per tag.

---

## 9. Out of scope (do not cite as main results)

| Experiment | Run ID | Note |
|------------|--------|------|
| Older Grok FT vs pretrained | `thesis_ft_v1` | Question A; different hparams |
| Grok vs LLM captions | `thesis_llm_full_llm` | Question B |
| Self-train loop | `thesis_self_v2` | Question C |
| Tag vs tag→LLM | `thesis_tag_llm` | Question D |
| Tag-only domain tradeoff | `thesis_tag_only` / `thesis_tag_mixed` | Superseded sparse-label run |

---

*Source: `data/eval/domain_tradeoff/` (job 122295). Single-file pack for Opus.*
