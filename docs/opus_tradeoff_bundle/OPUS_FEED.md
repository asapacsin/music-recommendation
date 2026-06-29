# Thesis data pack — Question E only (paste this entire file into Opus)

**Purpose:** Revise the thesis to focus on the **specialization–generalization tradeoff** only.  
**Rule for Opus:** Use only numbers in this document. Do not invent metrics.

---

## 1. What Opus should do

Rewrite the thesis around **one question**:

> When CLAP is fine-tuned on ACG music, how much **in-domain gold retrieval** is gained versus **public out-of-domain (OOD) retrieval** lost — and does **anime-only vs mixed** training corpus rebalance that tradeoff per tag?

**Remove from main thesis:** LLM captions (B), self-training (C), tag→LLM (D), drift filtering, hard-pair mining.  
**Do not use** `thesis_ft_v1` (Grok-caption fine-tune) as the main comparison — that is a different experiment.

---

## 2. Experimental arms (only these three)

| Arm | Run ID | Training corpus | Training text |
|-----|--------|-----------------|---------------|
| Pretrained CLAP | — | None | — |
| Anime-only fine-tune | `thesis_tag_only` | ACG catalog only (~65k clips) | Tag-only strings |
| Mixed fine-tune | `thesis_tag_mixed` | ACG + MTAT + OpenMIC (~72k clips) | Tag-only strings |

Both fine-tuned arms share **identical** optimizer, early-stopping, backbone freeze, and seeds (42, 43, 44). Only **corpus composition** differs.

---

## 3. Datasets and evaluation

| Item | Detail |
|------|--------|
| Private catalog | ~4,011 source tracks → 15s clips |
| Train clips (anime-only) | 65,041 |
| Train clips (mixed) | 71,651 = 65,041 anime + 4,576 MTAT + 2,034 OpenMIC |
| Val clips | 7,246 (`clap_val_15s.jsonl`, early-stop both arms) |
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

Anime-only train JSONL: `clap_train_tag.jsonl`  
Mixed train JSONL: `clap_train_tag_mixed.jsonl`  
Val JSONL: `clap_val_15s.jsonl`

---

## 5. Main result — Table 5.1 Specialization–generalization (P@10)

Mean over seeds 42–44. **Primary thesis table.**

| Tag | Pretrained OOD | Anime-only Gold | Mixed Gold | Δ Gold | Anime-only OOD | Mixed OOD | Δ OOD |
|-----|----------------|-----------------|------------|--------|----------------|-----------|-------|
| inst_piano | 0.98 | 0.20 | 0.30 | +0.10 | 0.70 | 0.69 | −0.01 |
| inst_vocal | 0.76 | 1.00 | 0.90 | −0.10 | 0.37 | 0.53 | +0.17 |
| mood_relaxing | 0.53 | 0.50 | 0.50 | 0.00 | 0.28 | 0.40 | +0.12 |

**Δ columns** = mixed minus anime-only.

**Interpretation (one sentence per row):**
- Piano: mixed helps in-domain (+0.10 gold); OOD unchanged.
- Vocal: anime-only best in-domain (ceiling 1.00); mixed recovers OOD (+0.17) at cost of in-domain (−0.10).
- Relaxing: gold tied; mixed improves OOD (+0.12).

**Overall:** Fine-tuning specializes (OOD far below pretrained 0.98/0.76/0.53). Mixed corpus **redistributes** the cost by tag — no single arm wins everywhere.

---

## 6. Main result — nDCG@10 (gold, in-domain)

Same seeds; identical across seeds for each arm in this run.

| Tag | Pretrained Gold | Anime-only Gold | Mixed Gold |
|-----|-----------------|-----------------|------------|
| inst_piano | 0.212 | 0.290 | 0.423 |
| inst_vocal | 0.556 | 1.000 | 0.927 |
| mood_relaxing | 0.543 | 0.423 | 0.414 |

Pretrained gold from zero-shot eval on same 200-song pool. Anime-only / mixed from `thesis_tag_only` / `thesis_tag_mixed` checkpoints.

---

## 7. OOD by dataset (P@10, mean seeds 42–44)

### Jamendo

| Tag | Anime-only | Mixed |
|-----|------------|-------|
| inst_piano | 0.70 | 0.53 |
| inst_vocal | 0.30 | 0.13 |
| mood_relaxing | 0.23 | 0.47 |

### MagnaTagATune (MTAT)

| Tag | Anime-only | Mixed |
|-----|------------|-------|
| inst_piano | 0.40 | 0.53 |
| inst_vocal | 0.80 | 0.97 |
| mood_relaxing | 0.33 | 0.33 |

### OpenMIC

| Tag | Anime-only | Mixed |
|-----|------------|-------|
| inst_piano | 1.00 | 1.00 |
| inst_vocal | 0.00 | 0.50 |
| mood_relaxing | n/a | n/a |

(OpenMIC has no mood labels.)

### Pretrained OOD reference (macro mean, same three datasets)

| Tag | Pretrained |
|-----|------------|
| inst_piano | 0.98 |
| inst_vocal | 0.76 |
| mood_relaxing | 0.53 |

---

## 8. Abstract-ready findings (copy-paste)

Fine-tuning CLAP on an ACG catalog under tag-only supervision improves some in-domain gold retrieval but sharply reduces public OOD retrieval versus the pretrained model. Comparing anime-only training (`thesis_tag_only`) to mixed training (`thesis_tag_mixed`: anime plus MTAT and OpenMIC, Jamendo held out) shows a **tag-dependent** tradeoff: mixed training improves OOD vocal (+0.17 P@10) and relaxing (+0.12) versus anime-only, and improves in-domain piano (+0.10), but reduces in-domain vocal (−0.10). No training regime dominates on every tag in both domains. **Corpus composition is a first-class design choice** for specialized music retrieval.

---

## 9. Thesis edits checklist

**Add**
- Arm mapping table (Section 2 above)
- Table 5.1 (Section 5) + nDCG table (Section 6)
- OOD-by-dataset table (Section 7)
- Gold prevalence table (Section 3)
- Hyperparameter table (Section 4)
- Preprocessing note: mono at load, no loudness norm

**Remove or appendix**
- Ch 4–5: LLM refinement, drift gate, hard mining, self-training loop
- Table 5.1 old version using `thesis_ft_v1` (Grok captions)
- Bootstrap CI claims (not computed)
- Triple-duplicated §3.6.1 paragraph
- Sentences claiming self-training “progressively improves” retrieval

**Fix**
- Do not conflate `thesis_ft_v1` with `thesis_tag_only`
- Vocal pretrained **gold** P@10 is 0.70 (not 1.00); ceiling 1.00 is after anime-only FT only

---

## 10. Out of scope (do not cite as main results)

| Experiment | Run ID | Note |
|------------|--------|------|
| Grok-caption FT vs pretrained | `thesis_ft_v1` | Question A; different training text |
| Grok vs LLM captions | `thesis_llm_full_llm` | Question B; null on gold |
| Self-train loop | `thesis_self_v2` | Question C; iter 1 regressed |
| Tag vs tag→LLM | `thesis_tag_llm` | Question D; not needed for E |

---

*Generated from repo eval: `data/eval/domain_tradeoff/`, gold CSVs, finetune params. Single-file pack for Opus — no other files required.*
