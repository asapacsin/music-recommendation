# Research directions — post full-corpus LLM experiment review

Living note for thesis discussion and follow-up work. **Review after** `data/eval/llm_full_ablation/REPORT.md` exists.

**Human-readable map of questions A–D:** [`docs/THESIS_QUESTIONS.md`](THESIS_QUESTIONS.md) (status, commands, outputs — read that first).

Last updated: 2026-06-01

---

## Current experiment (in flight)

| Item | Path / command |
|------|----------------|
| Job 1 — LLM song captions | `sbatch scripts/sbatch_llm_full_corpus_gen.sh` |
| Job 2 — cache + FT + symmetric eval | `SKIP_LLM_GEN=1 sbatch scripts/sbatch_llm_full_ablation.sh` |
| Outputs | `data/eval/llm_full_ablation/REPORT.md`, `summary_by_index.csv` |
| Prior sparse ablation (2.8%) | `data/eval/llm_ablation/REPORT.md` |
| Main FT positive result | `thesis_ft_v1` — pretrained vs fine-tuned |

### Review checklist (after Job 2)

- [ ] `clap_train_llm_full.jsonl` complete (~65k rows, all songs in progress JSONL)
- [ ] `thesis_llm_full_llm` checkpoints (seeds 42–43–44)
- [ ] **Metadata index** table: orig vs LLM @ piano / vocal / relaxing
- [ ] **Caption index** table: same tags
- [ ] Compare to sparse ablation (`llm_ablation/REPORT.md`)
- [ ] Write scoped claim (see § Claims below)

---

## Four thesis questions (keep separate)

See **[`docs/THESIS_QUESTIONS.md`](THESIS_QUESTIONS.md)** for full wording, commands, and output paths.

| ID | Question | Status |
|----|----------|--------|
| **A** | Does **supervised FT** beat **pretrained** CLAP on tag retrieval? | **Done** — `thesis_ft_v1` |
| **B** | Do **LLM captions** beat **Grok** when fine-tuning (matched recipe)? | **Done** — null; `data/eval/llm_full_ablation/REPORT.md` |
| **C** | Does **iterative self-train** help? | **Negative** — iter 1 regressed |
| **D** | Does **tag→LLM** beat **tag-only** training text on tag retrieval? | **Impl done** — run `sbatch scripts/sbatch_tag_llm_corpus_gen.sh` then Job 2 |

Do **not** merge A, B, and D in one table without noting different training text (Grok captions vs tag strings vs tag→LLM).

---

## Direction 1 — Text data: rewrite vs augmentation

### Full-corpus rewrite (what we run now)

- One LLM caption per **song** → propagated to all 15s clips
- **Replaces** Grok text in train JSONL
- Tests: “Is one **better** single description enough?”

### Caption augmentation (C-CLAPA-lite; not run)

- **Same audio**, **multiple** captions (original + 2+ LLM paraphrases)
- **Data enhancement**: richer wording, same meaning
- Stored as **extra JSONL rows** (same `audio_path`, different `text`)
- Trained with **same** diagonal CLAP loss (see `app/init_model.py`)

**Thesis line:** Prior work often wins with **multi-view text diversity**; we test **single-view full rewrite** only.

### Optional follow-up

- [ ] Build `clap_train_llm_aug.jsonl` (3 rows per clip: orig + 2 paraphrases)
- [ ] Fine-tune one arm; compare to full rewrite and Grok baseline
- [ ] Consider **one random paraphrase per epoch** to avoid same-batch paraphrase negatives

---

## Direction 2 — Extra contrastive views & loss (T-CLAP style; not run)

**Not** the same as paraphrase augmentation.

| | Paraphrase aug | T-CLAP-style extra views |
|--|----------------|---------------------------|
| Text | Same meaning, different words | **Wrong** structure (e.g. event **order** flipped) |
| Role | Extra **positives** | **Hard synthetic negatives** for same audio |
| Loss | Standard CLAP only | CLAP + **auxiliary temporal/contrast term** |

Example:

```text
Audio: piano then drums
Positive:  "Calm piano followed by steady drums."
Negative:  "Steady drums followed by calm piano."   ← synthetic temporal negative
L_total = L_CLAP + λ · L_temporal
```

**Thesis line:** Literature gains may need **structured negatives + extra loss**, not caption polish alone. Our null LLM result does not rule that out.

### Optional follow-up

- [ ] LLM generate order-flipped captions for subset of clips
- [ ] Add simple margin loss: `sim(audio, pos) > sim(audio, neg)`
- [ ] Re-eval tag retrieval (may matter less for piano/vocal/relaxing tags than for AudioCaps)

---

## Direction 3 — Other auxiliary objectives (C-CLAPA / BALSa; not run)

| Method | Idea |
|--------|------|
| **Caption decoder** | Reconstruct caption from embedding → regularization |
| **BALSa** | LLM synthetic QA / alignment data from tags/metadata |
| **GRL / modality alignment** | Reduce audio–text gap in embedding space |

Bundled with augmentation in many papers — cite as **“richer training package”** vs our **standard contrastive FT**.

---

## Direction 4 — Index & deployment alignment

| Item | What | Status |
|------|------|--------|
| **Fixed pretrained FAISS** | Query FT, old doc vectors | Old sparse ablation |
| **Per-checkpoint metadata FAISS** | Re-embed Grok catalog with each `best_model.pt` | **New pipeline** |
| **Caption-index eval** | Search train JSONL text, same checkpoint | **New pipeline** |
| **LLM metadata catalog** | LLM-rewrite **index** text (not only train JSONL) | Not run |

### Interpretation matrix (fill after experiment)

| Metadata index | Caption index | Discussion |
|----------------|---------------|------------|
| Flat | Flat | Rewrite-only + standard loss insufficient for tags |
| Flat | LLM ↑ | Train text improved; **catalog mismatch** at inference |
| LLM ↑ | LLM ↑ | LLM helps when index matches training text |
| LLM ↑ | Flat | Unusual — check index build |

### Optional follow-up

- [ ] Rebuild metadata index using **LLM captions** as indexed text (deployment experiment)
- [ ] Apply same symmetric rebuild to `thesis_ft_v1` eval for consistency

---

## Direction 5 — Where to spend LLM budget

| Strategy | Tested? | Notes |
|----------|---------|-------|
| Sparse hard-clip swap (~2.8%) | Yes | Mixed/null — `llm_ablation/` |
| Full-corpus tag-aware rewrite | **Running** | `llm_full_ablation/` |
| Dose curve (0 / 25 / 100% LLM rows) | No | Shows “more LLM ≠ better tags” |
| Confidence-filtered refine | No | LLM only on low-confidence metadata |
| Self-train iter 1 | Yes | Regressed vs iter 0 |

---

## Direction 6 — Query-side (no retrain)

| Tool | Path |
|------|------|
| Style queries | `data/eval/style_queries.json` |
| Composite query ablation | `scripts/sbatch_composite_query_ablation.sh` |
| Query expansion | Not implemented — e.g. `"piano"` → `"piano instrumental keyboard"` |

Improves retrieval without new LLM training captions.

---

## Direction 7 — Audio side

| Item | Status |
|------|--------|
| Audio backbone cache | **Implemented** — `music_precompute_clap_audio_cache` |
| Audio augmentation (noise, crop, mixup) | Not run |
| Hard audio negatives | Partially via self-train mining |

---

## Direction 8 — Evaluation & generalization

| Item | Status |
|------|--------|
| Gold pool ~200 songs, 3 primary tags | Done |
| Multi-seed mean ± std | 3 seeds ablation; 5 seeds `thesis_ft_v1` |
| Public OOD (Jamendo five-tag) | Download tooling; eval pipeline not wired |
| Human Top-K workflow | `music_eval_topk_prepare` / `score` |
| Statistical tests | Informal thresholds (|Δ| > 0.01); optional paired tests |

---

## Claims — robust wording

### Safe after full-corpus + symmetric index

> Full-corpus, tag-oriented LLM caption refinement under matched fine-tuning and per-checkpoint FAISS rebuild did **not** reliably improve human-gold tag retrieval @10 over original-caption fine-tuning on [metadata index / both indexes — fill from REPORT].

### vs literature (do not overclaim)

> Prior work reports LLM gains on **caption-level** audio–text retrieval with **text augmentation**, **synthetic contrastive views**, and **auxiliary losses**. Our study isolates **single-view caption replacement** and **standard CLAP contrastive fine-tuning** with **tag-query retrieval over a metadata catalog** — a deployment-oriented setting where those gains **did not transfer** [if null].

### Main positive (unchanged)

> Supervised fine-tuning on clip–caption pairs improves tag-style retrieval versus the pretrained CLAP model (`thesis_ft_v1`).

### After tag vs tag→LLM ablation (question D)

> Full-corpus training on primary-tag strings (gold join + `"music"` fallback for unlabeled clips), compared to Llama-expanded tag captions under matched fine-tuning, [did / did not] improve human-gold tag retrieval @10 — fill from `data/eval/tag_llm_ablation/REPORT.md`.

---

## Priority after this experiment

| Priority | Direction | Effort |
|----------|-----------|--------|
| 1 | **Write up** full + sparse LLM + FT main result | Writing |
| 2 | Fill **interpretation matrix** (metadata vs caption index) | 1 h reading |
| 3 | **C-CLAPA-lite** augmentation arm | Medium compute |
| 4 | **Jamendo OOD** eval | Medium impl |
| 5 | T-CLAP-style temporal negatives + extra loss | High impl |
| 6 | LLM metadata catalog index | Medium |

---

## Key file references

| Topic | File |
|-------|------|
| Full ablation orchestrator | `scripts/run_llm_full_ablation.sh` |
| LLM song refine | `app/data_handling/music_refine_full_corpus_captions.py` |
| Tag train JSONL | `app/data_handling/music_build_tag_train_jsonl.py` |
| Tag → LLM refine | `app/data_handling/music_refine_tag_captions.py` |
| Tag ablation orchestrator | `scripts/run_tag_llm_ablation.sh` |
| Tag ablation report | `app/data_handling/music_eval_tag_llm_ablation_report.py` |
| Checkpoint-aware FAISS | `app/data_handling/music_build_retrieval_faiss.py` |
| Full ablation report | `app/data_handling/music_eval_llm_full_ablation_report.py` |
| CLAP training loss | `app/init_model.py` (`model_creation`) |
| Sparse ablation report | `data/eval/llm_ablation/REPORT.md` |
| Agent entry point | `AGENTS.md` |
