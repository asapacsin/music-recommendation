# Thesis experiment questions (human reference)

**Read this first** when you need to remember what A / B / C / D mean, what each run trains, and where results live.  
**Live status:** [`docs/PROGRESS.md`](PROGRESS.md) — refresh: `bash scripts/refresh_progress.sh` — guide: [`docs/PROGRESS_MONITOR.md`](PROGRESS_MONITOR.md).  
Agents: also see [`AGENTS.md`](../AGENTS.md) and [`state/agent_state.json`](../state/agent_state.json).

**Primary evaluation tags (all questions that report retrieval):** `inst_piano`, `inst_vocal`, `mood_relaxing` — queries *piano*, *vocal*, *relaxing*.

**Three different “eval” ideas (do not mix):**

| When | What is measured | Data |
|------|------------------|------|
| **During training** | `val_similarity` — audio matches paired caption on held-out clips | `clap_val_15s.jsonl` |
| **In-domain thesis retrieval** | P@K, nDCG@K vs random on **human gold** songs | `gold_merged.jsonl` ∩ FAISS pool |
| **Public OOD test** (optional) | Same metrics on Jamendo / MTAT / OpenMIC | External manifests (post-train only) |

---

## Overview

| ID | Question in plain language | Compared arms | Main result file | Status |
|----|---------------------------|---------------|------------------|--------|
| **A** | Does **fine-tuning** beat **pretrained** CLAP on tag retrieval? | pretrained vs `thesis_ft_v1` | `data/eval/retrieval_vs_random_matrix*.csv`, FT logs | **Done** |
| **B** | Do **LLM-refined Grok captions** beat **original Grok** captions (same FT recipe)? | `thesis_llm_ablation_orig` vs `thesis_llm_ablation_llm` (sparse); `thesis_llm_full_llm` vs orig (full corpus) | `data/eval/llm_ablation/REPORT.md`, `data/eval/llm_full_ablation/REPORT.md` | **Done** (mostly null vs Grok) |
| **C** | Does **iterative self-train** (mine → LLM → FT) help? | iter 0 vs iter 1 checkpoints | `model/clap/self_train/...`, run docs | **Done** (negative: iter 1 regressed) |
| **D** | Does **tag→LLM** training text beat **tag-only** text on tag retrieval? | `thesis_tag_only` vs `thesis_tag_llm` | `data/eval/tag_llm_ablation/REPORT.md` | **Done** |
| **E** | Is OOD drop **forgetting** or **specialization**? (anime-only vs mixed FT, 2×2, Grok captions) | `thesis_grok_only` vs `thesis_grok_mixed` | `data/eval/domain_tradeoff/REPORT.md` | **Done** (job 122295) |

**Public OOD** (Jamendo / MTAT / OpenMIC) is **not** question A–D. It tests checkpoints **after** training on external audio. Question **E** adds mixed-domain training to interpret OOD vs in-domain jointly. See [Public OOD test](#public-ood-test-post-train) and [Question E](#question-e--forgetting-vs-specialization-mixed-domain-2×2).

---

## Question A — Fine-tune vs pretrained

**Question:** On our library, does supervised CLAP fine-tune improve retrieval of piano / vocal / relaxing vs the AudioSet backbone alone?

| | Detail |
|--|--------|
| **Train text** | Normal clip captions (`clap_train_15s.jsonl` — Grok/metadata style) |
| **Run ID** | `thesis_ft_v1` (multi-seed) |
| **Baseline** | Pretrained backbone, no `RAGWEB_CLAP_CHECKPOINT` |
| **Eval** | Gold retrieval vs random, metadata FAISS |
| **Commands** | `sbatch scripts/sbatch_clap_finetune.sh`; eval `sbatch scripts/sbatch_clap_retrieval_eval.sh` |
| **Outputs** | `model/clap/finetune/thesis_ft_v1/seed_*/best_model.pt`, `data/log/finetune_runs/thesis_ft_v1/summary.json` |

**Do not confuse with B** (caption rewrite) or **D** (tag strings as training text).

---

## Question B — Grok captions vs LLM-refined captions

**Question:** If we keep the same FT setup but replace training captions with **one LLM rewrite per song** (full corpus), do we beat training on **original Grok** text?

| | Detail |
|--|--------|
| **Train text** | Full **Grok/metadata** captions vs **LLM-expanded** captions (not tag-only strings) |
| **Run IDs** | Sparse: `thesis_llm_ablation_orig` / `thesis_llm_ablation_llm`. Full: orig reuses sparse orig; LLM arm `thesis_llm_full_llm` |
| **Eval** | Per-checkpoint metadata + caption FAISS; gold pool |
| **Commands** | `sbatch scripts/sbatch_llm_caption_ablation.sh`; full: `sbatch_llm_full_corpus_gen.sh` then `SKIP_LLM_GEN=1 sbatch_llm_full_ablation.sh` |
| **Outputs** | `data/eval/llm_full_ablation/REPORT.md` (headline), `data/eval/llm_ablation/REPORT.md` (sparse pilot) |

**Finding (2026-06):** Full-corpus LLM rewrite did **not** beat Grok on primary tags (metadata index); relaxing mixed on caption index. See [`docs/RESEARCH_DIRECTIONS.md`](RESEARCH_DIRECTIONS.md).

**Do not confuse with D** — B changes **rich Grok captions**, not short **tag** training strings.

---

## Question C — Self-train loop

**Question:** Does the self-training loop (hard mining → LLM refine → mixed JSONL → FT) improve over a single FT pass?

| | Detail |
|--|--------|
| **Driver** | `python -m app.train_clap_self_loop` |
| **Run ID** | e.g. `thesis_self_v2` under `model/clap/self_train/` |
| **Eval** | Iter metrics + optional gold eval per iter |
| **Docs** | [`docs/CLAP_SELF_TRAIN.md`](CLAP_SELF_TRAIN.md), [`docs/agent_runs/20260526_self_train_v2/`](agent_runs/20260526_self_train_v2/) |

**Finding:** Iter 1 **regressed** vs iter 0 best checkpoint — treat as negative result unless re-run.

---

## Question D — Tag-only vs tag→LLM training text

**Question:** On the **full 15s train corpus**, does training on **Llama-expanded tag phrases** beat training on **short tag strings** (piano / vocal / relaxing from gold; `"music"` fallback elsewhere) for **the same** gold tag retrieval?

| | Detail |
|--|--------|
| **Two models** | (1) **Tag-only** `thesis_tag_only` ← `clap_train_tag.jsonl`. (2) **Tag→LLM** `thesis_tag_llm` ← `clap_train_tag_llm.jsonl` |
| **Same** | Backbone, `train_params.json`, val early-stop on `clap_val_15s.jsonl`, 3 seeds |
| **Different** | Only the **text** paired with each clip during contrastive FT |
| **Eval** | Symmetric per-checkpoint FAISS (metadata + caption index); gold pool; **not** Jamendo |
| **Job 1** | `sbatch scripts/sbatch_tag_llm_corpus_gen.sh` — song-level Llama on tag strings |
| **Job 2** | `SKIP_LLM_GEN=1 sbatch scripts/sbatch_tag_llm_ablation.sh` — cache, FT both arms, eval, report |
| **Outputs** | Checkpoints under `model/clap/finetune/thesis_tag_only|thesis_tag_llm/`. Report: **`data/eval/tag_llm_ablation/REPORT.md`** |
| **Agent run** | [`docs/agent_runs/20260601_tag_train_llm_ablation/`](agent_runs/20260601_tag_train_llm_ablation/) |

**Status:** Report at **`data/eval/tag_llm_ablation/REPORT.md`**.

**Do not confuse with B** — D uses **tag-derived** training text, not full Grok caption replacement.

---

## Question E — Forgetting vs specialization (mixed-domain 2×2)

**Question:** When anime-only fine-tuning hurts public OOD, is that **catastrophic forgetting** (recoverable with mixed training) or **specialization** (in-domain gain at OOD cost)?

| | Detail |
|--|--------|
| **Arms** | (1) **Anime-only** `thesis_grok_only` ← `clap_train_15s.jsonl` (Grok/metadata per clip). (2) **Mixed** `thesis_grok_mixed` ← `clap_train_grok_mixed.jsonl` (Grok anime + MTAT/OpenMIC tag strings; Jamendo **never** in train) |
| **Training text** | Grok/metadata on all ACG clips; public clips use mapped dataset tags |
| **Seeds** | 42, 43, 44 |
| **2×2 eval** | Rows: anime-only vs mixed. Cols: in-domain gold vs public OOD (Jamendo + MTAT + OpenMIC) |
| **Build mixed JSONL** | `python -m app.data_handling.music_build_mixed_domain_train_jsonl --anime-jsonl data/mapping/clap_train_15s.jsonl --out-jsonl data/mapping/clap_train_grok_mixed.jsonl` |
| **Full pipeline** | `sbatch scripts/sbatch_domain_tradeoff_ablation.sh` |
| **Outputs** | `model/clap/finetune/thesis_grok_{only,mixed}/seed_*/best_model.pt`, **`data/eval/domain_tradeoff/REPORT.md`** |
| **Guide** | [`docs/DOMAIN_TRADEOFF.md`](DOMAIN_TRADEOFF.md) |
| **Agent run** | [`docs/agent_runs/20260619_grok_domain_tradeoff/`](agent_runs/20260619_grok_domain_tradeoff/) |

**Status:** **Done** — Slurm job **122295**. Report: **`data/eval/domain_tradeoff/REPORT.md`**.

**Finding:** Grok-caption FT lifts in-domain gold (vocal 0.90 anime-only) but OOD remains below pretrained. Mixed training improves OOD vocal (+0.27) and relaxing (+0.12) vs anime-only with small or no gold cost on piano/relaxing; vocal gold −0.10 with mixed.

**Superseded:** Tag-only domain tradeoff (`thesis_tag_only` / `thesis_tag_mixed`) — do not use for Question E.

---

## Public OOD test (post-train)

**Not a letter question.** After any checkpoint exists, ask: *Do these weights generalize on external datasets?*

| Dataset | Manifest | Test script |
|---------|----------|-------------|
| Jamendo five-tag | `data/eval/jamendo_five_tag_manifest.jsonl` | `bash scripts/run_public_eval.sh` |
| MagnaTagATune | `data/eval/mtat_manifest.jsonl` | same (`DATASETS=...`) |
| OpenMIC | `data/eval/openmic_manifest.jsonl` | same (piano + vocal only) |

**Outputs:** `data/eval/{jamendo,mtat,openmic}_public/`, combined `data/eval/REPORT.md`.  
**Does not train** anything; set `ARMS` to include `thesis_tag_only`, `thesis_tag_llm` after question **D** finishes.

---

## Quick “which report do I open?”

| You care about… | Open |
|-----------------|------|
| FT vs pretrained | `data/eval/retrieval_vs_random_matrix.csv` + `thesis_ft_v1` logs |
| Grok vs LLM captions | `data/eval/llm_full_ablation/REPORT.md` |
| Self-train | `docs/agent_runs/20260526_self_train_v2/STATE.md` |
| Tag vs tag→LLM | `data/eval/tag_llm_ablation/REPORT.md` |
| Forgetting vs specialization (2×2) | `data/eval/domain_tradeoff/REPORT.md` |
| Public sets | `data/eval/REPORT.md` *(public combined)* |

---

## Related docs

- Tag scope: [`docs/class_selected.txt`](class_selected.txt)
- Follow-up ideas: [`docs/RESEARCH_DIRECTIONS.md`](RESEARCH_DIRECTIONS.md)
- Fine-tune how-to: [`docs/FINE_TUNING_TUTORIAL.md`](FINE_TUNING_TUTORIAL.md)
- Operator commands: [`docs/OPERATIONS.md`](OPERATIONS.md)
- Gold merge: [`docs/README_eval_merge.md`](README_eval_merge.md)
