# PLAN — `20260601_tag_train_llm_ablation`

## Goal

Thesis question **D** — full spec: [`docs/THESIS_QUESTIONS.md`](../../THESIS_QUESTIONS.md#question-d--tag-only-vs-tagllm-training-text).

Does **tag→LLM** training text beat **tag-only** strings on human-gold tag retrieval (inst_piano, inst_vocal, mood_relaxing @ K=10)?

## Context

- Modules: `music_build_tag_train_jsonl`, `music_refine_tag_captions`, `music_eval_tag_llm_ablation_report`
- Scripts: `run_tag_llm_ablation.sh`, `sbatch_tag_llm_corpus_gen.sh`, `sbatch_tag_llm_ablation.sh`
- Distinct from question **B** (Grok vs LLM caption rewrite): [`docs/RESEARCH_DIRECTIONS.md`](../../RESEARCH_DIRECTIONS.md)

## Design

- **Full corpus** (~65k clips): primary-tag text from `gold_merged.jsonl` join on `source_path`; unlabeled clips → fallback `"music"`.
- **Tag-only arm:** `clap_train_tag.jsonl` → FT `thesis_tag_only`
- **Tag→LLM arm:** Llama expand tag strings per song → `clap_train_tag_llm.jsonl` → FT `thesis_tag_llm`
- **Eval:** symmetric per-checkpoint metadata + caption FAISS; same queries as other ablations.

## Commands

```bash
cd ~/music-recommendation
conda activate ragweb

# Build tag JSONL
python -m app.data_handling.music_build_tag_train_jsonl

# Job 1 — LLM expand tags (resumable)
sbatch scripts/sbatch_tag_llm_corpus_gen.sh

# Job 2 — cache + FT + eval + report
SKIP_LLM_GEN=1 sbatch scripts/sbatch_tag_llm_ablation.sh

# Report only
SKIP_BUILD=1 SKIP_LLM_GEN=1 SKIP_MERGE=1 SKIP_CACHE=1 SKIP_TRAIN=1 SKIP_EVAL=1 \
  bash scripts/run_tag_llm_ablation.sh
```

## Expected outputs

| Path | Check |
|------|--------|
| `data/mapping/clap_train_tag.jsonl` | 65k rows; counts in build summary |
| `data/mapping/clap_train_tag_llm.jsonl` | LLM text on all songs |
| `model/clap/finetune/thesis_tag_only/seed_*/best_model.pt` | 3 seeds |
| `model/clap/finetune/thesis_tag_llm/seed_*/best_model.pt` | 3 seeds |
| `data/eval/tag_llm_ablation/REPORT.md` | Primary-tag table |

## Risks

- Most clips use fallback `"music"` (~200 songs have human gold) — interpret as tag supervision at scale, not gold-only training.
- Job 1 LLM gen is long (one call per unique song).

## Definition of done

- [ ] Cluster jobs complete
- [ ] `REPORT.md` reviewed vs question D claim
- [ ] `REVIEW.md` PASS/FAIL
