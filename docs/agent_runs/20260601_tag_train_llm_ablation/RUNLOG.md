# RUNLOG — `20260601_tag_train_llm_ablation`

## 2026-06-01 — Implementation

- Added `app/data_handling/music_build_tag_train_jsonl.py`
- Added `app/data_handling/music_refine_tag_captions.py`
- Added `app/data_handling/music_eval_tag_llm_ablation_report.py`
- Added `scripts/run_tag_llm_ablation.sh`, `sbatch_tag_llm_ablation.sh`, `sbatch_tag_llm_corpus_gen.sh`
- Fixed `run_llm_full_ablation.sh` report step (`_SEEDS_CSV` + `run_llm_full_ablation_report_only.sh`)
- Tests: `tests/test_build_tag_train_jsonl.py` — 3 passed (ragweb env)

## Cluster (pending)

```bash
sbatch scripts/sbatch_tag_llm_corpus_gen.sh
SKIP_LLM_GEN=1 sbatch scripts/sbatch_tag_llm_ablation.sh
```
