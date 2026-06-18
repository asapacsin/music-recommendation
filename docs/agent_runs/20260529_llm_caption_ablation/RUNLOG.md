# RUNLOG — `20260529_llm_caption_ablation`

## 2026-05-29 — Implement pipeline (code only)

- Added `app/data_handling/music_build_llm_train_jsonl.py`
- Added `app/data_handling/music_eval_llm_caption_ablation_report.py`
- Added `data/eval/llm_ablation/train_params.json`
- Added `scripts/run_llm_caption_ablation.sh`, `scripts/sbatch_llm_caption_ablation.sh`
- Added `tests/test_build_llm_train_jsonl.py`
- Smoke: build JSONL → `n_total=65041`, `n_replaced=1817` (exit 0)
- Unit tests: `pytest tests/test_build_llm_train_jsonl.py` — pending user Slurm FT/eval run

## Next (user)

```bash
cd ~/music-recommendation
sbatch scripts/sbatch_llm_caption_ablation.sh
```
