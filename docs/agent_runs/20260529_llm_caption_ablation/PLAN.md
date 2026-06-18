# PLAN — `20260529_llm_caption_ablation`

## Goal

One-command pipeline: original vs LLM-swapped caption fine-tune (3 seeds), gold retrieval eval, primary-tag report.

## Context

- Modules: `app/data_handling/music_build_llm_train_jsonl.py`, `train_clap_multiseed`, `music_eval_llm_caption_ablation_report.py`
- Scripts: `scripts/run_llm_caption_ablation.sh`, `scripts/sbatch_llm_caption_ablation.sh`
- Refined source: `data/self_train/thesis_self_v2/iter_0/refined.jsonl` (~1,817 gate-passed swaps)

## Commands

```bash
cd ~/music-recommendation
sbatch scripts/sbatch_llm_caption_ablation.sh
# eval + report only after checkpoints exist:
SKIP_TRAIN=1 sbatch scripts/sbatch_llm_caption_ablation.sh
```

## Expected outputs

| Output path | What should be true |
|-------------|---------------------|
| `data/mapping/clap_train_llm_gated_iter0.jsonl` | 65041 rows, ~1817 LLM text swaps |
| `model/clap/finetune/thesis_llm_ablation_orig/seed_*/best_model.pt` | Original-caption FT |
| `model/clap/finetune/thesis_llm_ablation_llm/seed_*/best_model.pt` | LLM-swapped-caption FT |
| `data/eval/llm_ablation/orig_seed*.csv`, `llm_seed*.csv` | Per-seed matrices |
| `data/eval/llm_ablation/summary_primary.csv`, `REPORT.md` | Thesis comparison |

## Definition of done

- [ ] Pipeline code merged; unit test passes
- [ ] Slurm run completes (user-submitted)
- [ ] REVIEW.md PASS after outputs verified
