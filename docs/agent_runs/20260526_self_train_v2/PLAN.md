# PLAN — `20260526_self_train_v2`

## Goal

Full-library 15s CLAP manifests + self-train v2 (LLM refine + CLAP gate + val early stopping) for thesis B4 runs.

## Context

- Modules: `app/train_clap_self_loop.py`, `app/self_train/{mine,refine,gate,manifest,eval_iter}.py`, `app/init_model.py`, `app/llm_local.py`
- Guide: `docs/CLAP_SELF_TRAIN.md`, `AGENTS.md`

## Inputs

- `data/music_db/` (~4k tracks)
- `model/clap/music_audioset_epoch_15_esc_90.14.pt`
- `model/llama3.1-8b-instruct/` (4 safetensor shards)
- Human gold + FAISS for `--run-gold-eval` (unchanged size)

## Commands (ordered)

```bash
cd ~/music-recommendation
conda activate ragweb
export PYTHONPATH="$PWD"

# Phase A — full 15s (screen recommended)
python -m app.data_handling.music_split_to_15s
python -m app.data_handling.music_build_train_val_from_15s

# Phase C — smoke
python -m app.train_clap_self_loop \
  --run-id smoke_self_v2_refine \
  --n-iters 1 --refine --max-samples 32 \
  --num-epochs 2 --refine-max-hard 8

# Phase C — thesis B4 (Slurm)
# RUN_ID=thesis_self_v2 REFINE=1 RUN_GOLD_EVAL=1 sbatch scripts/sbatch_clap_self_train.sh
```

## Expected outputs

| Output path | What should be true |
|-------------|---------------------|
| `data/mapping/clap_split_summary.json` | `group_count_sources` >> 44 |
| `data/mapping/clap_train_15s.jsonl` | Full train manifest |
| `app/self_train/gate.py` | CLAP gate helpers |
| `model/clap/self_train/<run_id>/iter_*/best_model.pt` | Checkpoints |
| `data/log/self_train_runs/<run_id>/summary.json` | Per-iter val + refine stats |

## Risks / rollback

- 15s split long / disk-heavy → run in `screen`
- LLM+CLAP VRAM → `RAGWEB_LLM_4BIT=1`, unload LLM after refine
- Pilot manifests preserved until build overwrites JSONL

## Definition of done

- [ ] Full 15s manifests built
- [ ] v2 code merged; tests pass
- [ ] Smoke `--refine` completes
- [ ] REVIEW.md PASS
