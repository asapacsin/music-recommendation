# PLAN — `20260525_clap_self_train_skeleton`

## Goal

Ship a local-only CLAP iterative self-training loop (mine hard pairs → optional refine slot → mixed JSONL → fine-tune → eval) on `clap_train_15s.jsonl`, with `--no-refine` for v1.

## Context

- Modules: `app/self_train/`, `app/train_clap_self_loop.py`, `app/init_model.py` (`init_checkpoint`)
- Guide: `docs/CLAP_SELF_TRAIN.md`, `AGENTS.md` Self-training subsection

## Inputs

- Train: `data/mapping/clap_train_15s.jsonl`
- Val: `data/mapping/clap_val_15s.jsonl`
- Backbone: `model/clap/music_audioset_epoch_15_esc_90.14.pt`
- Optional gold eval: `data/eval/gold_merged.jsonl` + metadata FAISS index

## Commands (ordered)

```bash
cd /home/mc46451/music-recommendation
export PYTHONPATH="$PWD"

# Smoke: 1 iteration, no refine
python -m app.train_clap_self_loop \
  --run-id smoke_self_v1 \
  --n-iters 1 \
  --hard-frac 0.2 \
  --seed 42 \
  --no-refine \
  --num-epochs 1 \
  --max-samples 32

# Chain: 2 iterations (iter_1 uses iter_0 checkpoint for mine + train init)
python -m app.train_clap_self_loop \
  --run-id smoke_self_v2 \
  --n-iters 2 \
  --hard-frac 0.2 \
  --seed 42 \
  --no-refine \
  --num-epochs 1 \
  --max-samples 32

# Optional gold retrieval on iter checkpoint
export RAGWEB_CLAP_CHECKPOINT=model/clap/self_train/smoke_self_v1/iter_0/best_model.pt
python -m app.data_handling.music_eval_retrieval_vs_random
```

## Expected outputs

| Output path | What should be true |
|-------------|---------------------|
| `data/self_train/<run_id>/iter_<n>/hard_mined.jsonl` | Bottom `hard_frac` rows with `sim`, `error_score` |
| `data/self_train/<run_id>/iter_<n>/train_mixed.jsonl` | All originals + duplicated hard rows |
| `model/clap/self_train/<run_id>/iter_<n>/best_model.pt` | Fine-tuned checkpoint with `model_state_dict` |
| `data/log/self_train_runs/<run_id>/iter_<n>/params.json` | Includes `init_checkpoint` when n>0 |
| `data/self_train/<run_id>/iter_<n>/iter_metrics.json` | Finite `val_mean_similarity` |
| `data/log/self_train_runs/<run_id>/summary.json` | Per-iter summaries |

## Risks / rollback

- VRAM on full embed pass: use `--max-samples` for dev smoke
- Overwriting: each `--run-id` is isolated under `data/self_train/` and `model/clap/self_train/`

## Definition of done

- [x] All expected outputs exist for smoke runs
- [x] Commands reproducible from this PLAN
- [x] Ready for REVIEW phase
