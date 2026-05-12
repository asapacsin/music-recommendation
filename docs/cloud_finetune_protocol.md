# Cloud / multi-seed CLAP fine-tune protocol

Academic-style checklist for running contrastive fine-tune in [`app/init_model.py`](../app/init_model.py) on a remote machine and reporting numbers with **multiple RNG seeds**.

**Hands-on walkthrough (commands, Windows + bash):** [`FINE_TUNING_TUTORIAL.md`](FINE_TUNING_TUTORIAL.md).

## Primary evaluation tags (mandatory to log)

Per project decision ([`AGENTS.md`](../AGENTS.md), [`docs/mark_class.txt`](mark_class.txt)), headline retrieval metrics should include **`inst_piano`**, **`inst_vocal`**, **`mood_relaxing`** (rows in `retrieval_vs_random_matrix.csv` for the matching style queries). Full matrix export is optional.

## Before you start

1. **Reproducible environment** — Record Python version, `torch` + CUDA version, OS, GPU model, and `pip freeze` (or lockfile) in the run folder.
2. **Git state** — `git rev-parse HEAD` and whether the working tree was clean; paste into run notes / thesis appendix.
3. **Data layout** — Same relative paths as local (`data/music_db`, `data/mapping`, …) or document `BASE_DIR` overrides.
4. **Frozen splits** — Train/val JSONLs and gold files must be **identical across seeds**; only initialization (and any shuffling you add later) should differ.

## Training

1. **Multi-seed driver**

   ```bash
   python -m app.train_clap_multiseed --n-seeds 5 --base-seed 42 --run-id my_experiment_v1
   ```

   Optional merged hyperparameters / overrides:

   ```bash
   python -m app.train_clap_multiseed --seeds 42,43,44 --params-json path/to/overrides.json
   ```

2. **Artifacts per run** — Under `data/log/finetune_runs/<run_id>/`:

   - `seed_<n>/best_model.pt` — PyTorch dict including `seed`, `epoch`, `model_state_dict`, …
   - `seed_<n>/params.json` — full params passed to `model_creation`
   - `seed_<n>/metrics.jsonl` — one JSON line per epoch: `seed`, `epoch`, `loss`, `similarity` (train-time mean diagonal similarity)
   - `summary.json` — `per_seed` results plus `best_similarity_mean` and `best_similarity_stdev` over finite values

3. **Determinism** — `set_seed` sets `cudnn.deterministic=True` and `benchmark=False` (slower but more stable). Residual nondeterminism on GPU is normal; report **mean ± std over seeds**, not a single run.

4. **Single seed / local** — You can still call `model_creation(params)` from `app/main.py`; pass `"seed": 42` in `params` and optional `"metrics_path"`.

## Evaluation after each seed

Eval scripts load CLAP via `config.settings.CLAP_MODEL_FILE`. To evaluate a **fine-tuned** checkpoint without editing code:

```bash
export RAGWEB_CLAP_CHECKPOINT=/path/to/data/log/finetune_runs/<run_id>/seed_42/best_model.pt
python -m app.data_handling.music_eval_retrieval_vs_random --top-k 10 20
```

Unset the variable to restore the default pretrained checkpoint path.

**Note:** `best_model.pt` stores `model_state_dict` for the inner core only. Eval scripts use [`app/clap_eval_load.py`](../app/clap_eval_load.py): load the public backbone from `CLAP_PRETRAINED_BACKBONE_FILE`, then overlay those weights when `RAGWEB_CLAP_CHECKPOINT` points at a fine-tuned save.

## What to put in a thesis table (minimal)

| Column | Source |
|--------|--------|
| Seed(s) | `summary.json` → `seeds` |
| Mean ± std train similarity | `summary.json` → `best_similarity_mean`, `best_similarity_stdev` |
| Retrieval (piano / vocal / relaxing) | Per-seed CSV rows from `music_eval_retrieval_vs_random` after setting `RAGWEB_CLAP_CHECKPOINT` |
| Optional | Full matrix, tempo rows, tempo `--include-tempo` global columns |

## Out of scope (document only)

- Replacing `mock_path_list()` in `model_creation` with manifest-driven paths (separate data task).
- Rebuilding metadata FAISS per seed (expensive); decide whether the thesis fixes the index and only updates CLAP embeddings via a documented eval path.
