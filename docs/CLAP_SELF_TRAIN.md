# CLAP self-training loop (local 15s data)

Iterative loop: **mine hard pairs → LLM refine + CLAP gate → mixed JSONL → val early-stop fine-tune → eval**.

## Prerequisites

- Full-library manifests: `python -m app.data_handling.music_split_to_15s` then `music_build_train_val_from_15s`
- `data/mapping/clap_train_15s.jsonl`, `clap_val_15s.jsonl`
- `model/clap/music_audioset_epoch_15_esc_90.14.pt`
- `model/llama3.1-8b-instruct/` (see `docs/LLM_LOCAL.md`; `pip install accelerate`)
- GPU recommended

## Quick start (v2 — LLM refine)

```bash
cd ~/music-recommendation
conda activate ragweb
export PYTHONPATH="$PWD"
export RAGWEB_LLM_4BIT=1

python -m app.train_clap_self_loop \
  --run-id thesis_self_v2 \
  --n-iters 2 \
  --hard-frac 0.2 \
  --refine \
  --run-gold-eval
```

## Ablation flags

| Run | Command |
|-----|---------|
| B3 mining only | `--no-refine` |
| B4 full v2 | `--refine` |

## Dev smoke

```bash
python -m app.train_clap_self_loop \
  --run-id smoke_self_v2_refine \
  --n-iters 1 \
  --refine \
  --max-samples 32 \
  --refine-max-hard 8 \
  --num-epochs 2
```

## v2 behavior

- **`LlmRefiner`:** Llama 3.1 rewrites hard-row captions; **CLAP gate** (`app/self_train/gate.py`) requires sim gain + `min_text_cos` (default 0.85, env `RAGWEB_REFINE_MIN_TEXT_COS`).
- **Inner train:** `model_creation` uses **`val_jsonl`**; saves **best val** checkpoint; patience early stop (default 2).
- **Outer loop:** stops if val mean similarity plateaus across iters (`--min-val-delta`, `--outer-patience-iters`; disable with `--no-outer-early-stop`).

## Key flags

- `--refine` / `--no-refine` (default: no-refine for backward compat)
- `--gate-min-text-cos`, `--gate-min-sim-gain`
- `--llm-max-new-tokens`, `--refine-max-hard`
- `--early-stop-patience`, `--num-epochs` (max epochs, default 20)
- `--run-gold-eval`, `--max-samples`

## Outputs

| Path | Content |
|------|---------|
| `data/self_train/<run_id>/iter_<n>/hard_mined.jsonl` | Bottom `hard_frac` by diagonal similarity |
| `data/self_train/<run_id>/iter_<n>/refined.jsonl` | Gate-passed LLM captions |
| `data/self_train/<run_id>/iter_<n>/train_mixed.jsonl` | Full train + hard duplicates (LLM text) |
| `model/clap/self_train/<run_id>/iter_<n>/best_model.pt` | Best **val** checkpoint |
| `data/log/self_train_runs/<run_id>/summary.json` | Run-level summary |

## Slurm

```bash
RUN_ID=thesis_self_v2 REFINE=1 RUN_GOLD_EVAL=1 sbatch scripts/sbatch_clap_self_train.sh
```

Defaults in `scripts/sbatch_clap_self_train.sh`: `#SBATCH --mem=256G`, `BATCH_SIZE=32`, `MIN_EPOCHS=5`, `MIN_EPOCHS_BEFORE_OUTER=5`.

Env: `RUN_ID`, `N_ITERS`, `REFINE`, `RUN_GOLD_EVAL`, `RAGWEB_LLM_4BIT`, `MAX_SAMPLES`, `BATCH_SIZE`, `NUM_EPOCHS`, `EARLY_STOP_PATIENCE`, `MIN_EPOCHS`, `MIN_EPOCHS_BEFORE_OUTER`.

CLI: `--min-epochs`, `--min-epochs-before-outer`, `--batch-size`. Training OOM fix: batched train-sim in `model_creation` (no full-manifest embed per epoch).

## Agent run

`docs/agent_runs/20260526_self_train_v2/`
