# CLAP fine-tuning tutorial (local build)

Step-by-step guide to train with **multiple seeds**, save checkpoints, run **retrieval eval** on each checkpoint, and assemble results for a thesis-style report.

**Related:** checklist and theory notes in [`cloud_finetune_protocol.md`](cloud_finetune_protocol.md). **Primary tags to report:** `inst_piano`, `inst_vocal`, `mood_relaxing` — see [`mark_class.txt`](mark_class.txt) and [`AGENTS.md`](../AGENTS.md).

---

## 0. Prerequisites

1. **Project root** — Open a terminal at the repo root (folder that contains `app/`, `config/`, `data/`).

2. **Python env** — Activate the same conda/venv you use for the rest of Ragweb (GPU recommended).

3. **LAION CLAP backbone** — File must exist (path in `config/settings.py`):

   - `CLAP_PRETRAINED_BACKBONE_FILE` → default `model/clap/music_audioset_epoch_15_esc_90.14.pt`

4. **Training audio** — `model_creation` reads **`data/mapping/clap_train_15s.jsonl`** (`audio_path` → 15s clips, `text` captions). Build with `python -m app.data_handling.music_build_train_val_from_15s` after 15s segments exist under `data/music_db_15s/`. Legacy fallback: `--use-music-db-fallback` globs `data/music_db/`.

5. **Retrieval eval (after fine-tune)** — Merged gold + metadata FAISS index as in the main README (`gold_merged.jsonl`, `metadata_text` index built). Not required until you reach section 3.

---

## 1. Multi-seed training (recommended)

Pick a **run id** (folder name) and number of seeds (thesis suggestion: **5**).

**Windows (cmd):**

```bat
cd /d E:\Ragweb
python -m app.train_clap_multiseed --n-seeds 5 --base-seed 42 --run-id thesis_ft_v1
```

**PowerShell:**

```powershell
Set-Location E:\Ragweb
python -m app.train_clap_multiseed --n-seeds 5 --base-seed 42 --run-id thesis_ft_v1
```

**Explicit seed list instead of a range:**

```bash
python -m app.train_clap_multiseed --seeds 42,43,44,45,46 --run-id thesis_ft_v1
```

**Optional:** override hyperparameters with a JSON file (merged on top of defaults):

```bash
python -m app.train_clap_multiseed --n-seeds 5 --base-seed 42 --run-id thesis_ft_v1 --params-json path\to\overrides.json
```

### What you get

Under `data/log/finetune_runs/thesis_ft_v1/` (replace with your `--run-id`):

| Path | Contents |
|------|----------|
| `seed_42/best_model.pt` | Fine-tuned **core** weights (+ `seed`, `epoch`, …) |
| `seed_42/params.json` | Full `params` used for that run |
| `seed_42/metrics.jsonl` | One JSON line per epoch: `loss`, `similarity` |
| … | Same for each seed |
| `summary.json` | All seeds + **mean** / **stdev** of best train-time `similarity` |

If training fails for one seed, fix the error and re-run with a **new** `--run-id` so you do not mix partial runs.

---

## 2. Single-seed training (quick test)

From repo root:

```bash
python -m app.main
```

Uses `app/main.py` defaults (one seed **42** unless you edit `params` there). Checkpoint goes to `model/best_model.pt` unless you change `save_path` in `params`.

---

## 3. Evaluate each fine-tuned checkpoint (retrieval matrix)

Eval scripts load CLAP via `app/clap_eval_load.load_clap_module_httsat()`. If **`RAGWEB_CLAP_CHECKPOINT`** points at a `best_model.pt` produced by fine-tune, the loader applies **`model_state_dict`** on top of the public backbone.

**Windows cmd (one seed):**

```bat
set RAGWEB_CLAP_CHECKPOINT=E:\Ragweb\data\log\finetune_runs\thesis_ft_v1\seed_42\best_model.pt
cd /d E:\Ragweb
python -m app.data_handling.music_eval_retrieval_vs_random --top-k 10 20 --out-csv data\eval\retrieval_matrix_seed42.csv
set RAGWEB_CLAP_CHECKPOINT=
```

Repeat for `seed_43`, … with a **different** `--out-csv` each time so files are not overwritten.

**PowerShell:**

```powershell
$env:RAGWEB_CLAP_CHECKPOINT = "E:\Ragweb\data\log\finetune_runs\thesis_ft_v1\seed_42\best_model.pt"
Set-Location E:\Ragweb
python -m app.data_handling.music_eval_retrieval_vs_random --top-k 10 20 --out-csv data/eval/retrieval_matrix_seed42.csv
Remove-Item Env:RAGWEB_CLAP_CHECKPOINT
```

**Thesis table:** filter rows for **piano music**, **vocal music**, **relaxing music** (and optional tempo rows) from each CSV; compare across seeds (mean ± std or one column per seed).

**Other evals** (tempo zero-shot, Top-K prep, metadata FAISS build): same `RAGWEB_CLAP_CHECKPOINT` pattern before the usual `python -m …` command.

---

## 4. Build a “performance sheet”

There is **no** single command that merges five retrieval CSVs automatically. Practical flow:

1. **Training curve / best train similarity** — Open `summary.json` and each `metrics.jsonl` (Excel or Python).
2. **Retrieval** — Collect `retrieval_matrix_seed*.csv` from step 3; align rows by `query_text` + `top_k`.
3. **Document** — Save `git rev-parse HEAD`, GPU name, and `pip freeze` in the same folder as the run.

---

## 5. Defaults reference

| Item | Default |
|------|---------|
| Seeds in driver | `--n-seeds 3`, `--base-seed 42` → seeds 42,43,44 |
| `model_creation` seed if omitted | `42` |
| Backbone for training | `CLAP_PRETRAINED_BACKBONE_FILE` (not overridden by `RAGWEB_CLAP_CHECKPOINT`) |

---

## 6. Troubleshooting

| Problem | What to check |
|---------|----------------|
| `FileNotFoundError` for CLAP | Backbone `.pt` under `model/clap/` |
| CUDA OOM | Lower `batch_size` in `--params-json` |
| Retrieval run uses wrong weights | `echo %RAGWEB_CLAP_CHECKPOINT%` (cmd) or `$env:RAGWEB_CLAP_CHECKPOINT` (PowerShell) |
| `load` errors on `best_model.pt` | File must be from **this** repo’s `model_creation` (`model_state_dict` key); see `app/clap_eval_load.py` |

---

## 7. Next steps (code / data improvements)

- Point training at **train JSONL** clips instead of all of `data/music_db/` (requires editing `init_model.py` or a new data loader).
- Add a small script that merges N retrieval CSVs + `summary.json` into one Excel-ready CSV (optional future work).
