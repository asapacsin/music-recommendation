# Music Retrieval and CLAP Data Pipeline

This repository contains a local music retrieval workflow built around:

- metadata extraction from filenames,
- confidence routing and merge utilities,
- 15-second audio segmentation for CLAP fine-tuning,
- FAISS indexing for retrieval,
- and zero-shot tempo evaluation.

## What is included

- `app/data_handling/music_extract_metadata.py`
  - async metadata extraction to `data/mapping/music_metadata.json`
  - supports incremental update, rebuild, confidence filtering, batching, and checkpoint saves
- `app/data_handling/music_metadata_evaluate_confidence.py`
  - confidence triage (`human_pass_way.json`) and optional high-confidence export
- `app/data_handling/music_metadata_merge_process_meta.py`
  - merge process metadata with confidence threshold + human override
- `app/data_handling/music_split_to_15s.py`
  - split source audio into 15s clips and update `music_15s_map.json`
- `app/data_handling/music_build_train_val_from_15s.py`
  - grouped-by-source train/val manifests for CLAP:
    - `clap_train_15s.jsonl`
    - `clap_val_15s.jsonl`
    - `clap_split_summary.json`
- `app/metadata_faiss.py`
  - build and query FAISS text index from metadata rows
- `app/data_handling/music_eval_zeroshot_tempo.py`
  - BPM-derived pseudo labels + CLAP zero-shot tempo evaluation
  - supports resume/checkpoint for large datasets
- `app/data_handling/music_eval_build_song_manifest.py` / `music_eval_zeroshot_tempo_song.py` / `music_eval_prepare_gold_multihot_csv.py` / `music_eval_merge_gold.py` / `music_eval_export_gold_review_csv.py` / `music_eval_upgrade_gold_csv.py` / `music_eval_gold_label_counts.py` / `music_eval_gold_bpm_coverage.py`
  - song-level eval manifest, human gold sheet, merged `gold_merged.jsonl`, safe CSV column upgrades (see `docs/README_eval_merge.md`)
- `app/train_clap_multiseed.py` — multi-seed CLAP fine-tune driver (`python -m app.train_clap_multiseed`); protocol in **`docs/cloud_finetune_protocol.md`**

## Setup

### 1) Python environment

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Environment variables

Create `.env` in project root (if not already present):

```env
XAI_API_KEY=your_key
XAI_MODEL=grok-4.20-non-reasoning
XAI_BASE_URL=https://api.x.ai/v1
# Optional:
# XAI_REASONING_MODEL=...
```

### 3) Required local model file

The CLAP checkpoint path is configured in `config/settings.py`:

- **`CLAP_PRETRAINED_BACKBONE_FILE`** — public LAION weights used for **training** (`init_model.load_original_model`) and as the base when loading fine-tuned cores for eval.
- **`CLAP_MODEL_FILE`** — defaults to the same path; overridden when **`RAGWEB_CLAP_CHECKPOINT`** is set (fine-tuned `best_model.pt` for retrieval/tempo eval).

Make sure the backbone `.pt` exists under `model/clap/` before CLAP indexing/evaluation commands.

**Fine-tuned checkpoint for eval:** set **`RAGWEB_CLAP_CHECKPOINT`** to an absolute path of a saved `best_model.pt` (e.g. from `data/log/finetune_runs/.../`). **Step-by-step fine-tune + eval:** [`docs/FINE_TUNING_TUTORIAL.md`](docs/FINE_TUNING_TUTORIAL.md). Checklist / thesis notes: [`docs/cloud_finetune_protocol.md`](docs/cloud_finetune_protocol.md).

## Directory conventions

- Source music: `data/music_db`
- 15s segments: `data/music_db_15s`
- Mappings/metadata/manifests: `data/mapping`
- FAISS files: `data/index`

## End-to-end workflow

### Step A: Extract metadata from filenames

Incremental update (default behavior):

```bash
python app/data_handling/music_extract_metadata.py
```

Use reasoning model + update only low-confidence rows:

```bash
python app/data_handling/music_extract_metadata.py --reasoning --confidence 0.35
```

Force rebuild over union of existing JSON + disk files:

```bash
python app/data_handling/music_extract_metadata.py --rebuild
```

### Step B: Confidence routing

Create human review queue and optionally collect high-confidence rows:

```bash
python app/data_handling/music_metadata_evaluate_confidence.py
python app/data_handling/music_metadata_evaluate_confidence.py --collect-high --high-threshold 0.35
```

### Step C: Merge process metadata

```bash
python app/data_handling/music_metadata_merge_process_meta.py --music-confidence-min 0.7
```

### Step D: Split to 15-second clips

This script updates mapping with checkpoint-safe behavior:

```bash
python app/data_handling/music_split_to_15s.py
```

### Step E: Build CLAP train/val manifests

```bash
python app/data_handling/music_build_train_val_from_15s.py
```

## Metadata FAISS index module

Build metadata text index:

```bash
python -m app.metadata_faiss build --min-confidence 0.35
```

Search metadata index:

```bash
python -m app.metadata_faiss search --query "melancholic piano ballad" --top-k 5
```

Default outputs:

- `data/index/metadata_text_index.faiss`
- `data/mapping/metadata_id_mapping.json`

## Zero-shot tempo evaluation (before fine-tuning)

### Clip-level (15s rows in `clap_val_15s.jsonl`)

Run zero-shot tempo eval on validation manifest:

```bash
python -m app.data_handling.music_eval_zeroshot_tempo
```

Useful options for large datasets:

```bash
python -m app.data_handling.music_eval_zeroshot_tempo --save-every 50
python -m app.data_handling.music_eval_zeroshot_tempo --resume
python -m app.data_handling.music_eval_zeroshot_tempo --overwrite
```

Outputs:

- `data/mapping/tempo_eval_predictions.jsonl`
- `data/mapping/tempo_eval_metrics.json`

Evaluation logic:

- BPM estimated from audio (librosa) -> pseudo label:
  - `slow` `< 80`
  - `mid-tempo` `80-120`
  - `fast` `> 120`
- CLAP zero-shot prompt classification:
  - `"a slow tempo music track"`
  - `"a mid-tempo music track"`
  - `"a fast tempo music track"`

### Song-level (K clips per source track)

Build a manifest with up to **3** clips per song (first / middle / last segment in `music_15s_map.json`):

```bash
# All split songs
python -m app.data_handling.music_eval_build_song_manifest

# Only val songs — full pool (can be hundreds of tracks)
python -m app.data_handling.music_eval_build_song_manifest \
  --filter-val-jsonl data/mapping/clap_val_15s.jsonl

# Gold labeling: random ~150 songs from val (recommended workload)
python -m app.data_handling.music_eval_build_song_manifest \
  --filter-val-jsonl data/mapping/clap_val_15s.jsonl \
  --random-sample 150 --seed 42

# After the gold CSV exists: manifest = only labeled songs (sidecar paths)
python -m app.data_handling.music_eval_build_song_manifest \
  --filter-gold-sidecar data/eval/gold_labels_multihot_template.csv.sidecar.jsonl \
  --out data/eval/gold_tempo_manifest.jsonl
```

Run tempo eval aggregated per **song** (majority vote over clips; ties broken by mean BPM / mean CLAP scores):

```bash
python -m app.data_handling.music_eval_zeroshot_tempo_song
```

If you built `gold_tempo_manifest.jsonl` above, point the song eval at it (and optionally a separate `--pred-output` so you do not overwrite a full-pool run):

```bash
python -m app.data_handling.music_eval_zeroshot_tempo_song \
  --manifest data/eval/gold_tempo_manifest.jsonl \
  --pred-output data/eval/tempo_eval_song_predictions.jsonl
```

Check sidecar vs tempo ledger before merge (`--strict` fails CI-style if anything is missing):

```bash
python -m app.data_handling.music_eval_gold_bpm_coverage \
  --sidecar data/eval/gold_labels_multihot_template.csv.sidecar.jsonl \
  --tempo-jsonl data/eval/tempo_eval_song_predictions.jsonl
```

Outputs:

- `data/eval/song_eval_manifest.jsonl`
- `data/eval/tempo_eval_song_predictions.jsonl`
- `data/eval/tempo_eval_song_metrics.json`

Manifest script options: `--max-songs`, `--random-sample`, `--seed`.  
Song tempo eval options: `--resume` / `--overwrite`, `--save-every`.

### Human gold set (multi-hot 0/1 per class)

Minimal Excel sheet: **filename only** + label columns (no full path, no clip list, no uncertain/notes). Tempo remains program/BPM. Labels follow `docs/music_style.txt`: **instrumentation** (piano / orchestral / vocal) and **mood/character** (sad/melancholic, relaxing, dark/tense, elegant, epic).

After you have `song_eval_manifest.jsonl`:

```bash
python -m app.data_handling.music_eval_prepare_gold_multihot_csv
```

Outputs:

- `data/eval/gold_labels_multihot_template.csv` — columns: `song_name`, then multi-hot columns (all start at **0**)
- `data/eval/gold_labels_multihot_template.csv.sidecar.jsonl` — same row order: `{song_name, source_path}` for merging or disambiguating duplicate filenames (`--no-sidecar` to omit)

Label columns: `inst_piano`, `inst_orchestral`, `inst_vocal`, `mood_sad_melancholic`, `mood_relaxing`, `mood_dark_tense`, `mood_exciting`, `mood_elegant`, `mood_epic` — use **0** or **1**.

**Already labeling?** After taxonomy changes, upgrade in place (backs up `.bak`, preserves row order for sidecar):

```bash
python -m app.data_handling.music_eval_upgrade_gold_csv --csv data/eval/gold_labels_multihot_template.csv --in-place
```

Or write to a new file:

```bash
python -m app.data_handling.music_eval_upgrade_gold_csv --csv data/eval/gold_labels_multihot_template.csv --out data/eval/gold_labels_upgraded.csv
```

**UTF-8 with BOM** by default for Excel (`--no-bom` to disable).

### Merge gold labels + program tempo (+ metadata)

After human labeling and **`music_eval_zeroshot_tempo_song`** (same manifest):

```bash
python -m app.data_handling.music_eval_merge_gold
```

Writes **`data/eval/gold_merged.jsonl`** for downstream val/test. Full paths and ordering are documented in **`docs/README_eval_merge.md`**.

Minimal spreadsheet for human review (`song_name`, multihot columns, **`tempo_bin_bpm`** from program BPM):

```bash
python -m app.data_handling.music_eval_export_gold_review_csv
```

Writes **`data/eval/gold_merged_review.csv`** (UTF-8 BOM for Excel). Re-running overwrites the file; use **`--out-csv`** to write elsewhere if you edited the CSV by hand.

Per-class prevalence (positive counts per tag):

```bash
python -m app.data_handling.music_eval_gold_label_counts
```

Use **`--csv data/eval/gold_labels_multihot_template.csv`** instead of the default merged JSONL.

## Human-evaluated Top-10 style retrieval workflow

This workflow is for **instrumentation**, **mood**, and **style** (elegant / epic) queries — Top-10 retrieval and human labels. See `docs/music_style.txt` and `data/eval/style_queries.json`.

### 1) Prepare style query set

Default query file:

- `data/eval/style_queries.json`

You can regenerate defaults:

```bash
python -m app.data_handling.music_eval_topk_prepare --init-queries --overwrite-queries
```

### 2) Generate Top-10 candidates and human labeling sheet

```bash
python -m app.data_handling.music_eval_topk_prepare --top-k 10
```

Outputs:

- `data/eval/top10_candidates.jsonl` (machine records)
- `data/eval/top10_human_labels.csv` (fill `relevance`)

Human labeling guide for `relevance`:

- `0` = not relevant
- `1` = relevant
- optional graded relevance (`2` = very relevant)

### 3) Score human-labeled results

After filling `data/eval/top10_human_labels.csv`:

```bash
python -m app.data_handling.music_eval_topk_score --top-k 10
```

Output:

- `data/eval/top10_metrics.json`

Reported metrics:

- overall `precision@10`, `hitrate@10`, `ndcg@10`
- per-type breakdown (`instrumentation`, `mood`, `style`)
- per-query metrics

### 4) Automatic matrix: retrieval vs random baseline (gold multihot)

Uses **`gold_merged.jsonl`** (human multihot + `source_path`) and the **metadata text FAISS** index. The eval pool is all index rows whose metadata `audio` **basename** matches a gold song (same idea as `music_eval_merge_gold` metadata matching).

By default the matrix also includes **three tempo rows** per `top_k` (retrieval-oriented tempo phrases vs **BPM bin** `tempo_bin_bpm` as relevance; wording differs from song-level classifier prompts). Use **`--no-include-tempo-queries`** if you only want style-tag rows.

Prerequisites: build the metadata index (`python -m app.metadata_faiss build ...`) and merge gold (`python -m app.data_handling.music_eval_merge_gold`).

```bash
python -m app.data_handling.music_eval_retrieval_vs_random --top-k 10
```

Multiple cutoffs:

```bash
python -m app.data_handling.music_eval_retrieval_vs_random --top-k 5 10 20
```

**Optional: merge song-level tempo zero-shot** (BPM bin vs CLAP from `gold_merged.jsonl`’s `program_tempo`) into the same CSV/JSON. Requires that merged gold includes `tempo_bin_bpm` and `tempo_clap_zeroshot` (run **`music_eval_zeroshot_tempo_song`** on the same manifest, then **`music_eval_merge_gold`** — see *Merge gold labels*).

```bash
python -m app.data_handling.music_eval_retrieval_vs_random --top-k 5 10 20 --include-tempo
```

Outputs:

- `data/eval/retrieval_vs_random_matrix.json`
- `data/eval/retrieval_vs_random_matrix.csv`

CSV columns: `query_text` (prompt with trailing ` music` removed for CLAP where applicable), `top_k`, `n_pool`, `n_positive`, `prevalence`, `precision_at_k`, `precision_delta` (vs prevalence), `ndcg_at_k`, `ndcg_random_mean`, `ndcg_delta`. With `--include-tempo`, append **`tempo_accuracy`**, **`tempo_macro_f1`**, **`tempo_n_songs`** (same values on every row; songs = unique pool basenames with valid `program_tempo`). JSON has the same rows under `rows` plus `meta` (paths, pool size, seeds, skipped query ids, `include_tempo_queries`; with `--include-tempo`, `meta.tempo` includes accuracy, macro_f1, confusion matrix, per-class breakdown).

**Baseline semantics (thesis):** `precision_delta` / `ndcg_delta` compare retrieval ranking to a **random** baseline on the same gold pool — large deltas are expected and are **not** the main claim. The primary comparison is **pretrained CLAP vs fine-tuned CLAP** on the **same** metadata FAISS index and gold pool (`--no-include-tempo-queries` for headline style rows). Re-run pretrained eval without `RAGWEB_CLAP_CHECKPOINT`; fine-tuned eval with `RAGWEB_CLAP_CHECKPOINT` pointing at each `seed_*/best_model.pt` (see [`docs/FINE_TUNING_TUTORIAL.md`](docs/FINE_TUNING_TUTORIAL.md) §3).

## Thesis results (headline numbers)

Primary tags: **`inst_piano`**, **`inst_vocal`**, **`mood_relaxing`** (queries `piano music`, `vocal music`, `relaxing music` in the retrieval matrix). Training run: `thesis_ft_v1` — artifacts under `data/log/finetune_runs/thesis_ft_v1/` (`summary.json`, `seed_*/best_model.pt`, `metrics.jsonl`). Pretrained retrieval: `data/eval/retrieval_vs_random_matrix.csv` (no `RAGWEB_CLAP_CHECKPOINT`). Fine-tuned retrieval @K=10: same eval command with checkpoint set; **metrics identical across seeds 42–46** for the three headline queries.

### Training (RQ2)

| Metric | Value |
|--------|-------|
| Seeds | 42, 43, 44, 45, 46 |
| Best train similarity (per seed) | ~0.357–0.360 |
| Mean ± std | **0.3590 ± 0.0013** |
| Best epoch | 5 (all seeds) |

Source: `data/log/finetune_runs/thesis_ft_v1/summary.json`.

### Retrieval @K=10 — fine-tuned (all five seeds: same)

| Query (tag) | Positives / pool | precision@10 | nDCG@10 | Δ nDCG vs random |
|-------------|------------------|--------------|---------|------------------|
| piano | 29 / 200 | 0.30 | 0.428 | +0.288 |
| vocal | 139 / 200 | 1.00 | 1.00 | +0.303 |
| relaxing | 76 / 200 | 0.60 | 0.652 | +0.275 |

### Retrieval @K=10 — pretrained vs fine-tuned (RQ1)

| Query | Pretrained prec@10 / nDCG@10 | Fine-tuned prec@10 / nDCG@10 |
|-------|------------------------------|------------------------------|
| piano | 0.20 / 0.359 | 0.30 / 0.428 |
| vocal | 1.00 / 1.00 | 1.00 / 1.00 |
| relaxing | 0.50 / 0.537 | 0.60 / 0.652 |

Pretrained rows from `data/eval/retrieval_vs_random_matrix.csv` (`top_k=10`). Confirm fine-tuned numbers on cluster (E3) with `RAGWEB_CLAP_CHECKPOINT` per seed if reproducing.

**Interpretation:** Fine-tuning yields modest but consistent gains on **piano** and **relaxing**; **vocal** is already at ceiling with pretrained CLAP on this pool. **Orchestral**, **sad/melancholic**, **tempo** matrix rows, and other tags remain weak — report in appendix or a secondary table (`retrieval_vs_random_matrix.csv`, full export).

## Legacy audio retrieval scripts

- `app/recommend.py`: OpenL3-based audio similarity flow
- `app/init_model.py`: CLAP embedding/index utilities and retrieval helpers

## Notes

- Many scripts assume relative paths under project root.
- For long-running jobs, keep outputs in `data/mapping` and resume where supported.
- If GPU is available, CLAP components may use it depending on runtime environment; BPM extraction with `librosa` is CPU-side.
