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

- `model/clap/music_audioset_epoch_15_esc_90.14.pt`

Make sure this file exists before CLAP indexing/evaluation commands.

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

## Human-evaluated Top-10 style retrieval workflow

This workflow is for:

- instrumentation
- mood
- energy
- texture

using Top-10 retrieval and human labels.

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
- per-type breakdown (`instrumentation`, `mood`, `energy`, `texture`)
- per-query metrics

## Legacy audio retrieval scripts

- `app/recommend.py`: OpenL3-based audio similarity flow
- `app/init_model.py`: CLAP embedding/index utilities and retrieval helpers

## Notes

- Many scripts assume relative paths under project root.
- For long-running jobs, keep outputs in `data/mapping` and resume where supported.
- If GPU is available, CLAP components may use it depending on runtime environment; BPM extraction with `librosa` is CPU-side.
