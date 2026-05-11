# Agent / session context (read after Cursor reset)

Chat history is not stored in this repo. **This file is the durable snapshot** of decisions and entry points. Update it when workflow or priorities change.

## Multi-agent workflow (Plan / Execute / Review)

For non-trivial work, use **artifact-based handoffs** so sessions survive resets:

- Guide: [`docs/multi_agent_workflow.md`](docs/multi_agent_workflow.md)
- Per-run folders: `docs/agent_runs/<run_id>/` with `PLAN.md`, `RUNLOG.md`, `REVIEW.md` (see [`docs/agent_runs/README.md`](docs/agent_runs/README.md))
- Templates: [`docs/templates/`](docs/templates/) (`agent_plan`, `agent_runlog`, `agent_review`)
- Cursor reminder: [`.cursor/rules/ragweb_agent_pipeline.mdc`](.cursor/rules/ragweb_agent_pipeline.mdc)

Default is to **commit** PLAN/RUNLOG/REVIEW for thesis traceability; optional `.gitignore` of noisy logs is documented in `docs/agent_runs/README.md`.

## Project in one line

Local music metadata pipeline + CLAP: 15s clips, train/val JSONL, FAISS retrieval, tempo zero-shot eval, human multihot gold, style Top‑K retrieval.

## Fine-tuning (CLAP)

- **Code:** `app/init_model.py` — freeze backbone; optionally train `audio_projection`, `audio_transform`, `text_projection`, `text_transform` via `params['unfreeze_layers']`; contrastive loss (cross-entropy on scaled audio–text similarity); Adam; optional early stopping on mean diagonal similarity.
- **Data:** `app/data_handling/music_split_to_15s.py`, `music_build_train_val_from_15s.py` → `data/mapping/clap_train_15s.jsonl`, `clap_val_15s.jsonl`.
- **Checkpoint path:** `config/settings.py` → `CLAP_MODEL_FILE` / `BEST_MODEL_FILE` as applicable.

## Two different “zero-shot” flows (do not confuse)

| Goal | Module | What it does |
|------|--------|----------------|
| **Tempo** classification vs BPM pseudo-labels | `app/data_handling/music_eval_zeroshot_tempo.py` (+ song-level `music_eval_zeroshot_tempo_song.py`) | Fixed prompts: slow / mid-tempo / fast. Inputs: val JSONL with `audio_path`. |
| **Style / tag queries** → metadata retrieval | `app/metadata_faiss.py` (build index) + `app/data_handling/music_eval_topk_prepare.py` | CLAP text embedding of each query string → FAISS search over **metadata text** index. |

**Audio–text index (different use case):** `app/init_model.py` — embeds audio + text from `music_map`; `get_top_k_by_text_query` uses **that file’s description** as the query, not arbitrary tag prompts like “vocal music.”

## Primary evaluation tags (`docs/class_selected.txt`)

That file is **documentation only** (not loaded by code). Recommended primary tags:

- `inst_vocal` — strong prevalence; watch imbalance.
- `mood_relaxing` — best-supported mood.

Gold / template columns align with README (`gold_labels_multihot_template.csv`): `inst_vocal`, `mood_relaxing`, etc.

**Style retrieval defaults** already include matching `query_id`s and prompts in `app/data_handling/music_eval_topk_prepare.py` (`DEFAULT_STYLE_QUERIES`: e.g. `"vocal music"`, `"relaxing music"`). Override via `data/eval/style_queries.json` (see README “Human-evaluated Top-10 style retrieval workflow”).

**Gold vs random (matrix):** `app/data_handling.music_eval_retrieval_vs_random.py` scores each style query against the metadata FAISS index on the **gold-labeled pool** (basename join); CSV is precision + nDCG vs random (see README). CLAP uses the query string with trailing ` music` stripped. Use **`--include-tempo`** to append song-level BPM vs CLAP zero-shot metrics from `gold_merged.jsonl` (`program_tempo`) on the same export.

Top‑K human workflow (`music_eval_topk_prepare` / `music_eval_topk_score`) remains available for judged lists.

## Quick commands (details in README)

- Metadata FAISS: `python -m app.metadata_faiss build` / `search`
- Tempo zero-shot: `python -m app.data_handling.music_eval_zeroshot_tempo`
- Style Top‑K prep/score: `music_eval_topk_prepare`, `music_eval_topk_score`
- Gold merge / counts: `music_eval_merge_gold`, `music_eval_gold_label_counts`
- Retrieval vs random matrix: `python -m app.data_handling.music_eval_retrieval_vs_random` (merged gold + metadata FAISS index; add `--include-tempo` after song tempo eval + merge)

## After a Cursor reset

1. Read this file + `README.md`.
2. For tag choices and caveats, read `docs/class_selected.txt`.
3. For gold merge semantics, read `docs/README_eval_merge.md`.
4. For structured agent work, read `docs/multi_agent_workflow.md` and open any in-progress `docs/agent_runs/<run_id>/` folder.
