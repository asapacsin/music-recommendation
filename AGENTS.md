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

**Target labels (project decision):** fine-tune and primary reporting on **three** multihot classes only — **`inst_piano`**, **`inst_vocal`**, **`mood_relaxing`**. Gold retrieval-vs-random showed these as the most stable vs metadata text; other template columns may stay labeled for reference but are **out of scope for default fine-tune** unless explicitly reopened.

- **Code:** `app/init_model.py` — freeze backbone; optionally train `audio_projection`, `audio_transform`, `text_projection`, `text_transform` via `params['unfreeze_layers']`; contrastive loss (cross-entropy on scaled audio–text similarity); Adam; optional early stopping on mean diagonal similarity.
- **Data:** `app/data_handling/music_split_to_15s.py`, `music_build_train_val_from_15s.py` → `data/mapping/clap_train_15s.jsonl`, `clap_val_15s.jsonl`.
- **Checkpoint path:** `config/settings.py` → `CLAP_MODEL_FILE` / `BEST_MODEL_FILE` as applicable.

### Cloud / multi-seed fine-tune

- **Driver:** `python -m app.train_clap_multiseed` — loops seeds, writes `data/log/finetune_runs/<run_id>/seed_<n>/` (`best_model.pt`, `params.json`, `metrics.jsonl`) and run-level `summary.json` (**mean ± std** of best train-time similarity).
- **Seeding:** `model_creation` calls `set_seed` from `params["seed"]` (default `42`) and logs per-epoch metrics to `metrics.jsonl`. Training always initializes from **`CLAP_PRETRAINED_BACKBONE_FILE`** (unaffected by `RAGWEB_CLAP_CHECKPOINT`).
- **Eval on fine-tuned weights:** set env **`RAGWEB_CLAP_CHECKPOINT`** to a seed’s `best_model.pt`, then run retrieval / tempo eval as usual (`CLAP_MODEL_FILE` in settings becomes that path).
- **Report:** prioritize retrieval rows for **`inst_piano`**, **`inst_vocal`**, **`mood_relaxing`** per seed, then aggregate (mean ± std or table per seed). Full protocol: [`docs/cloud_finetune_protocol.md`](docs/cloud_finetune_protocol.md).

## Two different “zero-shot” flows (do not confuse)

| Goal | Module | What it does |
|------|--------|----------------|
| **Tempo** classification vs BPM pseudo-labels | `app/data_handling/music_eval_zeroshot_tempo.py` (+ song-level `music_eval_zeroshot_tempo_song.py`) | Fixed prompts: slow / mid-tempo / fast. Inputs: val JSONL with `audio_path`. |
| **Style / tag queries** → metadata retrieval | `app/metadata_faiss.py` (build index) + `app/data_handling/music_eval_topk_prepare.py` | CLAP text embedding of each query string → FAISS search over **metadata text** index. |

**Audio–text index (different use case):** `app/init_model.py` — embeds audio + text from `music_map`; `get_top_k_by_text_query` uses **that file’s description** as the query, not arbitrary tag prompts like “vocal music.”

## Primary evaluation tags (`docs/class_selected.txt`)

That file is **documentation only** (not loaded by code). For **fine-tuning and headline metrics**, treat only **`inst_piano`**, **`inst_vocal`**, **`mood_relaxing`** as first-class (see *Fine-tuning* above). Other columns in `gold_labels_multihot_template.csv` are optional / exploratory.

Gold / template columns align with README (`gold_labels_multihot_template.csv`).

**Style retrieval defaults** already include matching `query_id`s and prompts in `app/data_handling/music_eval_topk_prepare.py` (`DEFAULT_STYLE_QUERIES`: e.g. `"vocal music"`, `"relaxing music"`). Override via `data/eval/style_queries.json` (see README “Human-evaluated Top-10 style retrieval workflow”).

**Gold vs random (matrix):** `app/data_handling.music_eval_retrieval_vs_random.py` scores each style query against the metadata FAISS index on the **gold-labeled pool** (basename join); CSV is precision + nDCG vs random (see README). By default it also adds **three tempo query rows** per K (retrieval-tuned tempo text vs BPM `tempo_bin_bpm` positives). **`--no-include-tempo-queries`** drops those rows. Use **`--include-tempo`** to append global song-level BPM vs CLAP columns on every row.

Top‑K human workflow (`music_eval_topk_prepare` / `music_eval_topk_score`) remains available for judged lists.

## Quick commands (details in README)

- Metadata FAISS: `python -m app.metadata_faiss build` / `search`
- Tempo zero-shot: `python -m app.data_handling.music_eval_zeroshot_tempo`
- Style Top‑K prep/score: `music_eval_topk_prepare`, `music_eval_topk_score`
- Gold BPM end-to-end (after CSV is labeled): build manifest filtered by sidecar → `music_eval_zeroshot_tempo_song` → `music_eval_merge_gold` → **`music_eval_export_gold_review_csv`** (minimal review CSV: `song_name`, multihot, `tempo_bin_bpm`). Optional check: `music_eval_gold_bpm_coverage --sidecar …` (pre-merge) or `--merged-jsonl …` (post-merge). See `docs/README_eval_merge.md` *Gold-only manifest*.
- Gold merge / counts: `music_eval_merge_gold`, `music_eval_gold_label_counts`
- Retrieval vs random matrix: `python -m app.data_handling.music_eval_retrieval_vs_random` (merged gold + metadata FAISS index; tempo matrix rows on by default; `--include-tempo` adds global CLAP-vs-BPM columns)
- Multi-seed CLAP fine-tune: `python -m app.train_clap_multiseed` (see `docs/cloud_finetune_protocol.md`; eval with `RAGWEB_CLAP_CHECKPOINT`)

## After a Cursor reset

1. Read this file + `README.md`.
2. For tag choices and caveats, read `docs/class_selected.txt`.
3. For gold merge semantics, read `docs/README_eval_merge.md`.
4. For structured agent work, read `docs/multi_agent_workflow.md` and open any in-progress `docs/agent_runs/<run_id>/` folder.
