# Agent / session context (read after Cursor reset)

Chat history is not stored in this repo. **This file is the durable snapshot** of decisions and entry points. Update it when workflow or priorities change.

## Agent session state (AGENT_STATE)

Per-turn working memory for autonomous agents: **[`state/agent_state.json`](state/agent_state.json)** (PLAN / CODE / TEST / REVIEW / MEMORY). Cursor rule: [`.cursor/rules/agent_state_workflow.mdc`](.cursor/rules/agent_state_workflow.mdc). Read and update each session; do not reset unless user says **RESET AGENT STATE**.

## Multi-agent workflow (Plan / Execute / Review)

For non-trivial work, use **artifact-based handoffs** so sessions survive resets:

- Guide: [`docs/multi_agent_workflow.md`](docs/multi_agent_workflow.md)
- Per-run folders: `docs/agent_runs/<run_id>/` with `PLAN.md`, `RUNLOG.md`, `REVIEW.md` (see [`docs/agent_runs/README.md`](docs/agent_runs/README.md))
- Templates: [`docs/templates/`](docs/templates/) (`agent_plan`, `agent_runlog`, `agent_review`)
- Cursor reminder: [`.cursor/rules/ragweb_agent_pipeline.mdc`](.cursor/rules/ragweb_agent_pipeline.mdc)

Default is to **commit** PLAN/RUNLOG/REVIEW for thesis traceability; optional `.gitignore` of noisy logs is documented in `docs/agent_runs/README.md`.

## Project in one line

Local music metadata pipeline + CLAP: 15s clips, train/val JSONL, FAISS retrieval, tempo zero-shot eval, human multihot gold, style Top‑K retrieval.

## Thesis questions A–D (human doc)

**[`docs/THESIS_QUESTIONS.md`](docs/THESIS_QUESTIONS.md)** — what each question tests, run IDs, report paths, status. **Human-facing summary:** [`README.md`](README.md). **Operator commands:** [`docs/OPERATIONS.md`](docs/OPERATIONS.md).

## Fine-tuning (CLAP)

**Target labels (project decision):** fine-tune and primary reporting on **three** multihot classes only — **`inst_piano`**, **`inst_vocal`**, **`mood_relaxing`**. Gold retrieval-vs-random showed these as the most stable vs metadata text; other template columns may stay labeled for reference but are **out of scope for default fine-tune** unless explicitly reopened.

- **Code:** `app/init_model.py` — freeze backbone; optionally train `audio_projection`, `audio_transform`, `text_projection`, `text_transform` via `params['unfreeze_layers']`; contrastive loss (cross-entropy on scaled audio–text similarity); Adam; optional early stopping on mean diagonal similarity.
- **Data:** `app/data_handling/music_split_to_15s.py`, `music_build_train_val_from_15s.py` → `data/mapping/clap_train_15s.jsonl`, `clap_val_15s.jsonl`. **Training** loads that JSONL by default (`load_training_pairs` in `app/init_model.py`), not `music_db/` full tracks.
- **Checkpoint path:** `config/settings.py` → `CLAP_MODEL_FILE` / `BEST_MODEL_FILE` as applicable.

### Cloud / multi-seed fine-tune

- **Driver:** `python -m app.train_clap_multiseed` — checkpoints under `model/clap/finetune/<run_id>/seed_<n>/best_model.pt`; logs under `data/log/finetune_runs/<run_id>/seed_<n>/` (`params.json`, `metrics.jsonl`) and run-level `summary.json` (**mean ± std** of best train-time similarity).
- **Seeding:** `model_creation` calls `set_seed` from `params["seed"]` (default `42`) and logs per-epoch metrics to `metrics.jsonl`. Training always initializes from **`CLAP_PRETRAINED_BACKBONE_FILE`** (unaffected by `RAGWEB_CLAP_CHECKPOINT`).
- **Eval on fine-tuned weights:** set env **`RAGWEB_CLAP_CHECKPOINT`** to e.g. `model/clap/finetune/<run_id>/seed_<n>/best_model.pt`, then run retrieval / tempo eval as usual (`CLAP_MODEL_FILE` in settings becomes that path).
- **Report:** prioritize retrieval rows for **`inst_piano`**, **`inst_vocal`**, **`mood_relaxing`** per seed, then aggregate (mean ± std or table per seed). Step-by-step: [`docs/FINE_TUNING_TUTORIAL.md`](docs/FINE_TUNING_TUTORIAL.md). Checklist: [`docs/cloud_finetune_protocol.md`](docs/cloud_finetune_protocol.md).

## Self-training loop (CLAP + LLM refine v2)

- **Driver:** `python -m app.train_clap_self_loop` — mine → **Llama refine + CLAP gate** → mixed JSONL → **val early-stop** fine-tune. Guide: [`docs/CLAP_SELF_TRAIN.md`](docs/CLAP_SELF_TRAIN.md).
- **v2:** `--refine` uses `LlmRefiner` + [`app/self_train/gate.py`](app/self_train/gate.py); `--no-refine` for mining-only ablation (B3).
- **Training:** `model_creation` takes `val_jsonl`; best **val** checkpoint; patience default 2; max epochs 20; **`min_epochs` 5** before inner val early-stop; epoch train-sim computed in **batches** (avoids OOM on ~66k clips).
- **Outer loop:** outer early-stop only if prior iter `best_epoch >= min_epochs_before_outer` (default 5).
- **Prerequisite:** full-library 15s manifests (`music_split_to_15s`, `music_build_train_val_from_15s`).
- **Llama:** `model/llama3.1-8b-instruct/`; set `RAGWEB_LLM_4BIT=1` on single GPU.
- **Slurm:** `REFINE=1 RUN_GOLD_EVAL=1 sbatch scripts/sbatch_clap_self_train.sh` — defaults: **`--mem=256G`**, `BATCH_SIZE=32`, `MIN_EPOCHS=5`. Jobs **120645/120650** OOM'd at train (64G/128G, batch 128).
- **Agent run:** `docs/agent_runs/20260526_self_train_v2/` — **`STATE.md`** = latest handoff (Phase A progress, screens, next steps).
- **Phase A (2026-05-26):** `screen clap_full_15s` running on `master01`; `clap_split_summary.json` still **pilot (44 songs)** until build finishes; `phase_a.done` not set yet. B4 auto-submit: `screen clap_b4_wait` + `scripts/wait_phase_a_then_self_train.sh`.

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
- CLAP self-train loop: `python -m app.train_clap_self_loop` (see `docs/CLAP_SELF_TRAIN.md`)
- Full ablation job (pretrained + per-seed matrices + summary CSVs): `sbatch scripts/sbatch_clap_ablation.sh` → `data/eval/ablation/`; report module `app/data_handling/music_eval_ablation_report.py`
- **LLM vs original caption ablation** (3 seeds, matched FT): `sbatch scripts/sbatch_llm_caption_ablation.sh` → `data/eval/llm_ablation/` (`REPORT.md`, `summary_primary.csv`); build JSONL: `python -m app.data_handling.music_build_llm_train_jsonl`. **Audio cache (fast FT):** `python -m app.data_handling.music_precompute_clap_audio_cache --jsonl data/mapping/clap_train_15s.jsonl --jsonl data/mapping/clap_val_15s.jsonl` → `data/embeddings_cache/clap_backbone/<backbone>/`; training auto-uses via `RAGWEB_CLAP_AUDIO_CACHE` (orchestrator sets it). **Resume:** `SKIP_BUILD=1 SKIP_PRECOMPUTE=1 sbatch ...` skips done seeds; **one arm:** `SKIP_LLM=1` or `SKIP_ORIG=1`; **eval only:** `SKIP_BUILD=1 SKIP_TRAIN=1`. Slurm default `--time=48:00:00`; optional node-local MP3 staging (`STAGE_AUDIO=1`) superseded by backbone cache when present.
- **Full-corpus LLM ablation** (symmetric per-checkpoint FAISS + caption index): Job 1 `sbatch scripts/sbatch_llm_full_corpus_gen.sh` (72h, resumable song-level LLM → `data/mapping/clap_train_llm_full.jsonl`); Job 2 `SKIP_LLM_GEN=1 sbatch scripts/sbatch_llm_full_ablation.sh` (48h, audio cache + `thesis_llm_full_llm` FT + metadata/caption index rebuild per seed). Reuses orig checkpoints from `thesis_llm_ablation_orig` by default (`SKIP_ORIG_TRAIN=1`). Outputs: `data/eval/llm_full_ablation/REPORT.md`. Report-only: `bash scripts/run_llm_full_ablation_report_only.sh`. Modules: `music_refine_full_corpus_captions`, `music_build_retrieval_faiss`, `music_eval_llm_full_ablation_report`. **Post-run review:** [`docs/RESEARCH_DIRECTIONS.md`](docs/RESEARCH_DIRECTIONS.md).
- **Tag-only vs tag→LLM ablation** (thesis question D): `python -m app.data_handling.music_build_tag_train_jsonl` → Job 1 `sbatch scripts/sbatch_tag_llm_corpus_gen.sh` → Job 2 `SKIP_LLM_GEN=1 sbatch scripts/sbatch_tag_llm_ablation.sh`. Full corpus with gold join + fallback `"music"`; primary tags only for training text. Run IDs: `thesis_tag_only`, `thesis_tag_llm`. Output: `data/eval/tag_llm_ablation/REPORT.md`. Agent run: [`docs/agent_runs/20260601_tag_train_llm_ablation/`](docs/agent_runs/20260601_tag_train_llm_ablation/).
- **Domain tradeoff / forgetting vs specialization** (thesis question E): `python -m app.data_handling.music_build_mixed_domain_train_jsonl` → `sbatch scripts/sbatch_domain_tradeoff_ablation.sh`. Mixed train: anime tag JSONL + MTAT + OpenMIC (eval holdouts excluded; Jamendo OOD-only). Run IDs: `thesis_tag_only` (reference), `thesis_tag_mixed`. Output: `data/eval/domain_tradeoff/REPORT.md`. Guide: [`docs/DOMAIN_TRADEOFF.md`](docs/DOMAIN_TRADEOFF.md). Agent run: [`docs/agent_runs/20260609_domain_tradeoff/`](docs/agent_runs/20260609_domain_tradeoff/).

## Downloads (public eval + Llama)

- **Status (update after sessions):** [`docs/PROGRESS.md`](docs/PROGRESS.md) via `bash scripts/refresh_progress.sh` — human guide: [`docs/PROGRESS_MONITOR.md`](docs/PROGRESS_MONITOR.md); downloads: [`docs/DOWNLOAD_STATUS.md`](docs/DOWNLOAD_STATUS.md) via `bash scripts/refresh_download_status.sh`.
- **Public OOD pipeline (download → eval → report):** `bash scripts/run_public_ood_pipeline.sh` (auto one step) — progress: `docs/PROGRESS.md` § **Public OOD pipeline** (`bash scripts/refresh_progress.sh`). Download: `bash scripts/run_public_eval_download.sh mtat|openmic|all`; status: `bash scripts/status_public_eval_download.sh`. Guide: [`docs/PUBLIC_OOD_EVAL.md`](docs/PUBLIC_OOD_EVAL.md).
- **Public OOD retrieval (post-hoc):** `bash scripts/run_public_eval.sh` or `sbatch scripts/sbatch_public_eval.sh` — outputs `data/eval/{dataset}_public/` + `data/eval/REPORT.md`. `ARMS` e.g. `pretrained thesis_tag_only thesis_tag_llm`.
- **Llama:** `scripts/download_llama31_8b.sh` → `docs/LLM_LOCAL.md`.

## Local LLM (Llama 3.1 8B, optional)

- **Download:** `bash scripts/download_llama31_8b.sh` → `model/llama3.1-8b-instruct/` (gitignored).
- **Check:** `python -m app.llm_local --smoke-test`
- **Code:** `app/llm_local.py`; guide: `docs/LLM_LOCAL.md`
- **Use (planned):** caption refinement on hard-mined train clips + CLAP gate; metadata extraction stays on Grok (`music_extract_metadata.py`).

## After a Cursor reset

1. Read this file + `README.md`.
2. For **thesis questions A–D** (status, reports, do not mix B vs D), read **`docs/THESIS_QUESTIONS.md`**.
3. For tag choices and caveats, read `docs/class_selected.txt`. For fine-tune commands, read `docs/FINE_TUNING_TUTORIAL.md`.
4. For gold merge semantics, read `docs/README_eval_merge.md`.
5. For download progress, read `docs/DOWNLOAD_STATUS.md` (or run `bash scripts/refresh_download_status.sh`).
6. For structured agent work, read `docs/multi_agent_workflow.md` and open any in-progress `docs/agent_runs/<run_id>/` folder; read **`STATE.md`** there first if present.
