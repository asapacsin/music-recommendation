# Progress monitor — human guide

This doc explains how to **read and refresh** the auto-generated thesis progress dashboard. It does **not** assign tasks, due dates, or team workflows — it only reports what is already on disk and in recent Slurm logs.

**Live dashboard:** [`docs/PROGRESS.md`](PROGRESS.md)  
**Objectives (what A–D mean):** [`docs/THESIS_QUESTIONS.md`](THESIS_QUESTIONS.md)  
**Download progress (separate):** [`docs/DOWNLOAD_STATUS.md`](DOWNLOAD_STATUS.md)

---

## Quick start

From the repo root:

```bash
cd ~/music-recommendation
bash scripts/refresh_progress.sh
```

Then open **`docs/PROGRESS.md`** in the editor or run:

```bash
less docs/PROGRESS.md
```

Refresh after every Slurm batch finishes (or fails), and whenever you want a snapshot before submitting the next step.

**Machine-readable output** (for scripts or jq):

```bash
bash scripts/refresh_progress.sh --json-only
# or
cat data/eval/progress_snapshot.json
```

---

## What gets checked

The monitor **does not** call Slurm (`squeue` / `sacct`). It inspects:

| Source | Used for |
|--------|----------|
| Checkpoint files | `model/clap/finetune/<run_id>/seed_*/best_model.pt` |
| Training logs | `data/log/finetune_runs/<run_id>/seed_*/metrics.jsonl`, `training_complete.json` |
| Eval reports | e.g. `data/eval/tag_llm_ablation/REPORT.md`, `data/eval/llm_full_ablation/REPORT.md` |
| Train JSONL | `data/mapping/clap_train_tag*.jsonl` line counts |
| Audio cache | `data/embeddings_cache/clap_backbone/.../index.json` |
| Slurm stdout | Newest `slurm-<jobid>.out` files in the repo root |

If a job is **still running**, the log may not show `Done.` yet — the dashboard may show the previous unit as **running** or the job as **running** until you refresh after completion.

---

## Reading `docs/PROGRESS.md`

The file is **overwritten** on each refresh. Do not edit it by hand; change happens on disk (train, eval, Slurm), then re-run the refresh script.

### Header

Shows `Last refresh: <UTC time>`. If that timestamp is old, run `bash scripts/refresh_progress.sh` again.

### Thesis questions

| Column | Meaning |
|--------|---------|
| **ID** | Thesis question A–D (see [`THESIS_QUESTIONS.md`](THESIS_QUESTIONS.md)) |
| **Status** | Inferred from reports + checkpoints (see [Status words](#status-words) below) |
| **Report / artifacts** | Main result path and/or `run_id: N/3 seeds` |

**Example:** `thesis_tag_only: 3/3 seeds` means all three fine-tune seeds (42, 43, 44) have a completed checkpoint for that run.

### Question D pipeline

Step-by-step units for the tag ablation (independent of other questions):

| Unit | Step | When it is **done** |
|------|------|---------------------|
| 0 | Tag JSONL | `clap_train_tag.jsonl` exists (~65k lines) |
| 1 | Llama tag→text | `clap_train_tag_llm.jsonl` + 3440 song rows in `clap_train_tag_llm_songs.jsonl` |
| 2 | Audio cache | Backbone cache `index.json` present |
| 3 | FT `thesis_tag_only` | 3/3 seeds complete |
| 4 | FT `thesis_tag_llm` | 3/3 seeds complete |
| 5 | Gold eval + report | `data/eval/tag_llm_ablation/REPORT.md` exists |

Unit **state**:

- **done** — criteria met  
- **next** — first incomplete unit; safe to submit this step (if you intend to continue D)  
- **running** — this unit has partial seed progress (e.g. 1/3 checkpoints)  
- **pending** — later units, not started  

### Question D — training recipe

Auto-generated block in [`PROGRESS.md`](PROGRESS.md) after each refresh. Summary:

| Topic | Detail |
|-------|--------|
| **Arms** | `thesis_tag_only` (short tag strings) vs `thesis_tag_llm` (Llama-expanded text) — **same clips, val, backbone, hyperparams** |
| **Train data** | ~65k 15s clips; tag JSONL vs tag→LLM JSONL under `data/mapping/` |
| **Val data** | `clap_val_15s.jsonl` (Grok-style captions; used for early-stop only) |
| **Seeds** | 42, 43, 44 |
| **Stop rule** | Max `val_similarity` on val; patience 2; min_epochs 5; max_epochs 20 (see `data/eval/llm_ablation/train_params.json`) |
| **Thesis metric** | Gold retrieval P@K / nDCG — **Unit 5**, not `val_similarity` during training |

Full spec: [`THESIS_QUESTIONS.md`](THESIS_QUESTIONS.md) Question D.

### Public OOD download

```bash
bash scripts/run_public_eval_download.sh mtat    # or openmic | all
bash scripts/status_public_eval_download.sh
```

No screen. Full guide: **[`PUBLIC_OOD_EVAL.md`](PUBLIC_OOD_EVAL.md)**.

### Fine-tune seeds

| Topic | Detail |
|-------|--------|
| **Purpose** | Post-train retrieval on Jamendo / MTAT / OpenMIC (not A–D) |
| **Report** | `data/eval/REPORT.md` |
| **Prep** | Per-dataset manifest + local `audio_path`; download via `scripts/download_public_eval.sh` or per-dataset commands in `PUBLIC_OOD_EVAL.md` |
| **Download counts** | `bash scripts/refresh_download_status.sh` → merged into progress snapshot |
| **Eval** | `bash scripts/run_public_eval.sh` / `sbatch scripts/sbatch_public_eval.sh` |

Unit **prep_state**: **done** when all manifest rows have audio on disk; **partial** when some do; **pending** otherwise.  
Unit **eval_state**: counts `{dataset}_public/{arm}_seed{N}.csv` vs expected (4 arms × 3 seeds by default).

### Fine-tune seeds

Per `run_id`, lists each seed:

- **ok** — checkpoint + training looks complete  
- **…** — checkpoint exists but completion marker unclear  
- **—** — no checkpoint yet  

The `epN val=X.XXXX` values are the **best** validation similarity at the saved checkpoint epoch (`training_complete.json`), not the last training epoch.

### Recent Slurm jobs

Newest logs first (up to 5 shown in markdown; 8 in JSON). Each block has:

- **job id** — matches `slurm-<id>.out`  
- **state** — `done`, `skipped`, `failed`, or `running` (heuristic from log text)  
- **phase** — e.g. `ft_tag_only`, `ft_tag_llm`, `llm_corpus_gen`, `skipped_train`  
- **tail** — last few log lines for quick context  

To follow a live job:

```bash
tail -f slurm-<JOBID>.out
```

---

## Status words

| Word | Typical meaning |
|------|-----------------|
| **done** | Report exists or all seeds finished for that question |
| **partial** | Some artifacts exist, not enough for “done” |
| **running** | Work in progress (seeds or question D pipeline) |
| **eval pending** | Both D arms trained; report not generated yet |
| **next** | Next pipeline unit to run |
| **pending** | Not started |
| **skipped** | Log shows `SKIP_TRAIN=1` (or similar) without real train output |
| **failed** | `ERROR:` or Python traceback in the Slurm log |

---

## Typical workflows

### After submitting a Slurm job

1. Note the job id from `Submitted batch job <id>`.
2. `tail -f slurm-<id>.out`
3. When the job ends (or you return later): `bash scripts/refresh_progress.sh`
4. Open `docs/PROGRESS.md` — confirm the expected unit moved to **done** and the next unit is **next**.

### Before submitting the next Question D unit

Refresh and check:

- Units 0–2 should be **done** before any fine-tune.
- Unit 3 **done** before starting unit 4 (`SKIP_TAG_TRAIN=1`).
- Unit 4 **done** before eval-only submit (`SKIP_TRAIN=1`, `SKIP_EVAL=0`).

Commands for each unit are in [`THESIS_QUESTIONS.md`](THESIS_QUESTIONS.md) (Question D) and [`docs/agent_runs/20260601_tag_train_llm_ablation/PLAN.md`](agent_runs/20260601_tag_train_llm_ablation/PLAN.md).

### Public OOD pipeline (download → eval → report)

Separate from Question D. **`docs/PROGRESS.md` → Public OOD pipeline** shows:

- Numbered units **0–4** (Jamendo / MTAT / OpenMIC prep → retrieval eval → `data/eval/REPORT.md`)
- Mermaid flowchart and **prep/eval progress bars**
- **Eval matrix** (dataset × arm × seed CSV counts)
- **Next commands** (auto-generated)

**Orchestrator (one step per run):**

```bash
bash scripts/run_public_ood_pipeline.sh          # auto: download OR eval, whichever is next
bash scripts/run_public_ood_pipeline.sh status   # refresh + print plan
bash scripts/run_public_ood_pipeline.sh eval     # eval all prep-ready datasets
DRY_RUN=1 bash scripts/run_public_ood_pipeline.sh
```

JSON plan for scripts:

```bash
python -m app.progress_monitor --ood-plan-json
jq '.public_ood.pipeline_units[] | {unit, state, label}' data/eval/progress_snapshot.json
```

Download detail: `bash scripts/status_public_eval_download.sh`


The monitor counts seeds with `best_model.pt` + complete metrics. If only 2/3 seeds finished:

- Dashboard shows **running** on that unit.
- Re-submit the same sbatch command; training skips finished seeds when `SKIP_EXISTING=1` (default).

### Checking without opening markdown

```bash
# Question D unit states
jq '.question_d_units[] | {unit, label, state}' data/eval/progress_snapshot.json

# Thesis question D summary
jq '.thesis_questions[] | select(.id=="D")' data/eval/progress_snapshot.json

# Latest Slurm job state
jq '.slurm_recent[0] | {job_id, state, phase}' data/eval/progress_snapshot.json
```

---

## What this monitor does *not* cover

| Topic | Where to look instead |
|-------|------------------------|
| **What** A–D mean, eval definitions | [`THESIS_QUESTIONS.md`](THESIS_QUESTIONS.md) |
| **Jamendo / MTAT / OpenMIC downloads** | [`DOWNLOAD_STATUS.md`](DOWNLOAD_STATUS.md) + [`PUBLIC_OOD_EVAL.md`](PUBLIC_OOD_EVAL.md) + `bash scripts/refresh_download_status.sh` |
| **Detailed experiment plan / commands** | `docs/agent_runs/<run_id>/PLAN.md` |
| **Agent session memory** | `state/agent_state.json` |
| **Whether a job is queued on the cluster** | `squeue -u $USER` (not integrated here) |

---

## Troubleshooting

**Dashboard looks stale**  
Run `bash scripts/refresh_progress.sh` and check the UTC timestamp at the top of `PROGRESS.md`.

**Unit still “pending” but I ran training**  
Confirm checkpoints exist:

```bash
ls model/clap/finetune/thesis_tag_only/seed_*/best_model.pt
ls model/clap/finetune/thesis_tag_llm/seed_*/best_model.pt
```

Then refresh again.

**Slurm job shows `skipped` immediately**  
Often `SKIP_TRAIN=1` was still exported in the shell. Fix and resubmit:

```bash
unset SKIP_TRAIN
export SKIP_TRAIN=0
# ... other SKIP_* flags ...
sbatch scripts/sbatch_tag_llm_ablation.sh
```

**Job shows `done` but unit not done**  
Read the log tail in `PROGRESS.md` — the job may have skipped train or eval. Match `phase` and `SKIP_*` lines to your intent.

**Question A shows `partial`**  
Needs both retrieval matrix CSVs under `data/eval/` and completed `thesis_ft_v1` seeds; either alone yields partial.

---

## For developers

- Implementation: `app/progress_monitor.py`
- Shell entry: `scripts/refresh_progress.sh`
- Tests: `tests/test_progress_monitor.py`

To add a new tracked experiment, extend the checks in `progress_monitor.py` (thesis question row and/or pipeline units). Keep [`THESIS_QUESTIONS.md`](THESIS_QUESTIONS.md) as the human spec for what “done” means.
