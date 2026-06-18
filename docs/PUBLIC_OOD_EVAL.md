# Public OOD retrieval test

Post-train evaluation only — **does not train CLAP**. Separate from thesis questions **A–D** (see [`THESIS_QUESTIONS.md`](THESIS_QUESTIONS.md)).

**Readiness:** [`docs/PROGRESS.md`](PROGRESS.md) → **Public OOD pipeline** (`bash scripts/refresh_progress.sh`).

**End-to-end orchestrator** (download → eval → report, one step at a time):

```bash
bash scripts/run_public_ood_pipeline.sh          # auto
bash scripts/run_public_ood_pipeline.sh status # plan only
bash scripts/run_public_ood_pipeline.sh eval     # eval prep-ready datasets
```

Progress visualization: unit table, mermaid diagram, prep/eval bars, eval matrix — all in `PROGRESS.md` after refresh.

---

## Download datasets (one command — no screen)

From repo root. **Submit and close the terminal** — Slurm runs on the cluster.

```bash
cd ~/music-recommendation

# MTAT (~2.7 GB + extract + manifest)
bash scripts/run_public_eval_download.sh mtat

# OpenMIC (~2.6 GB + extract + manifest)
bash scripts/run_public_eval_download.sh openmic

# Both still needed (Jamendo usually done already):
bash scripts/run_public_eval_download.sh all
```

**Check anytime:**

```bash
bash scripts/status_public_eval_download.sh
```

Each dataset shows one of **COMPLETED**, **RUNNING**, **FAILED**, or **NOT STARTED**.  
**FAILED** prints a large error banner; the script exits **1** if any dataset failed, **2** if still running/incomplete, **0** only when all three are **COMPLETED**.

Status files: `data/log/public_eval_downloads/{jamendo,mtat,openmic}.status.json`  
Detail logs:

| Target | Log |
|--------|-----|
| Slurm | `slurm-<jobid>.out` (printed when you submit) |
| MTAT | `data/log/public_eval_downloads/mtat_backend.log` |
| OpenMIC | `data/log/public_eval_downloads/openmic_backend.log` |

---

## What each download produces

| Target | On disk after success |
|--------|------------------------|
| **mtat** | MP3s under `data/public_eval/magnatagatune/`; `data/eval/mtat_manifest.jsonl` with `audio_path` |
| **openmic** | `data/public_eval/openmic/openmic-2018/audio/*.ogg`; `data/eval/openmic_manifest.jsonl` |
| **jamendo** | `data/public_eval/jamendo/audio_five_tag/` (~297 MP3s); manifest already built |

MTAT uses **split zip** parts `mp3.zip.001`–`.003`. After each part’s byte size is verified, join with **`cat mp3.zip.001 mp3.zip.002 mp3.zip.003 > mp3_all.zip`**, then **`unzip -t mp3_all.zip`** before extract. (Do not join if `.003` is the wrong size — that produces a corrupt archive.)

---

## Run public eval (after downloads)

```bash
# Jamendo + MTAT when both show prep done in PROGRESS.md
DATASETS="jamendo mtat" ARMS="pretrained thesis_tag_only thesis_tag_llm" \
  sbatch scripts/sbatch_public_eval.sh

# All three when OpenMIC ready too
BUILD_MANIFESTS=1 DATASETS="jamendo mtat openmic" \
  ARMS="pretrained thesis_tag_only thesis_tag_llm" \
  sbatch scripts/sbatch_public_eval.sh
```

**Report:** `data/eval/REPORT.md`

---

## Purpose & datasets

After checkpoints exist: test **P@K / nDCG** on external audio (Jamendo, MTAT, OpenMIC).  
Not the same as Question D report (`data/eval/tag_llm_ablation/REPORT.md`).

| Dataset | Queries | Manifest |
|---------|---------|----------|
| Jamendo five-tag | piano, vocal, relaxing | `data/eval/jamendo_five_tag_manifest.jsonl` |
| MTAT | piano, vocal, relaxing | `data/eval/mtat_manifest.jsonl` |
| OpenMIC | piano, vocal only | `data/eval/openmic_manifest.jsonl` |

---

## Readiness checklist

- [ ] `bash scripts/status_public_eval_download.sh` — MTAT/OpenMIC prep **done**
- [ ] `docs/PROGRESS.md` Public OOD — manifest rows **ready**
- [ ] `sbatch scripts/sbatch_public_eval.sh` submitted
- [ ] `data/eval/REPORT.md` exists

---

## Related

- [`DOWNLOAD_STATUS.md`](DOWNLOAD_STATUS.md) — raw download counts
- [`PROGRESS_MONITOR.md`](PROGRESS_MONITOR.md) — how to read PROGRESS.md
- [`AGENTS.md`](../AGENTS.md)
