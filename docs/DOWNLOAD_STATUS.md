# Dataset & model download status

Living reference for **public eval audio**, **local LLM weights**, and related paths. Update after download sessions (commands at bottom).

**Public OOD runbook:** [`PUBLIC_OOD_EVAL.md`](PUBLIC_OOD_EVAL.md) — download, manifests, eval, readiness in [`PROGRESS.md`](PROGRESS.md).

**Machine-readable snapshot:** `data/eval/download_status_snapshot.json` (refresh via `bash scripts/refresh_download_status.sh`).

## Quick summary

| Asset | Status | Notes |
|-------|--------|--------|
| In-domain `music_db` | On disk | ~4011 tracks; not a remote download |
| CLAP 15s train/val | Partial | 44 source songs → 743 segments; see `data/mapping/clap_split_summary.json` |
| Jamendo annotations | Done | `data/public_eval/jamendo/mtg-jamendo-dataset/` |
| **Jamendo five-tag eval** | In progress | 297-track manifest; per-track CDN MP3s → `audio_five_tag/` |
| Jamendo bulk moodtheme | Stalled / skip | Huge tar under `jamendo/audio/`; use five-tag downloader instead |
| MTAT | Partial | CSVs done; split zip parts incomplete |
| OpenMIC | Not started | `data/public_eval/openmic/` empty |
| Llama 3.1 8B | Partial | Need 4× `model-*.safetensors` shards (~16 GB) |

*Last refreshed:* see `download_status_snapshot.json` → `updated_utc`.

## Public post-train test manifests

| Dataset | Manifest | Builder |
|---------|----------|---------|
| Jamendo | `data/eval/jamendo_five_tag_manifest.jsonl` | `python -m app.data_handling.music_eval_jamendo_five_tag_download` |
| MTAT | `data/eval/mtat_manifest.jsonl` | `python -m app.data_handling.music_build_mtat_manifest` |
| OpenMIC | `data/eval/openmic_manifest.jsonl` | `python -m app.data_handling.music_build_openmic_manifest` |

Combined test: `bash scripts/run_public_eval.sh` (see `AGENTS.md`).

## Jamendo five-tag (primary public OOD audio)

| Item | Path |
|------|------|
| Manifest | `data/eval/jamendo_five_tag_manifest.jsonl` |
| Summary | `data/eval/jamendo_five_tag_manifest.summary.json` |
| Audio | `data/public_eval/jamendo/audio_five_tag/` |
| Log | `data/log/jamendo_five_tag_download.log` |

**Tags (cap 60 per tag):** `pub_piano`, `pub_guitar`, `pub_vocal`, `pub_relaxing`, `pub_epic`.

```bash
# Manifest only
python -m app.data_handling.music_eval_jamendo_five_tag_download --manifest-only

# Download audio (skips existing non-empty MP3s)
python -m app.data_handling.music_eval_jamendo_five_tag_download --max-per-tag 60
```

`audio_path` is written to the manifest **only when the download job finishes** (or re-run after partial completion).

## MTAT (MagnaTagATune)

| Item | Path |
|------|------|
| Dir | `data/public_eval/magnatagatune/` |
| CSVs | `annotations_final.csv`, `clip_info_final.csv`, `comparisons_final.csv` |
| Audio | `mp3.zip.001` … `mp3.zip.003` (1.1G / 1.1G / 737M) → verify sizes → **`cat`** → `unzip -t` → extract |

**Recommended:** Slurm backend (size checks + manifest):

```bash
bash scripts/run_public_eval_download.sh mtat
bash scripts/status_public_eval_download.sh
```

Extract only when zip parts already verified:

```bash
EXTRACT_ONLY=1 bash scripts/run_public_eval_download.sh mtat
```

Manual (verify each part size first):

```bash
cd data/public_eval/magnatagatune
wget -c https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.001
wget -c https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.002
# If .003 wrong size (must be exactly 772769864 bytes), rm and fresh wget
wget -c https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.003
cat mp3.zip.001 mp3.zip.002 mp3.zip.003 > mp3_all.zip
unzip -t mp3_all.zip && unzip -q mp3_all.zip
```

Or: `SKIP_JAMENDO=1 SKIP_OPENMIC=1 bash scripts/download_public_eval.sh`

## OpenMIC

```bash
cd data/public_eval/openmic
wget -c -O openmic-2018-v1.0.0.tgz \
  https://zenodo.org/record/1432913/files/openmic-2018-v1.0.0.tgz
tar -xzf openmic-2018-v1.0.0.tgz
```

Or: `SKIP_JAMENDO=1 SKIP_MTAT=1 bash scripts/download_public_eval.sh`

## Jamendo bulk (not recommended)

Full `autotagging_moodtheme` audio via `scripts/download_public_eval.sh` (~10 GB+). Prefer five-tag capped download above.

## Llama 3.1 8B Instruct

| Item | Path |
|------|------|
| Weights | `model/llama3.1-8b-instruct/` (gitignored) |
| Script | `scripts/download_llama31_8b.sh` |
| Guide | `docs/LLM_LOCAL.md` |

Requires Hugging Face Meta license approval, then:

```bash
hf auth login
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct \
  --local-dir model/llama3.1-8b-instruct \
  --include "model-*.safetensors"
python -m app.llm_local --check-only
```

Alternative: ModelScope `LLM-Research/Meta-Llama-3.1-8B-Instruct` if HF access is denied.

## In-domain CLAP pipeline (not a download)

Full library not split to 15s yet. Next steps: `music_split_to_15s` + `music_build_train_val_from_15s` on full `music_db`.

## Refresh local snapshot

```bash
bash scripts/refresh_download_status.sh
```

Prints counts and updates `data/eval/download_status_snapshot.json`. Paste new numbers into the table above if you maintain this file by hand.

## Entry points

- **`bash scripts/run_public_eval_download.sh mtat|openmic|all`** — submit download to Slurm (close terminal after)
- **`bash scripts/status_public_eval_download.sh`** — progress + logs + PROGRESS.md snippet
- Low-level: `scripts/download_*_backend.sh`, `scripts/sbatch_download_public_eval.sh`
- Legacy bulk: `scripts/download_public_eval.sh`
- `app/data_handling/music_eval_jamendo_five_tag_download.py` — capped five-tag Jamendo
- `scripts/download_llama31_8b.sh` — Llama weights
- Agent context: `AGENTS.md`
