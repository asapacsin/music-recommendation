# Jamendo public OOD eval pipeline

**Run ID:** `20260601_jamendo_public_eval`

## Goal

Post-hoc CLAP retrieval on MTG-Jamendo five-tag manifest (OOD). No changes to training or gold ablation orchestrators.

## Prereqs

1. `data/eval/jamendo_five_tag_manifest.jsonl` (download module).
2. Local MP3s under `data/public_eval/jamendo/audio_five_tag/` (re-run download until enough rows have `audio_path`).

## Commands

```bash
# Default: pretrained + thesis_ft_v1, seeds 42–44
bash scripts/run_jamendo_public_eval.sh

# More arms (checkpoints must exist)
ARMS="pretrained thesis_ft_v1 thesis_tag_only thesis_tag_llm" bash scripts/run_jamendo_public_eval.sh

sbatch scripts/sbatch_jamendo_public_eval.sh
SKIP_EXISTING=1 sbatch scripts/sbatch_jamendo_public_eval.sh
```

## Outputs

- `data/eval/jamendo_public/{arm}_seed{N}.csv` (+ `.json`)
- `data/eval/jamendo_public/REPORT.md`, `summary_primary.csv`, `summary.json`

## Risks

- Partial download → small `n_pool` and unstable metrics.
- Full-track embed per eval (no FAISS); ~minutes per arm×seed on GPU.
