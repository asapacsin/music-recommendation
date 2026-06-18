# Public OOD eval (Jamendo + MTAT + OpenMIC)

**Run ID:** `20260601_public_ood_eval`

## Goal

Post-train retrieval test on three public corpora; no training changes.

## Commands

```bash
# Build manifests (Jamendo also downloads MP3s)
BUILD_MANIFESTS=1 bash scripts/run_public_eval.sh

# Or per dataset after data on disk
python -m app.data_handling.music_build_mtat_manifest --max-per-tag 60
python -m app.data_handling.music_build_openmic_manifest --max-per-tag 60

# Test checkpoints
ARMS="pretrained thesis_ft_v1 thesis_tag_only thesis_tag_llm" \
  DATASETS="jamendo mtat openmic" \
  sbatch scripts/sbatch_public_eval.sh
```

## Outputs

- `data/eval/jamendo_public/`, `mtat_public/`, `openmic_public/`
- `data/eval/REPORT.md` (combined)

## Label mapping

- **MTAT:** vocal OR cols; relaxing = calm OR mellow
- **OpenMIC:** piano + voice only (long-format aggregated CSV)
