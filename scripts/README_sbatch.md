# Slurm fine-tune script (SICC / DGX)

## Upload

Copy `scripts/sbatch_clap_finetune.sh` to the cluster (with your repo), e.g.:

```bash
# WSL on your PC
scp /mnt/e/Ragweb/scripts/sbatch_clap_finetune.sh \
  mc46451@dgx.sicc.um.edu.mo:/home/mc46451/music-recommendation/scripts/
```

Or place it in `$HOME` as `~/run_clap.sh`.

## Before first submit

On the cluster:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ragweb
cd ~/music-recommendation
export PYTHONPATH="$HOME/music-recommendation/app:$HOME/music-recommendation"

# 15s audio visible to the repo
ln -sfn ~/music_db ~/music-recommendation/data/music_db_15s

# Rebuild train manifest if needed (after fixing music_15s_map.json on cluster)
python -m app.data_handling.music_build_train_val_from_15s

python3 -c "
from app.init_model import load_training_pairs
p, t = load_training_pairs({})
print('clips:', len(p))
"
```

Need **`clips: > 0`** and backbone at `model/clap/music_audioset_epoch_15_esc_90.14.pt`.

## Submit

```bash
chmod +x ~/music-recommendation/scripts/sbatch_clap_finetune.sh
cd ~/music-recommendation
sbatch scripts/sbatch_clap_finetune.sh
```

Or from home:

```bash
sbatch ~/music-recommendation/scripts/sbatch_clap_finetune.sh
```

## Monitor

```bash
squeue -u $USER
tail -f ~/slurm-<jobid>.out
```

## Override run name / seeds without editing the file

```bash
RUN_ID=smoke_test N_SEEDS=1 sbatch scripts/sbatch_clap_finetune.sh
```

## After the fine-tune job

- Training record: `data/log/finetune_runs/<RUN_ID>/summary.json`
- Per-seed checkpoints: `model/clap/finetune/<RUN_ID>/seed_<n>/best_model.pt`
- Per-seed logs: `data/log/finetune_runs/<RUN_ID>/seed_<n>/` (`params.json`, `metrics.jsonl`)

## Retrieval eval (after fine-tune)

Script: `scripts/sbatch_clap_retrieval_eval.sh` — runs `music_eval_retrieval_vs_random` for each seed with `RAGWEB_CLAP_CHECKPOINT`. Builds `metadata_text_index.faiss` first if missing.

Prerequisites: `thesis_ft_v1` (or your `RUN_ID`) checkpoints under `model/clap/finetune/<RUN_ID>/`, backbone at `model/clap/…`, gold at `data/eval/gold_merged.jsonl`.

```bash
chmod +x ~/music-recommendation/scripts/sbatch_clap_retrieval_eval.sh
cd ~/music-recommendation
sbatch scripts/sbatch_clap_retrieval_eval.sh
```

Overrides:

```bash
RUN_ID=thesis_ft_v1 SEEDS="42" sbatch scripts/sbatch_clap_retrieval_eval.sh
```

After the job:

- `data/eval/retrieval_matrix_seed<seed>.csv` (and `.json`)
- Thesis headline rows: filter `query_text` ∈ `piano`, `vocal`, `relaxing`

Details: `docs/FINE_TUNING_TUTORIAL.md` §3

## Full ablation (pretrained + all seeds + report)

Script: `scripts/sbatch_clap_ablation.sh` — one job that:

1. Runs **full** retrieval matrix (9 style + 3 tempo queries, default flags) for **pretrained** CLAP.
2. Repeats for each fine-tuned seed (`model/clap/finetune/<RUN_ID>/seed_<n>/best_model.pt`).
3. Writes summary tables via `python -m app.data_handling.music_eval_ablation_report`.

```bash
chmod +x ~/music-recommendation/scripts/sbatch_clap_ablation.sh
cd ~/music-recommendation
sbatch scripts/sbatch_clap_ablation.sh
```

Outputs under `data/eval/ablation/`:

| File | Contents |
|------|----------|
| `pretrained.csv` / `pretrained.json` | Backbone-only retrieval |
| `ft_seed<N>.csv` / `.json` | Per-seed fine-tuned retrieval |
| `summary_primary.csv` | Pretrained vs FT mean±std for **piano, vocal, relaxing** @ `REPORT_TOP_K` (default 10) |
| `summary_all_queries.csv` | Same for **every** query in the pretrained matrix |
| `summary.json` | Paths and metadata |
| `query_ablation_report.md` | **Query-set ablation** — tiers + cumulative add-tag (see below) |
| `query_ablation_tiers.csv` / `query_ablation_cumulative.csv` | Macro P@K / ΔnDCG as query set grows |

**Query-set ablation** (expanding tag queries, not checkpoint comparison):

```bash
python -m app.data_handling.music_eval_query_ablation_report --top-k 10
```

Uses existing `pretrained.csv` and `ft_seed*.csv` in `data/eval/ablation/`.

Overrides:

```bash
RUN_ID=thesis_ft_v1 SEEDS="42 43" TOP_K="10 20" REPORT_TOP_K=20 sbatch scripts/sbatch_clap_ablation.sh
SKIP_EVAL=1 sbatch scripts/sbatch_clap_ablation.sh   # report only (CSVs must already exist)
```

If you already have `data/eval/retrieval_matrix_seed*.csv`, copy or symlink them into `data/eval/ablation/` as `ft_seed<N>.csv`, add `pretrained.csv`, then `SKIP_EVAL=1`.

## Composite query ablation (tags in one prompt)

Script: `scripts/sbatch_composite_query_ablation.sh` — cumulative prompts **without** trailing `music`:

1. `piano` → 2. `piano vocal` → 3. `piano vocal relaxing` (AND gold relevance; primary 3 tags only).

Runs pretrained + each fine-tuned seed; writes `data/eval/ablation/composite/composite_query_report.md`.

```bash
cd ~/music-recommendation
sbatch scripts/sbatch_composite_query_ablation.sh
```

```bash
SKIP_EVAL=1 sbatch scripts/sbatch_composite_query_ablation.sh   # report only
```

## Public eval datasets (download)

Script: `scripts/download_public_eval.sh` — MTG-Jamendo, MagnaTagATune, OpenMIC-2018 into `data/public_eval/` (gitignored).

```bash
conda activate ragweb   # pip install gdown if Jamendo step fails
bash scripts/download_public_eval.sh
```

Partial: `SKIP_JAMENDO=1` / `SKIP_MTAT=1` / `SKIP_OPENMIC=1`. Jamendo audio is large and slow.
