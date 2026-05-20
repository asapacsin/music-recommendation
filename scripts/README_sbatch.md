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

## After the job

- Training record: `data/log/finetune_runs/<RUN_ID>/summary.json`
- Per-seed checkpoints: `seed_<n>/best_model.pt`
- Retrieval eval (separate): see `docs/FINE_TUNING_TUTORIAL.md` §3
