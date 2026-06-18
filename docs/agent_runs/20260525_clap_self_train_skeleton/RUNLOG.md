# RUNLOG — `20260525_clap_self_train_skeleton`

Append-only. Newest entries at the **bottom**.

---

## Entry — 2026-05-25

- **Actor**: Cursor agent
- **Command**:

```bash
cd /home/mc46451/music-recommendation
source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate ragweb
export PYTHONPATH="$PWD"

python -m app.train_clap_self_loop \
  --run-id smoke_self_v1 \
  --n-iters 1 \
  --no-refine \
  --num-epochs 1 \
  --max-samples 8 \
  --embed-batch-size 2
```

- **Exit code**: 0 (after `init_model` device fix for CPU labels)
- **Outputs created / updated**:
  - `data/self_train/smoke_self_v1/iter_0/hard_mined.jsonl`
  - `data/self_train/smoke_self_v1/iter_0/train_mixed.jsonl`
  - `model/clap/self_train/smoke_self_v1/iter_0/best_model.pt`
  - `data/self_train/smoke_self_v1/iter_0/iter_metrics.json`
  - `data/log/self_train_runs/smoke_self_v1/summary.json`
- **Notes**: First attempt failed on `.cuda()` without GPU; fixed labels to use embedding device.

---

## Entry — 2026-05-25

- **Actor**: Cursor agent
- **Command**:

```bash
python -m app.train_clap_self_loop \
  --run-id smoke_self_v2 \
  --n-iters 2 \
  --no-refine \
  --num-epochs 1 \
  --max-samples 8 \
  --embed-batch-size 2
```

- **Exit code**: 0
- **Outputs created / updated**:
  - `data/self_train/smoke_self_v2/iter_{0,1}/` artifacts
  - `model/clap/self_train/smoke_self_v2/iter_{0,1}/best_model.pt`
  - `data/log/self_train_runs/smoke_self_v2/iter_1/params.json` includes `init_checkpoint` → iter_0
  - `data/log/self_train_runs/smoke_self_v2/summary.json`
- **Notes**: iter_1 val_mean_similarity finite (~0.081). Full-dataset GPU run deferred to Slurm.
