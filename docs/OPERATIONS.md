# Operations guide — Question E

Commands to build data and run the **Grok domain tradeoff** pipeline. Story and numbers: [`README.md`](../README.md), [`THESIS_QUESTIONS.md`](THESIS_QUESTIONS.md).

---

## 1. Metadata & 15s clips

```bash
python app/data_handling/music_extract_metadata.py
python app/data_handling/music_build_train_val_from_15s.py
```

Outputs: `data/mapping/clap_train_15s.jsonl`, `clap_val_15s.jsonl`.

---

## 2. Mixed-domain train JSONL

```bash
python -m app.data_handling.music_build_mixed_domain_train_jsonl \
  --anime-jsonl data/mapping/clap_train_15s.jsonl \
  --out-jsonl data/mapping/clap_train_grok_mixed.jsonl
```

---

## 3. Full Question E pipeline (Slurm)

```bash
sbatch scripts/sbatch_domain_tradeoff_ablation.sh
```

Stages: mixed JSONL → audio cache → FT `thesis_grok_only` + `thesis_grok_mixed` (seeds 42–44) → gold eval → public OOD → `data/eval/domain_tradeoff/REPORT.md`.

Eval-only:

```bash
SKIP_BUILD=1 SKIP_CACHE=1 SKIP_TRAIN=1 bash scripts/run_domain_tradeoff_ablation.sh
```

Details: [`DOMAIN_TRADEOFF.md`](DOMAIN_TRADEOFF.md). FT params: `data/eval/domain_tradeoff/train_params.json`.

---

## 4. Gold set (in-domain eval)

```bash
python -m app.data_handling.music_eval_merge_gold
```

See [`README_eval_merge.md`](README_eval_merge.md).

---

## 5. Metadata FAISS + pretrained baseline matrix

```bash
python -m app.metadata_faiss build --min-confidence 0.35
python -m app.data_handling.music_eval_retrieval_vs_random --top-k 10
```

Fine-tuned eval: `export RAGWEB_CLAP_CHECKPOINT=.../best_model.pt` then rerun matrix.

---

## 6. Public OOD (optional standalone)

```bash
ARMS="pretrained thesis_grok_only thesis_grok_mixed" bash scripts/run_public_eval.sh
```

Or: `sbatch scripts/sbatch_public_eval.sh` with the same `ARMS`.

---

## 7. Audio cache (speeds up FT)

```bash
python -m app.data_handling.music_precompute_clap_audio_cache \
  --jsonl data/mapping/clap_train_15s.jsonl \
  --jsonl data/mapping/clap_val_15s.jsonl \
  --jsonl data/mapping/clap_train_grok_mixed.jsonl
```

---

## Key modules

| Module | Role |
|--------|------|
| `app/init_model.py` | CLAP load, train loop |
| `app/train_clap_multiseed.py` | Multi-seed FT driver |
| `app/data_handling/music_build_mixed_domain_train_jsonl.py` | Mixed train JSONL |
| `app/data_handling/music_eval_domain_tradeoff_report.py` | 2×2 report |
| `app/metadata_faiss.py` | Metadata index |
| `app/data_handling/music_eval_retrieval_vs_random.py` | Gold retrieval matrix |

Progress: `bash scripts/refresh_progress.sh` → [`PROGRESS.md`](PROGRESS.md).
