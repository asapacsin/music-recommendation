# Operations guide

Commands to **build data** and **run evals** referenced in the thesis. For the research story and results, read [`README.md`](../README.md) and [`THESIS_QUESTIONS.md`](THESIS_QUESTIONS.md).

---

## Pipeline (in order)

### 1. Metadata

```bash
python app/data_handling/music_extract_metadata.py
python app/data_handling/music_metadata_merge_process_meta.py --music-confidence-min 0.7
```

### 2. 15s clips + train/val manifests

```bash
python app/data_handling/music_split_to_15s.py
python app/data_handling/music_build_train_val_from_15s.py
```

Outputs: `data/mapping/clap_train_15s.jsonl`, `clap_val_15s.jsonl`, `clap_split_summary.json`.

### 3. Fine-tune (multi-seed)

```bash
python -m app.train_clap_multiseed --run-id thesis_ft_v1 --seeds 42,43,44
```

Or Slurm: `sbatch scripts/sbatch_clap_finetune.sh`. Tutorial: [`FINE_TUNING_TUTORIAL.md`](FINE_TUNING_TUTORIAL.md).

### 4. Metadata FAISS + gold retrieval matrix

```bash
python -m app.metadata_faiss build --min-confidence 0.35
python -m app.data_handling.music_eval_retrieval_vs_random --top-k 10
```

Eval with fine-tuned weights: `export RAGWEB_CLAP_CHECKPOINT=.../best_model.pt` then same matrix command.

---

## Human gold set

Build song manifest → label CSV → merge:

```bash
python -m app.data_handling.music_eval_build_song_manifest \
  --filter-val-jsonl data/mapping/clap_val_15s.jsonl --random-sample 150 --seed 42

python -m app.data_handling.music_eval_prepare_gold_multihot_csv
# label data/eval/gold_labels_multihot_template.csv (0/1 per column)

python -m app.data_handling.music_eval_merge_gold
```

Details: [`README_eval_merge.md`](README_eval_merge.md).

---

## Thesis ablation Slurm jobs

| Question | Command |
|----------|---------|
| B — LLM vs Grok (full corpus) | `sbatch scripts/sbatch_llm_full_corpus_gen.sh` then `SKIP_LLM_GEN=1 sbatch scripts/sbatch_llm_full_ablation.sh` |
| D — Tag vs tag→LLM | `sbatch scripts/sbatch_tag_llm_corpus_gen.sh` then `SKIP_LLM_GEN=1 sbatch scripts/sbatch_tag_llm_ablation.sh` |
| E — Domain tradeoff | `sbatch scripts/sbatch_domain_tradeoff_ablation.sh` |
| Public OOD | `bash scripts/run_public_eval.sh` or `sbatch scripts/sbatch_public_eval.sh` |

Progress: `bash scripts/refresh_progress.sh` → [`PROGRESS.md`](PROGRESS.md).

---

## Audio cache (optional, speeds up FT)

```bash
python -m app.data_handling.music_precompute_clap_audio_cache \
  --jsonl data/mapping/clap_train_15s.jsonl \
  --jsonl data/mapping/clap_val_15s.jsonl
```

Training picks up cache automatically when present under `data/embeddings_cache/`.

---

## Key modules

| Module | Role |
|--------|------|
| `app/init_model.py` | CLAP load, train loop, contrastive loss |
| `app/train_clap_multiseed.py` | Multi-seed fine-tune driver |
| `app/metadata_faiss.py` | Build / search metadata index |
| `app/data_handling/music_eval_retrieval_vs_random.py` | Gold retrieval matrix |
| `app/data_handling/music_eval_tag_llm_ablation_report.py` | Question D report |
| `app/data_handling/music_eval_llm_full_ablation_report.py` | Question B report |

Legacy / out of thesis scope: `app/recommend.py` (OpenL3), manual Top-10 human labeling workflow (`music_eval_topk_*`), clip-level tempo zero-shot as a headline metric.
