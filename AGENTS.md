# Agent / session context (read after Cursor reset)

Chat history is not stored in this repo. **This file is the durable snapshot** of decisions and entry points.

## Agent session state

Per-turn working memory: **[`state/agent_state.json`](state/agent_state.json)**. Cursor rule: [`.cursor/rules/agent_state_workflow.mdc`](.cursor/rules/agent_state_workflow.mdc).

## Multi-agent workflow

For non-trivial work: [`docs/multi_agent_workflow.md`](docs/multi_agent_workflow.md) → `docs/agent_runs/<run_id>/` (PLAN, RUNLOG, REVIEW).

## Project in one line

CLAP fine-tuning on anime music (Grok captions) + mixed-domain training (MTAT/OpenMIC) → gold in-domain retrieval vs public OOD — **Question E** (specialization vs forgetting).

## Thesis question E (canonical pipeline)

**Question:** When we add public-domain clips to fine-tuning, is the OOD drop **forgetting** or **specialization**?

| Arm | Run ID | Train JSONL |
|-----|--------|-------------|
| Anime-only | `thesis_grok_only` | `data/mapping/clap_train_15s.jsonl` |
| Mixed | `thesis_grok_mixed` | `data/mapping/clap_train_grok_mixed.jsonl` |

- **Driver:** `sbatch scripts/sbatch_domain_tradeoff_ablation.sh` (or `bash scripts/run_domain_tradeoff_ablation.sh`)
- **Report:** `data/eval/domain_tradeoff/REPORT.md`, `summary.json`
- **Checkpoints:** `model/clap/finetune/thesis_grok_{only,mixed}/seed_{42,43,44}/best_model.pt`
- **Eval:** pretrained baseline + both arms; gold P@10 + OOD macro P@10 (Jamendo, MTAT, OpenMIC)
- **Guide:** [`docs/DOMAIN_TRADEOFF.md`](docs/DOMAIN_TRADEOFF.md), [`docs/THESIS_QUESTIONS.md`](docs/THESIS_QUESTIONS.md)
- **Opus pack:** [`docs/opus_tradeoff_bundle/OPUS_FEED.md`](docs/opus_tradeoff_bundle/OPUS_FEED.md)
- **Agent run:** [`docs/agent_runs/20260619_grok_domain_tradeoff/`](docs/agent_runs/20260619_grok_domain_tradeoff/) (job 122295)

**Primary tags:** `inst_piano`, `inst_vocal`, `mood_relaxing`.

**FT hyperparams:** `data/eval/domain_tradeoff/train_params.json` (batch 32, max 20 epochs, val early-stop).

## Core data & training

- **15s manifests:** `music_split_to_15s` → `music_build_train_val_from_15s` → `clap_train_15s.jsonl`, `clap_val_15s.jsonl`
- **Mixed JSONL:** `python -m app.data_handling.music_build_mixed_domain_train_jsonl --anime-jsonl data/mapping/clap_train_15s.jsonl --out-jsonl data/mapping/clap_train_grok_mixed.jsonl`
- **CLAP code:** `app/init_model.py`, `app/train_clap_multiseed.py`
- **Audio cache (optional):** `music_precompute_clap_audio_cache` on train+val JSONLs → `data/embeddings_cache/`

## Eval (shared infrastructure)

- **Metadata FAISS:** `python -m app.metadata_faiss build`
- **Gold retrieval matrix (pretrained baseline):** `python -m app.data_handling.music_eval_retrieval_vs_random`
- **Public OOD:** `bash scripts/run_public_eval.sh` — default arms: `pretrained thesis_grok_only thesis_grok_mixed`
- **Human gold:** `music_eval_merge_gold` → `data/eval/gold_merged.jsonl` (see [`docs/README_eval_merge.md`](docs/README_eval_merge.md))

## Quick commands

```bash
# Full E pipeline (build → cache → train → eval → report)
sbatch scripts/sbatch_domain_tradeoff_ablation.sh

# Eval / report only
SKIP_BUILD=1 SKIP_CACHE=1 SKIP_TRAIN=1 bash scripts/run_domain_tradeoff_ablation.sh

# Public OOD for grok arms
ARMS="pretrained thesis_grok_only thesis_grok_mixed" bash scripts/run_public_eval.sh

# Progress refresh
bash scripts/refresh_progress.sh
```

## After a Cursor reset

1. Read this file + [`README.md`](README.md)
2. Results: [`docs/THESIS_QUESTIONS.md`](docs/THESIS_QUESTIONS.md), [`data/eval/domain_tradeoff/REPORT.md`](data/eval/domain_tradeoff/REPORT.md)
3. Operator detail: [`docs/OPERATIONS.md`](docs/OPERATIONS.md)
