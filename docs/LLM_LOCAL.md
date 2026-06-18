# Local LLM (Llama 3.1 8B Instruct)

For later **caption refinement** in the CLAP self-training loop (text-only; does not hear audio).

## Model

| Item | Default |
|------|---------|
| Hugging Face repo | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| On-disk path | `model/llama3.1-8b-instruct/` (gitignored) |
| Env overrides | `RAGWEB_LLM_HF_REPO_ID`, `RAGWEB_LLM_MODEL_DIR` |
| 4-bit load | `RAGWEB_LLM_4BIT=1` (needs `pip install bitsandbytes`) |

Newer/smaller alternatives (set `RAGWEB_LLM_HF_REPO_ID` before download):

- `meta-llama/Meta-Llama-3.2-3B-Instruct` — lighter GPU smoke tests
- `meta-llama/Meta-Llama-3.1-70B-Instruct` — much higher quality; multi-GPU only

## Download

1. Accept the model license on Hugging Face for [Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct).
2. `huggingface-cli login` (or `export HF_TOKEN=...`).
3. From repo root:

```bash
bash scripts/download_llama31_8b.sh
```

Disk: ~16 GB (FP16 weights). Use `RAGWEB_LLM_4BIT=1` at runtime to reduce VRAM if needed.

## Verify

```bash
python -m app.llm_local --check-only
python -m app.llm_local --smoke-test
```

## Code

- `app/llm_local.py` — `load_llm()`, `chat_generate()`, smoke test CLI
- `app/text_processing.load_model()` — delegates to `load_llm()` (legacy entry point)
- `config/settings.py` — `LLM_MODEL_DIR`, `LLM_HF_REPO_ID`

Planned use: refine train captions on **hard CLAP-mined clips** only, with a CLAP embedding gate (not full 68k LLM passes).
