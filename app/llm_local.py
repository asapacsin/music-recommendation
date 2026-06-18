"""
Local Hugging Face causal LM (default: Meta-Llama-3.1-8B-Instruct) for caption refinement.

Weights live under ``settings.LLM_MODEL_DIR`` (gitignored). Download::

    bash scripts/download_llama31_8b.sh

Smoke test::

    python -m app.llm_local --smoke-test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import settings

_DEFAULT_MAX_NEW_TOKENS = 256


def resolve_model_path(model_path: Path | str | None = None) -> Path:
    path = Path(model_path) if model_path is not None else settings.LLM_MODEL_DIR
    return path.expanduser().resolve()


def model_is_ready(model_path: Path | str | None = None) -> bool:
    """True if ``config.json`` exists under the model directory."""
    path = resolve_model_path(model_path)
    return (path / "config.json").is_file()


def load_llm(
    model_path: Path | str | None = None,
    *,
    device: str = "auto",
    load_in_4bit: bool | None = None,
):
    """
    Load tokenizer + causal LM from ``model_path`` (default ``settings.LLM_MODEL_DIR``).

    ``device``: ``auto`` (CUDA if available), ``cuda``, or ``cpu``.
    ``load_in_4bit``: default from env ``RAGWEB_LLM_4BIT`` (0/1); needs ``bitsandbytes``.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = resolve_model_path(model_path)
    if not model_is_ready(path):
        raise FileNotFoundError(
            f"Local LLM not found at {path}\n"
            f"Run: bash scripts/download_llama31_8b.sh\n"
            f"Or set RAGWEB_LLM_MODEL_DIR to an existing Hugging Face snapshot."
        )

    if load_in_4bit is None:
        load_in_4bit = settings.LLM_LOAD_IN_4BIT

    tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {}
    use_4bit = bool(load_in_4bit)
    if use_4bit:
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model_kwargs["device_map"] = "auto"
        except ImportError:
            print(
                "warning: RAGWEB_LLM_4BIT=1 but bitsandbytes missing; "
                "falling back to float16 load",
                flush=True,
            )
            use_4bit = False

    if not use_4bit:
        use_cuda = device in ("auto", "cuda") and torch.cuda.is_available()
        resolved = "cuda" if use_cuda else "cpu"
        if device == "cuda" and not use_cuda:
            print("warning: CUDA requested but unavailable; using CPU", flush=True)
        model_kwargs["torch_dtype"] = torch.float16 if resolved == "cuda" else torch.float32
        model_kwargs["device_map"] = resolved

    model = AutoModelForCausalLM.from_pretrained(str(path), **model_kwargs)
    return model, tokenizer


def chat_generate(
    user_prompt: str,
    model,
    tokenizer,
    *,
    system_prompt: str | None = None,
    max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.2,
) -> str:
    """Run one instruct-style turn; returns assistant text only."""
    import torch

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt_text = f"{user_prompt}\n\nAssistant:"

    inputs = tokenizer(prompt_text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def smoke_test(model_path: Path | str | None = None) -> int:
    """Load model and print a one-line generation."""
    path = resolve_model_path(model_path)
    print(f"Model directory: {path}", flush=True)
    if not model_is_ready(path):
        print("STATUS: not downloaded (missing config.json)", flush=True)
        return 1

    model, tokenizer = load_llm(path)
    reply = chat_generate(
        "Reply with exactly: ok",
        model,
        tokenizer,
        system_prompt="You are a concise assistant.",
        max_new_tokens=32,
        temperature=0.0,
    )
    print(f"STATUS: ok\nSample output: {reply!r}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Local Llama 3.1 smoke test / readiness check.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help=f"Override model path (default: {settings.LLM_MODEL_DIR})",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only verify config.json exists; do not load weights.",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Load model and run one prompt.")
    args = parser.parse_args()

    if args.check_only:
        ok = model_is_ready(args.model_dir)
        print("ready" if ok else "missing", flush=True)
        return 0 if ok else 1

    if args.smoke_test:
        return smoke_test(args.model_dir)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
