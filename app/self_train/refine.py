"""Caption refinement (NoOp v1, LLM + CLAP gate v2)."""
from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.init_model import load_model_from_checkpoint
from app.llm_local import chat_generate, load_llm
from app.self_train.gate import default_gate_params, passes_gate
from app.self_train.jsonl_io import load_jsonl_rows, write_jsonl_rows

_REFINE_SYSTEM = (
    "You write short English captions for music retrieval. "
    "Do not invent artist names, song titles, or lyrics. "
    "One or two sentences."
)


def build_refine_user_prompt(row: dict[str, Any]) -> str:
    """Build LLM user prompt from a hard-mined JSONL row."""
    text_orig = row.get("text_orig", row.get("text", ""))
    mood = row.get("mood")
    confidence = row.get("confidence")
    source_path = row.get("source_path", "")
    sim = row.get("sim")
    parts = [
        "Improve this music caption so it better matches the audio for retrieval.",
        f"Original caption: {text_orig}",
    ]
    if mood is not None and str(mood).strip():
        parts.append(f"Mood hint: {mood}")
    if confidence is not None:
        parts.append(f"Metadata confidence: {confidence}")
    if source_path:
        parts.append(f"Source track: {source_path}")
    if sim is not None:
        parts.append(f"Current CLAP audio-text similarity: {sim:.4f}")
    parts.append("Return only the revised caption.")
    return "\n".join(parts)


class Refiner(ABC):
    @abstractmethod
    def refine(self, row: dict[str, Any], context: dict[str, Any]) -> str | None:
        """Return refined caption text, or None to drop the row."""

    def close(self) -> None:
        """Release heavy resources (LLM)."""


class NoOpRefiner(Refiner):
    """Pass through original text (writes ``text`` unchanged, ``text_source`` preserved)."""

    def refine(self, row: dict[str, Any], context: dict[str, Any]) -> str | None:
        return row.get("text_orig", row.get("text", "None"))


class LlmRefiner(Refiner):
    """Llama caption refine + CLAP gate on hard-mined rows."""

    def __init__(
        self,
        clap_model,
        *,
        gate_params: dict[str, float] | None = None,
        llm_params: dict[str, Any] | None = None,
    ) -> None:
        self.clap_model = clap_model
        self.gate_params = gate_params or default_gate_params()
        self.llm_params = llm_params or {}
        self._llm_model = None
        self._llm_tokenizer = None
        self.stats = {
            "n_llm_calls": 0,
            "n_accepted": 0,
            "n_rejected_sim": 0,
            "n_rejected_drift": 0,
            "n_rejected_empty": 0,
        }

    @classmethod
    def create(
        cls,
        *,
        init_checkpoint: str | Path | None = None,
        gate_params: dict[str, float] | None = None,
        llm_params: dict[str, Any] | None = None,
    ) -> LlmRefiner:
        clap = load_model_from_checkpoint(
            str(init_checkpoint) if init_checkpoint else None
        )
        return cls(clap_model=clap, gate_params=gate_params, llm_params=llm_params)

    def _ensure_llm(self) -> None:
        if self._llm_model is None:
            self._llm_model, self._llm_tokenizer = load_llm(
                self.llm_params.get("model_path"),
                load_in_4bit=self.llm_params.get("load_in_4bit"),
            )

    def refine(self, row: dict[str, Any], context: dict[str, Any]) -> str | None:
        self._ensure_llm()
        text_orig = row.get("text_orig", row.get("text", "None"))
        user_prompt = build_refine_user_prompt(row)
        max_new_tokens = int(self.llm_params.get("max_new_tokens", 256))
        temperature = float(self.llm_params.get("temperature", 0.2))

        caption = chat_generate(
            user_prompt,
            self._llm_model,
            self._llm_tokenizer,
            system_prompt=_REFINE_SYSTEM,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        self.stats["n_llm_calls"] += 1

        if not caption or not caption.strip():
            self.stats["n_rejected_empty"] += 1
            return None

        caption = caption.strip()
        accepted, diag = passes_gate(
            row["audio_path"],
            text_orig,
            caption,
            self.clap_model,
            min_sim_gain=float(self.gate_params.get("min_sim_gain", 0.0)),
            min_text_cos=float(self.gate_params.get("min_text_cos", 0.85)),
        )
        if not accepted:
            reason = diag.get("reject_reason")
            if reason == "sim":
                self.stats["n_rejected_sim"] += 1
            elif reason == "drift":
                self.stats["n_rejected_drift"] += 1
            return None

        self.stats["n_accepted"] += 1
        row["_gate_diag"] = diag
        return caption

    def close(self) -> None:
        self._llm_model = None
        self._llm_tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def get_refiner(
    name: str,
    *,
    init_checkpoint: str | Path | None = None,
    gate_params: dict[str, float] | None = None,
    llm_params: dict[str, Any] | None = None,
) -> Refiner:
    if name in ("noop", "none", "no_refine"):
        return NoOpRefiner()
    if name == "llm":
        return LlmRefiner.create(
            init_checkpoint=init_checkpoint,
            gate_params=gate_params,
            llm_params=llm_params,
        )
    raise ValueError(f"Unknown refiner: {name!r}. Use 'noop' or 'llm'.")


def refine_hard_manifest(
    *,
    hard_jsonl: Path,
    out_path: Path,
    refiner: Refiner,
    iter_n: int,
    text_source: str = "grok",
    refine_max_hard: int | None = None,
) -> dict[str, Any]:
    rows = load_jsonl_rows(hard_jsonl, require_audio=True)
    if refine_max_hard is not None and refine_max_hard > 0:
        rows = rows[:refine_max_hard]

    refined: list[dict[str, Any]] = []
    dropped = 0
    context: dict[str, Any] = {"iter": iter_n}
    for row in rows:
        text = refiner.refine(row, context)
        if text is None:
            dropped += 1
            continue
        out = dict(row)
        out["text"] = text
        out["text_orig"] = row.get("text_orig", row.get("text"))
        out["text_source"] = text_source
        out["iter"] = iter_n
        if "_gate_diag" in row:
            out["gate"] = row["_gate_diag"]
        refined.append(out)

    write_jsonl_rows(out_path, refined)

    summary: dict[str, Any] = {
        "hard_jsonl": str(hard_jsonl),
        "out_path": str(out_path),
        "n_in": len(rows),
        "n_out": len(refined),
        "n_dropped": dropped,
        "refine_max_hard": refine_max_hard,
    }
    if isinstance(refiner, LlmRefiner):
        summary["llm_stats"] = dict(refiner.stats)
    return summary
