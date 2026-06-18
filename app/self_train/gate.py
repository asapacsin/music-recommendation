"""CLAP gates for LLM caption refinement (similarity gain + text drift)."""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from app.init_model import embed_pipeline, normalize_embeddings


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return float(raw)


def default_gate_params() -> dict[str, float]:
    return {
        "min_sim_gain": _env_float("RAGWEB_REFINE_MIN_SIM_GAIN", 0.0),
        "min_text_cos": _env_float("RAGWEB_REFINE_MIN_TEXT_COS", 0.85),
    }


def diagonal_sim(audio_path: str, text: str, model) -> float:
    """Cosine similarity between audio and text embeddings (single pair)."""
    audio_embed, text_embed = embed_pipeline(
        [audio_path], model, tensor_mode=False, texts=[text]
    )
    audio_np = normalize_embeddings(np.array(audio_embed, dtype=np.float32))
    text_np = normalize_embeddings(np.array(text_embed, dtype=np.float32))
    return float(np.dot(audio_np[0], text_np[0]))


def text_cosine(text_a: str, text_b: str, model) -> float:
    """Cosine similarity between two text embeddings."""
    text_embed = model.get_text_embedding(x=[text_a, text_b], use_tensor=False)
    text_np = normalize_embeddings(np.array(text_embed, dtype=np.float32))
    return float(np.dot(text_np[0], text_np[1]))


def passes_gate(
    audio_path: str,
    text_old: str,
    text_new: str,
    model,
    *,
    min_sim_gain: float = 0.0,
    min_text_cos: float = 0.85,
    require_strict_sim_gain: bool = True,
) -> tuple[bool, dict[str, Any]]:
    """
  Return (accepted, diagnostics).

  Accept when:
    - sim_new > sim_old + min_sim_gain (strict > if require_strict_sim_gain)
    - text_cos(text_old, text_new) >= min_text_cos
    """
    sim_old = diagonal_sim(audio_path, text_old, model)
    sim_new = diagonal_sim(audio_path, text_new, model)
    t_cos = text_cosine(text_old, text_new, model)

    if require_strict_sim_gain:
        sim_ok = sim_new > sim_old + min_sim_gain
    else:
        sim_ok = sim_new >= sim_old + min_sim_gain

    drift_ok = t_cos >= min_text_cos
    accepted = sim_ok and drift_ok

    diag: dict[str, Any] = {
        "sim_old": sim_old,
        "sim_new": sim_new,
        "sim_gain": sim_new - sim_old,
        "text_cos": t_cos,
        "accepted": accepted,
        "reject_reason": None,
    }
    if not accepted:
        if not sim_ok:
            diag["reject_reason"] = "sim"
        elif not drift_ok:
            diag["reject_reason"] = "drift"
    return accepted, diag
