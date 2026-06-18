"""Hard-pair mining via diagonal audio–text cosine similarity."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.init_model import embed_pipeline, load_model_from_checkpoint, normalize_embeddings
from app.self_train.jsonl_io import load_jsonl_rows, write_jsonl_rows


def _diagonal_cosine_similarities(
    model,
    rows: list[dict[str, Any]],
    *,
    batch_size: int = 8,
) -> list[float]:
    sims: list[float] = []
    paths = [r["audio_path"] for r in rows]
    texts = [r["text"] for r in rows]
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        batch_texts = texts[i : i + batch_size]
        audio_embed, text_embed = embed_pipeline(
            batch_paths, model, tensor_mode=False, texts=batch_texts
        )
        audio_np = normalize_embeddings(np.array(audio_embed, dtype=np.float32))
        text_np = normalize_embeddings(np.array(text_embed, dtype=np.float32))
        for j in range(len(batch_paths)):
            sims.append(float(np.dot(audio_np[j], text_np[j])))
    return sims


def mine_hard_pairs(
    *,
    train_jsonl: Path,
    out_path: Path,
    init_checkpoint: Path | None = None,
    hard_frac: float = 0.2,
    batch_size: int = 8,
    max_samples: int | None = None,
    text_source: str = "grok",
) -> dict[str, Any]:
    """
    Score each row with diagonal cos(audio, text); export bottom ``hard_frac`` fraction.

    Returns summary dict with counts and paths.
    """
    if not 0 < hard_frac <= 1:
        raise ValueError(f"hard_frac must be in (0, 1], got {hard_frac}")

    rows = load_jsonl_rows(train_jsonl, max_samples=max_samples)
    model = load_model_from_checkpoint(
        str(init_checkpoint) if init_checkpoint else None
    )

    sims = _diagonal_cosine_similarities(model, rows, batch_size=batch_size)
    n = len(rows)
    order = sorted(range(n), key=lambda i: sims[i])
    n_hard = max(1, math.ceil(n * hard_frac))
    hard_indices = set(order[:n_hard])

    scored: list[dict[str, Any]] = []
    for rank_i, idx in enumerate(order):
        row = dict(rows[idx])
        sim = sims[idx]
        row["sim"] = sim
        row["error_score"] = 1.0 - sim
        row["rank_pct"] = (rank_i + 1) / n
        row["text_orig"] = row.get("text_orig", row["text"])
        row["text_source"] = text_source
        row["is_hard"] = idx in hard_indices
        scored.append(row)

    hard_rows = [r for r in scored if r["is_hard"]]
    write_jsonl_rows(out_path, hard_rows)

    mean_sim = float(np.mean(sims)) if sims else 0.0
    hard_mean = float(np.mean([r["sim"] for r in hard_rows])) if hard_rows else 0.0
    return {
        "train_jsonl": str(train_jsonl),
        "out_path": str(out_path),
        "n_total": n,
        "n_hard": len(hard_rows),
        "hard_frac": hard_frac,
        "mean_sim_all": mean_sim,
        "mean_sim_hard": hard_mean,
        "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
    }
