"""Build mixed training manifest (all originals + emphasized hard rows)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.self_train.jsonl_io import load_jsonl_rows, write_jsonl_rows


def build_mixed_manifest(
    *,
    train_jsonl: Path,
    hard_jsonl: Path,
    refined_jsonl: Path | None,
    out_path: Path,
    iter_n: int,
    hard_weight: float = 2.0,
    dup_hard: bool = True,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """
    Union of full train set plus hard (and optionally refined) rows.

    When ``dup_hard`` is True, hard clips appear twice: once as ``source=orig`` and
    once as ``source=hard`` with higher ``weight`` (v1 emphasizes hard subset in sampling
    via duplication; training loss remains unweighted in ``model_creation``).
    """
    all_rows = load_jsonl_rows(train_jsonl, max_samples=max_samples)
    hard_rows = load_jsonl_rows(hard_jsonl, max_samples=None)
    hard_paths = {r["audio_path"] for r in hard_rows}

    refined_rows: list[dict[str, Any]] = []
    if refined_jsonl is not None and refined_jsonl.is_file() and refined_jsonl.stat().st_size > 0:
        refined_rows = load_jsonl_rows(
            refined_jsonl, max_samples=None, allow_empty=True
        )

    refined_by_path = {r["audio_path"]: r for r in refined_rows}

    mixed: list[dict[str, Any]] = []
    for row in all_rows:
        mixed.append(
            {
                "audio_path": row["audio_path"],
                "text": row["text"],
                "text_orig": row.get("text_orig", row["text"]),
                "text_source": row.get("text_source", "grok"),
                "source": "orig",
                "weight": 1.0,
                "iter": iter_n,
            }
        )

    n_hard_extra = 0
    for ap in hard_paths:
        ref = refined_by_path.get(ap)
        if ref is None:
            continue
        if dup_hard:
            mixed.append(
                {
                    "audio_path": ap,
                    "text": ref["text"],
                    "text_orig": ref.get("text_orig", ref["text"]),
                    "text_source": ref.get("text_source", "grok"),
                    "source": "hard",
                    "weight": hard_weight,
                    "iter": iter_n,
                    "sim": ref.get("sim"),
                    "error_score": ref.get("error_score"),
                }
            )
            n_hard_extra += 1

    write_jsonl_rows(out_path, mixed)
    return {
        "train_jsonl": str(train_jsonl),
        "hard_jsonl": str(hard_jsonl),
        "refined_jsonl": str(refined_jsonl) if refined_jsonl else None,
        "out_path": str(out_path),
        "n_orig": len(all_rows),
        "n_hard_unique": len(hard_paths),
        "n_hard_extra": n_hard_extra,
        "n_mixed": len(mixed),
        "iter": iter_n,
    }
