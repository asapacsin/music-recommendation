"""Shared JSONL load/write helpers for self-train manifests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import settings


def resolve_project_path(value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p.resolve()
    return (settings.BASE_DIR / p).resolve()


def load_jsonl_rows(
    jsonl_path: Path,
    *,
    require_audio: bool = True,
    max_samples: int | None = None,
    allow_empty: bool = False,
) -> list[dict[str, Any]]:
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            audio_value = row.get("audio_path")
            if not isinstance(audio_value, str) or not audio_value.strip():
                continue
            abs_path = resolve_project_path(audio_value)
            if require_audio and not abs_path.is_file():
                continue
            row = dict(row)
            row["audio_path"] = str(abs_path)
            if isinstance(row.get("text"), str):
                row["text"] = row["text"].strip() or "None"
            else:
                row["text"] = "None"
            rows.append(row)
            if max_samples is not None and len(rows) >= max_samples:
                break
    if not rows and not allow_empty:
        raise ValueError(f"No usable rows in {jsonl_path}")
    return rows


def write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
