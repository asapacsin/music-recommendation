"""
Build FAISS text indexes for gold retrieval eval (metadata or train captions).

Uses ``RAGWEB_CLAP_CHECKPOINT`` when set (via ``app.clap_eval_load``).

Subcommands:
  build-metadata  — ``music_metadata.json`` (Grok song descriptions)
  build-caption   — dedupe ``clap_train*.jsonl`` to one row per song (``source_path``)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from tqdm import tqdm

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import settings

from app.metadata_faiss import (
    _compose_text,
    _load_json_array,
    _safe_confidence,
)
from app.self_train.jsonl_io import resolve_project_path


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def _load_clap_model():
    from app.clap_eval_load import load_clap_module_httsat

    return load_clap_module_httsat()


def _song_audio_basename(source_path: str) -> str:
    return Path(str(source_path).strip().replace("\\", "/")).name


def collect_caption_rows_from_jsonl(
    jsonl_path: Path,
    *,
    min_confidence: float = 0.0,
) -> list[dict[str, Any]]:
    """
    One index row per ``source_path`` (first clip wins).

    Mapping ``audio`` field = song basename (matches gold ``source_path`` join).
    """
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    by_song: dict[str, dict[str, Any]] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            source_path = row.get("source_path")
            if not isinstance(source_path, str) or not source_path.strip():
                continue
            try:
                conf = float(row.get("confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0
            if conf < min_confidence:
                continue

            key = _norm_source_key(source_path)
            if key in by_song:
                continue

            text = str(row.get("text") or "").strip()
            mood = row.get("mood")
            by_song[key] = {
                "audio": _song_audio_basename(source_path),
                "text": text,
                "mood": mood if isinstance(mood, str) else None,
                "confidence": conf,
                "source_path": source_path,
            }

    if not by_song:
        raise ValueError(f"No caption rows in {jsonl_path}")

    return [by_song[k] for k in sorted(by_song)]


def _norm_source_key(source_path: str) -> str:
    return str(resolve_project_path(source_path))


def build_text_faiss_index(
    rows: list[dict[str, Any]],
    *,
    output_index_path: Path,
    output_mapping_path: Path,
    batch_size: int = 256,
    compose_mood: bool = True,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if not rows:
        raise ValueError("No rows to index")

    model = _load_clap_model()

    texts: list[str] = []
    mapping: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        if compose_mood:
            texts.append(_compose_text(row))
        else:
            texts.append(str(row.get("text") or "").strip() or "unknown music")
        mapping[str(idx)] = {
            "audio": row.get("audio"),
            "text": row.get("text"),
            "mood": row.get("mood"),
            "confidence": row.get("confidence", 0.0),
        }

    embedding_batches: list[np.ndarray] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for start in tqdm(
        range(0, len(texts), batch_size),
        total=total_batches,
        desc="Embedding index text",
        unit="batch",
    ):
        batch = texts[start : start + batch_size]
        batch_embed = model.get_text_embedding(x=batch, use_tensor=False)
        embedding_batches.append(np.asarray(batch_embed, dtype="float32"))

    embeddings = np.vstack(embedding_batches).astype("float32")
    embeddings = _normalize_embeddings(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    output_index_path.parent.mkdir(parents=True, exist_ok=True)
    output_mapping_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_index_path))
    with output_mapping_path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    import os

    ckpt = os.environ.get("RAGWEB_CLAP_CHECKPOINT", "")
    return {
        "output_index_path": str(output_index_path),
        "output_mapping_path": str(output_mapping_path),
        "indexed_rows": len(rows),
        "embedding_dim": int(embeddings.shape[1]),
        "checkpoint_env": ckpt or None,
    }


def build_caption_faiss_from_jsonl(
    *,
    jsonl_path: Path,
    output_index_path: Path,
    output_mapping_path: Path,
    min_confidence: float = 0.0,
    batch_size: int = 256,
) -> dict[str, Any]:
    rows = collect_caption_rows_from_jsonl(jsonl_path, min_confidence=min_confidence)
    stats = build_text_faiss_index(
        rows,
        output_index_path=output_index_path,
        output_mapping_path=output_mapping_path,
        batch_size=batch_size,
        compose_mood=True,
    )
    stats["jsonl_path"] = str(jsonl_path.resolve())
    stats["index_kind"] = "caption"
    return stats


def build_metadata_faiss_for_checkpoint(
    *,
    metadata_path: Path = settings.MUSIC_METADATA_FILE,
    output_index_path: Path,
    output_mapping_path: Path,
    min_confidence: float = 0.35,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Rebuild metadata index using current ``RAGWEB_CLAP_CHECKPOINT`` text encoder."""
    rows = _load_json_array(metadata_path)
    filtered = [r for r in rows if _safe_confidence(r) >= min_confidence]
    if not filtered:
        raise ValueError("No metadata rows matched confidence filter")

    index_rows: list[dict[str, Any]] = []
    for row in filtered:
        audio = row.get("audio")
        if not isinstance(audio, str) or not audio.strip():
            continue
        index_rows.append(
            {
                "audio": audio,
                "text": row.get("text"),
                "mood": row.get("mood"),
                "confidence": _safe_confidence(row),
            }
        )

    stats = build_text_faiss_index(
        index_rows,
        output_index_path=output_index_path,
        output_mapping_path=output_mapping_path,
        batch_size=batch_size,
        compose_mood=True,
    )
    stats["metadata_path"] = str(metadata_path.resolve())
    stats["index_kind"] = "metadata"
    stats["min_confidence"] = min_confidence
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build FAISS text index for retrieval eval (checkpoint-aware)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    meta_p = sub.add_parser("build-metadata", help="Index Grok metadata text")
    meta_p.add_argument(
        "--metadata",
        type=Path,
        default=settings.MUSIC_METADATA_FILE,
    )
    meta_p.add_argument("--out-index", type=Path, required=True)
    meta_p.add_argument("--out-mapping", type=Path, required=True)
    meta_p.add_argument("--min-confidence", type=float, default=0.35)
    meta_p.add_argument("--batch-size", type=int, default=256)

    cap_p = sub.add_parser("build-caption", help="Index train JSONL captions (one row/song)")
    cap_p.add_argument("--jsonl", type=Path, required=True)
    cap_p.add_argument("--out-index", type=Path, required=True)
    cap_p.add_argument("--out-mapping", type=Path, required=True)
    cap_p.add_argument("--min-confidence", type=float, default=0.0)
    cap_p.add_argument("--batch-size", type=int, default=256)

    args = parser.parse_args()

    if args.command == "build-metadata":
        stats = build_metadata_faiss_for_checkpoint(
            metadata_path=args.metadata,
            output_index_path=args.out_index,
            output_mapping_path=args.out_mapping,
            min_confidence=args.min_confidence,
            batch_size=args.batch_size,
        )
    elif args.command == "build-caption":
        stats = build_caption_faiss_from_jsonl(
            jsonl_path=args.jsonl,
            output_index_path=args.out_index,
            output_mapping_path=args.out_mapping,
            min_confidence=args.min_confidence,
            batch_size=args.batch_size,
        )
    else:
        parser.error(f"Unknown command: {args.command}")
        return 2

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
