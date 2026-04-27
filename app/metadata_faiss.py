from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import faiss
import laion_clap
import numpy as np
from tqdm import tqdm

from config import settings


def _load_json_array(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}")
    return [row for row in data if isinstance(row, dict)]


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def _load_clap_model() -> laion_clap.CLAP_Module:
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    model.load_ckpt(str(settings.CLAP_MODEL_FILE))
    return model


def _compose_text(row: dict[str, Any]) -> str:
    text = str(row.get("text") or "").strip()
    mood = row.get("mood")
    if isinstance(mood, str):
        mood = mood.strip()

    if text and mood:
        return f"{text}. mood: {mood}"
    if text:
        return text
    if mood:
        return f"music mood {mood}"
    return "unknown music"


def _safe_confidence(row: dict[str, Any]) -> float:
    try:
        return float(row.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0


def build_metadata_faiss_index(
    *,
    metadata_path: Path = settings.MUSIC_METADATA_FILE,
    output_index_path: Path = settings.METADATA_TEXT_INDEX_FILE,
    output_mapping_path: Path = settings.METADATA_ID_MAPPING_FILE,
    min_confidence: float = 0.0,
    batch_size: int = 256,
) -> dict[str, Any]:
    """
    Build a FAISS text index from metadata records.

    Each indexed row stores minimal retrieval payload in mapping:
    audio, text, mood, confidence.
    """
    if not (0.0 <= min_confidence <= 1.0):
        raise ValueError("min_confidence must be in [0, 1]")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    rows = _load_json_array(metadata_path)
    filtered_rows = [row for row in rows if _safe_confidence(row) >= min_confidence]
    if not filtered_rows:
        raise ValueError("No rows matched the confidence filter; index not built.")

    model = _load_clap_model()

    texts: list[str] = []
    mapping: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(filtered_rows):
        texts.append(_compose_text(row))
        mapping[str(idx)] = {
            "audio": row.get("audio"),
            "text": row.get("text"),
            "mood": row.get("mood"),
            "confidence": _safe_confidence(row),
        }

    embedding_batches: list[np.ndarray] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for start in tqdm(
        range(0, len(texts), batch_size),
        total=total_batches,
        desc="Embedding metadata",
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

    return {
        "metadata_path": str(metadata_path),
        "output_index_path": str(output_index_path),
        "output_mapping_path": str(output_mapping_path),
        "total_rows": len(rows),
        "indexed_rows": len(filtered_rows),
        "min_confidence": min_confidence,
        "embedding_dim": int(embeddings.shape[1]),
    }


def search_metadata_text(
    query_text: str,
    *,
    top_k: int = 5,
    index_path: Path = settings.METADATA_TEXT_INDEX_FILE,
    mapping_path: Path = settings.METADATA_ID_MAPPING_FILE,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if not query_text.strip():
        raise ValueError("query_text cannot be empty")

    index = faiss.read_index(str(index_path))
    with mapping_path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)

    model = _load_clap_model()
    query_embed = model.get_text_embedding(x=[query_text], use_tensor=False)
    query_embed = _normalize_embeddings(np.asarray(query_embed, dtype="float32"))

    distances, indices = index.search(query_embed, top_k)
    results: list[dict[str, Any]] = []
    for score, idx in zip(distances[0], indices[0]):
        row = mapping.get(str(int(idx)))
        if row is None:
            continue
        results.append({"score": float(score), **row})
    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build/search FAISS index from music metadata text fields."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build metadata FAISS index")
    build_parser.add_argument(
        "--metadata",
        type=Path,
        default=settings.MUSIC_METADATA_FILE,
        help="Input metadata JSON array path",
    )
    build_parser.add_argument(
        "--out-index",
        type=Path,
        default=settings.METADATA_TEXT_INDEX_FILE,
        help="Output metadata FAISS index path",
    )
    build_parser.add_argument(
        "--out-mapping",
        type=Path,
        default=settings.METADATA_ID_MAPPING_FILE,
        help="Output id->metadata mapping path",
    )
    build_parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Only index rows with confidence >= this value",
    )
    build_parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Text embedding batch size",
    )

    search_parser = subparsers.add_parser("search", help="Query metadata FAISS index")
    search_parser.add_argument("--query", required=True, help="Text query")
    search_parser.add_argument("--top-k", type=int, default=5, help="Top-k results")
    search_parser.add_argument(
        "--index",
        type=Path,
        default=settings.METADATA_TEXT_INDEX_FILE,
        help="Metadata FAISS index path",
    )
    search_parser.add_argument(
        "--mapping",
        type=Path,
        default=settings.METADATA_ID_MAPPING_FILE,
        help="Metadata id->row mapping path",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.command == "build":
        stats = build_metadata_faiss_index(
            metadata_path=args.metadata,
            output_index_path=args.out_index,
            output_mapping_path=args.out_mapping,
            min_confidence=args.min_confidence,
            batch_size=args.batch_size,
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0

    if args.command == "search":
        results = search_metadata_text(
            args.query,
            top_k=args.top_k,
            index_path=args.index,
            mapping_path=args.mapping,
        )
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
