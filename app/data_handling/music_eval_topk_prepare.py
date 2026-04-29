"""
Prepare Top-K retrieval candidates for human evaluation of style queries.

Workflow:
1) Load style queries (instrumentation, mood, energy, texture).
2) Run text retrieval on metadata FAISS index.
3) Write machine candidates JSONL and human labeling CSV template.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import faiss
import laion_clap
import numpy as np
from tqdm import tqdm

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import settings

DEFAULT_STYLE_QUERIES: list[dict[str, str]] = [
    {"query_id": "inst_piano", "query_type": "instrumentation", "label": "piano", "query_text": "piano music"},
    {"query_id": "inst_electronic", "query_type": "instrumentation", "label": "electronic", "query_text": "electronic music"},
    {"query_id": "inst_orchestral", "query_type": "instrumentation", "label": "orchestral", "query_text": "orchestral music"},
    {"query_id": "inst_vocal", "query_type": "instrumentation", "label": "vocal", "query_text": "vocal music"},
    {"query_id": "mood_happy", "query_type": "mood", "label": "happy", "query_text": "happy music"},
    {"query_id": "mood_sad_melancholic", "query_type": "mood", "label": "sad/melancholic", "query_text": "sad melancholic music"},
    {"query_id": "mood_energetic", "query_type": "mood", "label": "energetic", "query_text": "energetic music"},
    {"query_id": "mood_relaxing", "query_type": "mood", "label": "relaxing", "query_text": "relaxing music"},
    {"query_id": "mood_dark_tense", "query_type": "mood", "label": "dark/tense", "query_text": "dark tense music"},
    {"query_id": "energy_low", "query_type": "energy", "label": "low", "query_text": "low energy ambient chill music"},
    {"query_id": "energy_high", "query_type": "energy", "label": "high", "query_text": "high energy dance rock music"},
    {"query_id": "texture_solo", "query_type": "texture", "label": "solo-instrument", "query_text": "solo instrument music"},
    {"query_id": "texture_multi", "query_type": "texture", "label": "multi-instrument", "query_text": "multi instrument music"},
]


def _normalize_embeddings(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def _load_model() -> laion_clap.CLAP_Module:
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    model.load_ckpt(str(settings.CLAP_MODEL_FILE))
    return model


def _load_queries(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Query file not found: {path}. Run with --init-queries to create a template."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON array in {path}")

    rows: list[dict[str, str]] = []
    required = {"query_id", "query_type", "label", "query_text"}
    for item in payload:
        if not isinstance(item, dict):
            continue
        if not required.issubset(item.keys()):
            continue
        row = {k: str(item[k]).strip() for k in required}
        if all(row.values()):
            rows.append(row)
    if not rows:
        raise ValueError(f"No valid query rows found in {path}")
    return rows


def _init_queries_file(path: Path, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Query file already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(DEFAULT_STYLE_QUERIES, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_candidates(
    *,
    queries: list[dict[str, str]],
    top_k: int,
    run_id: str,
    index_path: Path,
    mapping_path: Path,
) -> list[dict[str, Any]]:
    index = faiss.read_index(str(index_path))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    model = _load_model()

    candidates: list[dict[str, Any]] = []
    for query in tqdm(queries, desc="Retrieving Top-K", unit="query"):
        query_embed = model.get_text_embedding(x=[query["query_text"]], use_tensor=False)
        query_embed = _normalize_embeddings(np.asarray(query_embed, dtype="float32"))
        distances, indices = index.search(query_embed, top_k)

        for rank, (score, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            row = mapping.get(str(int(idx)))
            if row is None:
                continue
            candidates.append(
                {
                    "run_id": run_id,
                    "query_id": query["query_id"],
                    "query_type": query["query_type"],
                    "query_label": query["label"],
                    "query_text": query["query_text"],
                    "rank": rank,
                    "score": float(score),
                    "audio": row.get("audio"),
                    "text": row.get("text"),
                    "mood": row.get("mood"),
                    "confidence": row.get("confidence"),
                }
            )
    return candidates


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_human_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "query_id",
        "query_type",
        "query_label",
        "query_text",
        "rank",
        "score",
        "audio",
        "text",
        "mood",
        "confidence",
        "relevance",
        "judge_confidence",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["relevance"] = ""
            out["judge_confidence"] = ""
            out["notes"] = ""
            writer.writerow(out)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Top-K retrieval sheet for human evaluation.")
    parser.add_argument(
        "--queries",
        type=Path,
        default=settings.DATA_DIR / "eval" / "style_queries.json",
        help="Style query JSON path.",
    )
    parser.add_argument(
        "--init-queries",
        action="store_true",
        help="Initialize style query file with default taxonomy queries.",
    )
    parser.add_argument(
        "--overwrite-queries",
        action="store_true",
        help="Allow overwriting existing query file when using --init-queries.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-K retrieval depth per query.")
    parser.add_argument("--run-id", default="", help="Run ID for tracking and comparison.")
    parser.add_argument(
        "--index",
        type=Path,
        default=settings.METADATA_TEXT_INDEX_FILE,
        help="Metadata FAISS index path.",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=settings.METADATA_ID_MAPPING_FILE,
        help="Metadata id mapping JSON path.",
    )
    parser.add_argument(
        "--out-candidates",
        type=Path,
        default=settings.DATA_DIR / "eval" / "top10_candidates.jsonl",
        help="Machine candidate JSONL output.",
    )
    parser.add_argument(
        "--out-human-csv",
        type=Path,
        default=settings.DATA_DIR / "eval" / "top10_human_labels.csv",
        help="Human labeling CSV template output.",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")

    if args.init_queries:
        _init_queries_file(args.queries, overwrite=args.overwrite_queries)
        print(f"Initialized query file: {args.queries}")
        return 0

    queries = _load_queries(args.queries)
    run_id = args.run_id.strip() or datetime.now().strftime("topk_%Y%m%d_%H%M%S")
    candidates = _build_candidates(
        queries=queries,
        top_k=args.top_k,
        run_id=run_id,
        index_path=args.index,
        mapping_path=args.mapping,
    )
    _write_jsonl(args.out_candidates, candidates)
    _write_human_csv(args.out_human_csv, candidates)

    print(
        json.dumps(
            {
                "run_id": run_id,
                "queries": len(queries),
                "top_k": args.top_k,
                "candidate_rows": len(candidates),
                "candidates_output": str(args.out_candidates),
                "human_csv_output": str(args.out_human_csv),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
