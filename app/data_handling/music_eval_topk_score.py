"""
Score human-labeled Top-K retrieval sheet.

Input:
  - CSV produced by music_eval_topk_prepare.py and filled by human

Output:
  - JSON metrics summary (overall + by query_type + by query_id)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import settings


def _parse_relevance(value: str) -> float | None:
    v = value.strip()
    if not v:
        return None
    try:
        score = float(v)
    except ValueError:
        return None
    if score < 0:
        return None
    return score


def _dcg(scores: list[float]) -> float:
    total = 0.0
    for i, rel in enumerate(scores, start=1):
        total += (2.0**rel - 1.0) / math.log2(i + 1)
    return total


def _score_rows(rows: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        qid = row["query_id"]
        grouped.setdefault(qid, []).append(row)

    per_query: list[dict[str, Any]] = []
    by_type_acc: dict[str, list[dict[str, float]]] = {}
    overall_precisions: list[float] = []
    overall_hits: list[float] = []
    overall_ndcgs: list[float] = []

    for query_id, qrows in grouped.items():
        qrows_sorted = sorted(qrows, key=lambda r: int(r["rank"]))[:top_k]
        judged = [r for r in qrows_sorted if r["relevance"] is not None]
        if not judged:
            continue
        binary = [1.0 if r["relevance"] > 0 else 0.0 for r in judged]
        precision_k = sum(binary) / len(judged)
        hitrate_k = 1.0 if any(x > 0 for x in binary) else 0.0

        judged_rels = [float(r["relevance"]) for r in judged]
        dcg = _dcg(judged_rels)
        ideal = sorted(judged_rels, reverse=True)
        idcg = _dcg(ideal)
        ndcg_k = (dcg / idcg) if idcg > 0 else 0.0

        query_type = judged[0]["query_type"]
        row = {
            "query_id": query_id,
            "query_type": query_type,
            "num_judged": len(judged),
            "precision_at_k": precision_k,
            "hitrate_at_k": hitrate_k,
            "ndcg_at_k": ndcg_k,
        }
        per_query.append(row)
        by_type_acc.setdefault(query_type, []).append(
            {
                "precision_at_k": precision_k,
                "hitrate_at_k": hitrate_k,
                "ndcg_at_k": ndcg_k,
            }
        )
        overall_precisions.append(precision_k)
        overall_hits.append(hitrate_k)
        overall_ndcgs.append(ndcg_k)

    by_type: dict[str, dict[str, float]] = {}
    for query_type, vals in by_type_acc.items():
        by_type[query_type] = {
            "queries": float(len(vals)),
            "precision_at_k": sum(v["precision_at_k"] for v in vals) / len(vals),
            "hitrate_at_k": sum(v["hitrate_at_k"] for v in vals) / len(vals),
            "ndcg_at_k": sum(v["ndcg_at_k"] for v in vals) / len(vals),
        }

    return {
        "top_k": top_k,
        "num_queries_scored": len(per_query),
        "overall": {
            "precision_at_k": (sum(overall_precisions) / len(overall_precisions))
            if overall_precisions
            else 0.0,
            "hitrate_at_k": (sum(overall_hits) / len(overall_hits)) if overall_hits else 0.0,
            "ndcg_at_k": (sum(overall_ndcgs) / len(overall_ndcgs)) if overall_ndcgs else 0.0,
        },
        "by_type": by_type,
        "by_query": sorted(per_query, key=lambda x: x["query_id"]),
    }


def _load_human_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Human label CSV not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            query_id = (raw.get("query_id") or "").strip()
            query_type = (raw.get("query_type") or "").strip()
            rank = (raw.get("rank") or "").strip()
            if not query_id or not query_type or not rank:
                continue
            try:
                rank_num = int(rank)
            except ValueError:
                continue
            rows.append(
                {
                    "query_id": query_id,
                    "query_type": query_type,
                    "rank": rank_num,
                    "relevance": _parse_relevance(raw.get("relevance", "")),
                }
            )
    return rows


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score human-labeled Top-K retrieval sheet.")
    parser.add_argument(
        "--human-csv",
        type=Path,
        default=settings.DATA_DIR / "eval" / "top10_human_labels.csv",
        help="Human-labeled CSV path.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-K cut used for scoring.")
    parser.add_argument(
        "--out-metrics",
        type=Path,
        default=settings.DATA_DIR / "eval" / "top10_metrics.json",
        help="Metrics JSON output path.",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")

    rows = _load_human_csv(args.human_csv)
    metrics = _score_rows(rows, top_k=args.top_k)
    metrics.update(
        {
            "input_human_csv": str(args.human_csv),
            "num_rows_loaded": len(rows),
        }
    )

    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.out_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved metrics: {args.out_metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
