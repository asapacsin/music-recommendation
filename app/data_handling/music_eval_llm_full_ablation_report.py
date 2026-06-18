"""
Report for full-corpus LLM ablation with symmetric per-checkpoint FAISS indexes.

Expects CSVs under ``data/eval/llm_full_ablation/``:
  - ``orig_meta_seed{N}.csv`` / ``llm_meta_seed{N}.csv``  (metadata index)
  - ``orig_caption_seed{N}.csv`` / ``llm_caption_seed{N}.csv``  (caption index)

Writes ``summary_primary.csv``, ``summary_by_index.csv``, ``REPORT.md``, ``summary.json``.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import settings

from app.data_handling.music_eval_ablation_report import (
    PRIMARY_TAGS,
    _load_matrix_csv,
    _mean_stdev,
    _pick_row,
)

ABLATION_NOTE = (
    "Full-corpus tag-aware LLM caption rewrite (one call per song); "
    "eval uses per-checkpoint FAISS rebuild (metadata and/or caption index)."
)

INDEX_KINDS = ("meta", "caption")


def _parse_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for part in raw.replace(",", " ").split():
        part = part.strip()
        if part:
            seeds.append(int(part))
    if not seeds:
        raise ValueError("No seeds provided")
    return seeds


def _discover_csvs(ablation_dir: Path, arm: str, index_kind: str, seeds: list[int]) -> list[Path]:
    paths: list[Path] = []
    for seed in seeds:
        p = ablation_dir / f"{arm}_{index_kind}_seed{seed}.csv"
        if not p.is_file():
            raise FileNotFoundError(f"Missing matrix CSV: {p}")
        paths.append(p)
    return paths


def _metric_lists(
    csv_paths: list[Path],
    query_text: str,
    top_k: int,
    field: str,
) -> list[float]:
    out: list[float] = []
    for p in csv_paths:
        row = _pick_row(_load_matrix_csv(p), query_text, top_k)
        if row is not None:
            out.append(float(row[field]))
    return out


def _summarize_arm_pair(
    *,
    orig_paths: list[Path],
    llm_paths: list[Path],
    query_text: str,
    top_k: int,
    tag_id: str,
) -> dict[str, Any]:
    orig_row = _pick_row(_load_matrix_csv(orig_paths[0]), query_text, top_k)
    if orig_row is None:
        raise ValueError(f"Query {query_text!r} not found in {orig_paths[0]}")

    op = _metric_lists(orig_paths, query_text, top_k, "precision_at_k")
    on = _metric_lists(orig_paths, query_text, top_k, "ndcg_delta")
    lp = _metric_lists(llm_paths, query_text, top_k, "precision_at_k")
    ln = _metric_lists(llm_paths, query_text, top_k, "ndcg_delta")
    opm, ops = _mean_stdev(op)
    onm, ons = _mean_stdev(on)
    lpm, lps = _mean_stdev(lp)
    lnm, lns = _mean_stdev(ln)
    if opm is None or lpm is None or onm is None or lnm is None:
        raise ValueError(f"Insufficient metrics for {query_text}")

    return {
        "tag_id": tag_id,
        "query_text": query_text,
        "top_k": top_k,
        "n_positive": orig_row.get("n_positive", ""),
        "orig_precision_mean": opm,
        "orig_precision_stdev": ops,
        "orig_ndcg_delta_mean": onm,
        "orig_ndcg_delta_stdev": ons,
        "llm_precision_mean": lpm,
        "llm_precision_stdev": lps,
        "llm_ndcg_delta_mean": lnm,
        "llm_ndcg_delta_stdev": lns,
        "delta_precision": lpm - opm,
        "delta_ndcg": lnm - onm,
        "n_seeds": len(op),
    }


def build_summary(
    *,
    ablation_dir: Path,
    seeds: list[int],
    top_k: int,
    index_kinds: list[str] | None = None,
) -> dict[str, Any]:
    kinds = index_kinds or list(INDEX_KINDS)
    by_index: dict[str, list[dict[str, Any]]] = {}

    for kind in kinds:
        orig_paths = _discover_csvs(ablation_dir, "orig", kind, seeds)
        llm_paths = _discover_csvs(ablation_dir, "llm", kind, seeds)
        rows: list[dict[str, Any]] = []
        for tag_id, qt in PRIMARY_TAGS:
            rows.append(
                _summarize_arm_pair(
                    orig_paths=orig_paths,
                    llm_paths=llm_paths,
                    query_text=qt,
                    top_k=top_k,
                    tag_id=tag_id,
                )
            )
        by_index[kind] = rows

    # Primary CSV: metadata index (headline)
    primary_path = ablation_dir / "summary_primary.csv"
    _write_summary_csv(primary_path, by_index.get("meta", []))

    by_index_path = ablation_dir / "summary_by_index.csv"
    _write_by_index_csv(by_index_path, by_index)

    report_path = ablation_dir / "REPORT.md"
    _write_report_md(report_path, by_index=by_index, top_k=top_k, seeds=seeds, ablation_dir=ablation_dir)

    meta: dict[str, Any] = {
        "ablation_dir": str(ablation_dir),
        "seeds": seeds,
        "top_k": top_k,
        "note": ABLATION_NOTE,
        "index_kinds": kinds,
        "by_index": {
            kind: [
                {k: v for k, v in r.items() if k not in ("orig_precision_stdev", "llm_precision_stdev")}
                for r in rows
            ]
            for kind, rows in by_index.items()
        },
        "outputs": {
            "summary_primary": str(primary_path),
            "summary_by_index": str(by_index_path),
            "report_md": str(report_path),
        },
    }
    summary_json = ablation_dir / "summary.json"
    summary_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "tag_id",
        "query_text",
        "top_k",
        "n_positive",
        "orig_precision_mean",
        "orig_precision_stdev",
        "orig_ndcg_delta_mean",
        "orig_ndcg_delta_stdev",
        "llm_precision_mean",
        "llm_precision_stdev",
        "llm_ndcg_delta_mean",
        "llm_ndcg_delta_stdev",
        "delta_precision",
        "delta_ndcg",
        "n_seeds",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.6f}" if isinstance(r[k], float) else r[k]) for k in fieldnames})


def _write_by_index_csv(path: Path, by_index: dict[str, list[dict[str, Any]]]) -> None:
    fieldnames = [
        "index_kind",
        "tag_id",
        "query_text",
        "orig_precision_mean",
        "llm_precision_mean",
        "delta_precision",
        "orig_ndcg_delta_mean",
        "llm_ndcg_delta_mean",
        "delta_ndcg",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for kind, rows in by_index.items():
            for r in rows:
                w.writerow(
                    {
                        "index_kind": kind,
                        "tag_id": r["tag_id"],
                        "query_text": r["query_text"],
                        "orig_precision_mean": f"{r['orig_precision_mean']:.6f}",
                        "llm_precision_mean": f"{r['llm_precision_mean']:.6f}",
                        "delta_precision": f"{r['delta_precision']:.6f}",
                        "orig_ndcg_delta_mean": f"{r['orig_ndcg_delta_mean']:.6f}",
                        "llm_ndcg_delta_mean": f"{r['llm_ndcg_delta_mean']:.6f}",
                        "delta_ndcg": f"{r['delta_ndcg']:.6f}",
                    }
                )


def _write_report_md(
    path: Path,
    *,
    by_index: dict[str, list[dict[str, Any]]],
    top_k: int,
    seeds: list[int],
    ablation_dir: Path,
) -> None:
    lines = [
        "# Full-corpus LLM vs original caption ablation",
        "",
        ABLATION_NOTE,
        "",
        f"- Ablation dir: `{ablation_dir}`",
        f"- Seeds: {', '.join(str(s) for s in seeds)}",
        f"- Top-K: {top_k}",
        "",
    ]
    for kind, rows in by_index.items():
        label = "Metadata index (Grok catalog)" if kind == "meta" else "Caption index (train JSONL text)"
        lines.extend(
            [
                f"## {label}",
                "",
                "| Tag | Original P@K | LLM P@K | Δ precision | Original nDCG Δ | LLM nDCG Δ | Δ nDCG |",
                "|-----|--------------|---------|-------------|-----------------|------------|--------|",
            ]
        )
        for r in rows:
            lines.append(
                f"| {r['tag_id']} | {r['orig_precision_mean']:.3f} | {r['llm_precision_mean']:.3f} "
                f"| {r['delta_precision']:+.3f} | {r['orig_ndcg_delta_mean']:.3f} "
                f"| {r['llm_ndcg_delta_mean']:.3f} | {r['delta_ndcg']:+.3f} |"
            )
        lines.extend(["", "### Interpretation", ""])
        for r in rows:
            dp = r["delta_precision"]
            if dp > 0.01:
                verdict = "LLM arm better"
            elif dp < -0.01:
                verdict = "Original arm better"
            else:
                verdict = "Roughly tied"
            lines.append(f"- **{r['tag_id']}** ({kind}): {verdict} (Δ precision {dp:+.3f})")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Full LLM ablation report (meta + caption indexes).")
    parser.add_argument(
        "--ablation-dir",
        type=Path,
        default=settings.DATA_DIR / "eval" / "llm_full_ablation",
    )
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--index-kinds",
        type=str,
        default="meta,caption",
        help="Comma-separated: meta, caption",
    )
    args = parser.parse_args()

    kinds = [k.strip() for k in args.index_kinds.split(",") if k.strip()]
    meta = build_summary(
        ablation_dir=args.ablation_dir.resolve(),
        seeds=_parse_seeds(args.seeds),
        top_k=args.top_k,
        index_kinds=kinds,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
