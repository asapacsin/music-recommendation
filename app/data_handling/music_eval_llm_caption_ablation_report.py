"""
Summarize LLM-caption vs original-caption fine-tune retrieval matrices.

Expects per-seed CSVs under ``data/eval/llm_ablation/``:
  - ``orig_seed{N}.csv``
  - ``llm_seed{N}.csv``

Writes ``summary_primary.csv``, ``summary.json``, and ``REPORT.md``.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
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
    "LLM arm uses gate-passed hard-clip caption replacement (~2.8% of train rows); "
    "not full-corpus LLM refinement."
)


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


def _parse_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for part in raw.replace(",", " ").split():
        part = part.strip()
        if part:
            seeds.append(int(part))
    if not seeds:
        raise ValueError("No seeds provided")
    return seeds


def _discover_arm_csvs(ablation_dir: Path, arm: str, seeds: list[int]) -> list[Path]:
    paths: list[Path] = []
    for seed in seeds:
        p = ablation_dir / f"{arm}_seed{seed}.csv"
        if not p.is_file():
            raise FileNotFoundError(f"Missing matrix CSV: {p}")
        paths.append(p)
    return paths


def write_report_md(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    top_k: int,
    seeds: list[int],
    ablation_dir: Path,
) -> None:
    lines = [
        "# LLM vs original caption ablation",
        "",
        ABLATION_NOTE,
        "",
        f"- Ablation dir: `{ablation_dir}`",
        f"- Seeds: {', '.join(str(s) for s in seeds)}",
        f"- Top-K: {top_k}",
        f"- Primary tags: piano, vocal, relaxing",
        "",
        "| Tag | Original P@K | LLM P@K | Δ precision | Original nDCG Δ | LLM nDCG Δ | Δ nDCG |",
        "|-----|--------------|---------|-------------|-----------------|------------|--------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['tag_id']} | {r['orig_precision_mean']:.3f} | {r['llm_precision_mean']:.3f} "
            f"| {r['delta_precision']:+.3f} | {r['orig_ndcg_delta_mean']:.3f} "
            f"| {r['llm_ndcg_delta_mean']:.3f} | {r['delta_ndcg']:+.3f} |"
        )
    lines.extend(["", "## Interpretation", ""])
    for r in rows:
        dp = r["delta_precision"]
        if dp > 0.01:
            verdict = "LLM arm better"
        elif dp < -0.01:
            verdict = "Original arm better"
        else:
            verdict = "Roughly tied"
        lines.append(f"- **{r['tag_id']}**: {verdict} (Δ precision {dp:+.3f})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary(
    *,
    ablation_dir: Path,
    seeds: list[int],
    top_k: int,
) -> dict[str, Any]:
    orig_paths = _discover_arm_csvs(ablation_dir, "orig", seeds)
    llm_paths = _discover_arm_csvs(ablation_dir, "llm", seeds)

    summary_rows: list[dict[str, Any]] = []
    csv_fieldnames = [
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

    tag_by_query = {q: tid for tid, q in PRIMARY_TAGS}
    primary_path = ablation_dir / "summary_primary.csv"
    primary_path.parent.mkdir(parents=True, exist_ok=True)

    with primary_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fieldnames)
        w.writeheader()
        for _tag_id, qt in PRIMARY_TAGS:
            orig_row = _pick_row(_load_matrix_csv(orig_paths[0]), qt, top_k)
            if orig_row is None:
                continue
            op = _metric_lists(orig_paths, qt, top_k, "precision_at_k")
            on = _metric_lists(orig_paths, qt, top_k, "ndcg_delta")
            lp = _metric_lists(llm_paths, qt, top_k, "precision_at_k")
            ln = _metric_lists(llm_paths, qt, top_k, "ndcg_delta")
            opm, ops = _mean_stdev(op)
            onm, ons = _mean_stdev(on)
            lpm, lps = _mean_stdev(lp)
            lnm, lns = _mean_stdev(ln)
            if opm is None or lpm is None or onm is None or lnm is None:
                continue
            row = {
                "tag_id": tag_by_query.get(qt, ""),
                "query_text": qt,
                "top_k": top_k,
                "n_positive": orig_row.get("n_positive", ""),
                "orig_precision_mean": f"{opm:.6f}",
                "orig_precision_stdev": f"{ops:.6f}",
                "orig_ndcg_delta_mean": f"{onm:.6f}",
                "orig_ndcg_delta_stdev": f"{ons:.6f}",
                "llm_precision_mean": f"{lpm:.6f}",
                "llm_precision_stdev": f"{lps:.6f}",
                "llm_ndcg_delta_mean": f"{lnm:.6f}",
                "llm_ndcg_delta_stdev": f"{lns:.6f}",
                "delta_precision": f"{lpm - opm:.6f}",
                "delta_ndcg": f"{lnm - onm:.6f}",
                "n_seeds": len(op),
            }
            w.writerow(row)
            summary_rows.append(
                {
                    "tag_id": row["tag_id"],
                    "query_text": qt,
                    "orig_precision_mean": opm,
                    "llm_precision_mean": lpm,
                    "delta_precision": lpm - opm,
                    "orig_ndcg_delta_mean": onm,
                    "llm_ndcg_delta_mean": lnm,
                    "delta_ndcg": lnm - onm,
                }
            )

    report_path = ablation_dir / "REPORT.md"
    write_report_md(
        report_path,
        rows=summary_rows,
        top_k=top_k,
        seeds=seeds,
        ablation_dir=ablation_dir,
    )

    meta: dict[str, Any] = {
        "ablation_dir": str(ablation_dir),
        "seeds": seeds,
        "top_k": top_k,
        "note": ABLATION_NOTE,
        "orig_csvs": [str(p) for p in orig_paths],
        "llm_csvs": [str(p) for p in llm_paths],
        "primary_tags": [{"tag_id": t, "query_text": q} for t, q in PRIMARY_TAGS],
        "summary_rows": summary_rows,
        "outputs": {
            "summary_primary": str(primary_path),
            "report_md": str(report_path),
        },
    }
    summary_json = ablation_dir / "summary.json"
    summary_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report: original-caption FT vs LLM-swapped-caption FT (per seed matrices)."
    )
    parser.add_argument(
        "--ablation-dir",
        type=Path,
        default=settings.DATA_DIR / "eval" / "llm_ablation",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="Comma- or space-separated seeds (default: 42,43,44).",
    )
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    meta = build_summary(
        ablation_dir=args.ablation_dir.resolve(),
        seeds=_parse_seeds(args.seeds),
        top_k=args.top_k,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
