"""
Report for public Jamendo retrieval matrices (OOD).

Expects CSVs under ``data/eval/jamendo_public/``:
  ``{arm}_seed{N}.csv``  e.g. ``pretrained_seed42.csv``, ``thesis_ft_v1_seed43.csv``

Writes ``REPORT.md``, ``summary_primary.csv``, ``summary.json``.
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
    "MTG-Jamendo five-tag OOD eval (split-0 test manifest). "
    "Audio–text ranking on manifest pool; labels gold_pub_*; "
    "queries aligned with in-domain primary tags (piano, vocal, relaxing)."
)


def _parse_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for part in raw.replace(",", " ").split():
        part = part.strip()
        if part:
            seeds.append(int(part))
    if not seeds:
        raise ValueError("No seeds provided")
    return seeds


def _discover_csvs(eval_dir: Path, arm: str, seeds: list[int]) -> list[Path]:
    paths: list[Path] = []
    for seed in seeds:
        p = eval_dir / f"{arm}_seed{seed}.csv"
        if not p.is_file():
            raise FileNotFoundError(f"Missing Jamendo matrix CSV: {p}")
        paths.append(p)
    return paths


def _summarize_arm(
    csv_paths: list[Path],
    *,
    query_text: str,
    top_k: int,
    tag_id: str,
) -> dict[str, Any]:
    prec: list[float] = []
    ndcg_d: list[float] = []
    row0 = _pick_row(_load_matrix_csv(csv_paths[0]), query_text, top_k)
    if row0 is None:
        raise ValueError(f"Query {query_text!r} not in {csv_paths[0]}")

    for p in csv_paths:
        row = _pick_row(_load_matrix_csv(p), query_text, top_k)
        if row is None:
            raise ValueError(f"Query {query_text!r} not in {p}")
        prec.append(float(row["precision_at_k"]))
        ndcg_d.append(float(row["ndcg_delta"]))

    pm, ps = _mean_stdev(prec)
    nm, ns = _mean_stdev(ndcg_d)
    if pm is None:
        raise ValueError(f"No precision values for {query_text}")
    return {
        "tag_id": tag_id,
        "query_text": query_text,
        "top_k": top_k,
        "n_positive": row0.get("n_positive", ""),
        "n_pool": row0.get("n_pool", ""),
        "precision_mean": pm,
        "precision_stdev": ps,
        "ndcg_delta_mean": nm if nm is not None else 0.0,
        "ndcg_delta_stdev": ns if ns is not None else 0.0,
        "n_seeds": len(prec),
    }


def build_summary(
    *,
    eval_dir: Path,
    arms: list[str],
    seeds: list[int],
    top_k: int,
    baseline_arm: str | None = None,
) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for arm in arms:
        paths = _discover_csvs(eval_dir, arm, seeds)
        rows: list[dict[str, Any]] = []
        for tag_id, qt in PRIMARY_TAGS:
            rows.append(_summarize_arm(paths, query_text=qt, top_k=top_k, tag_id=tag_id))
        by_arm[arm] = rows

    primary_path = eval_dir / "summary_primary.csv"
    _write_primary_csv(primary_path, by_arm, arms, baseline_arm=baseline_arm)

    report_path = eval_dir / "REPORT.md"
    _write_report_md(report_path, by_arm=by_arm, arms=arms, seeds=seeds, top_k=top_k)

    meta: dict[str, Any] = {
        "eval_dir": str(eval_dir),
        "arms": arms,
        "seeds": seeds,
        "top_k": top_k,
        "baseline_arm": baseline_arm,
        "note": ABLATION_NOTE,
        "by_arm": by_arm,
        "outputs": {
            "summary_primary": str(primary_path),
            "report_md": str(report_path),
        },
    }
    summary_json = eval_dir / "summary.json"
    summary_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def _write_primary_csv(
    path: Path,
    by_arm: dict[str, list[dict[str, Any]]],
    arms: list[str],
    *,
    baseline_arm: str | None,
) -> None:
    fieldnames = [
        "arm",
        "tag_id",
        "query_text",
        "top_k",
        "n_pool",
        "n_positive",
        "precision_mean",
        "precision_stdev",
        "ndcg_delta_mean",
        "ndcg_delta_stdev",
        "delta_precision_vs_baseline",
        "n_seeds",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    base_rows = by_arm.get(baseline_arm or "") if baseline_arm else None

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for arm in arms:
            for r in by_arm[arm]:
                dp = ""
                if base_rows and baseline_arm and arm != baseline_arm:
                    br = next((x for x in base_rows if x["tag_id"] == r["tag_id"]), None)
                    if br:
                        dp = f"{r['precision_mean'] - br['precision_mean']:.6f}"
                w.writerow(
                    {
                        "arm": arm,
                        "tag_id": r["tag_id"],
                        "query_text": r["query_text"],
                        "top_k": r["top_k"],
                        "n_pool": r["n_pool"],
                        "n_positive": r["n_positive"],
                        "precision_mean": f"{r['precision_mean']:.6f}",
                        "precision_stdev": f"{r['precision_stdev']:.6f}",
                        "ndcg_delta_mean": f"{r['ndcg_delta_mean']:.6f}",
                        "ndcg_delta_stdev": f"{r['ndcg_delta_stdev']:.6f}",
                        "delta_precision_vs_baseline": dp,
                        "n_seeds": r["n_seeds"],
                    }
                )


def _write_report_md(
    path: Path,
    *,
    by_arm: dict[str, list[dict[str, Any]]],
    arms: list[str],
    seeds: list[int],
    top_k: int,
) -> None:
    lines = [
        "# Jamendo public (OOD) retrieval report",
        "",
        ABLATION_NOTE,
        "",
        f"- Eval dir: `{path.parent}`",
        f"- Arms: {', '.join(arms)}",
        f"- Seeds: {', '.join(str(s) for s in seeds)}",
        f"- Top-K: {top_k}",
        "",
        "| Tag | " + " | ".join(arms) + " |",
        "|-----|" + "|".join(["--------"] * len(arms)) + "|",
    ]
    for tag_id, _qt in PRIMARY_TAGS:
        cells = [tag_id]
        for arm in arms:
            r = next((x for x in by_arm[arm] if x["tag_id"] == tag_id), None)
            cells.append(f"{r['precision_mean']:.3f}" if r else "n/a")
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "### nDCG Δ (vs random)",
            "",
            "| Tag | " + " | ".join(arms) + " |",
            "|-----|" + "|".join(["--------"] * len(arms)) + "|",
        ]
    )
    for tag_id, _qt in PRIMARY_TAGS:
        cells = [tag_id]
        for arm in arms:
            r = next((x for x in by_arm[arm] if x["tag_id"] == tag_id), None)
            cells.append(f"{r['ndcg_delta_mean']:.3f}" if r else "n/a")
        lines.append("| " + " | ".join(cells) + " |")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Jamendo public OOD report.")
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=settings.DATA_DIR / "eval" / "jamendo_public",
    )
    parser.add_argument("--arms", type=str, default="pretrained,thesis_ft_v1")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--baseline-arm",
        type=str,
        default="pretrained",
        help="Arm for delta_precision_vs_baseline column in summary CSV.",
    )
    args = parser.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    meta = build_summary(
        eval_dir=args.eval_dir.resolve(),
        arms=arms,
        seeds=_parse_seeds(args.seeds),
        top_k=args.top_k,
        baseline_arm=args.baseline_arm.strip() or None,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
