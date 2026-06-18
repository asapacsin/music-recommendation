"""
Combined report for public OOD retrieval (Jamendo, MTAT, OpenMIC).

Expects per-dataset dirs ``data/eval/{dataset}_public/{arm}_seed{N}.csv``.
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
    _load_matrix_csv,
    _mean_stdev,
    _pick_row,
)
from app.data_handling.music_eval_public_retrieval import PRIMARY_BY_DATASET

DATASET_NOTES: dict[str, str] = {
    "jamendo": "MTG-Jamendo five-tag (split-0 test cap). Queries: piano, vocal, relaxing.",
    "mtat": "MagnaTagATune cap. Vocal = OR(vocals, male vocal, female voice, singer, voice). "
    "Relaxing = OR(calm, mellow).",
    "openmic": "OpenMIC-2018 cap. Piano + voice only (no mood labels in dataset).",
}


def _parse_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for part in raw.replace(",", " ").split():
        part = part.strip()
        if part:
            seeds.append(int(part))
    return seeds or [42]


def _parse_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.replace(",", " ").split() if x.strip()]


def _discover_csvs(eval_dir: Path, arm: str, seeds: list[int]) -> list[Path]:
    paths: list[Path] = []
    for seed in seeds:
        p = eval_dir / f"{arm}_seed{seed}.csv"
        if not p.is_file():
            raise FileNotFoundError(f"Missing matrix CSV: {p}")
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


def _tags_for_dataset(dataset: str) -> list[tuple[str, str]]:
    """Return (tag_id, query_text) for report rows."""
    primary = PRIMARY_BY_DATASET.get(dataset, [])
    return [(t[0], t[2]) for t in primary]


def build_dataset_summary(
    *,
    eval_dir: Path,
    dataset: str,
    arms: list[str],
    seeds: list[int],
    top_k: int,
) -> dict[str, list[dict[str, Any]]]:
    by_arm: dict[str, list[dict[str, Any]]] = {}
    tags = _tags_for_dataset(dataset)
    for arm in arms:
        paths = _discover_csvs(eval_dir, arm, seeds)
        rows: list[dict[str, Any]] = []
        for tag_id, qt in tags:
            rows.append(
                _summarize_arm(paths, query_text=qt, top_k=top_k, tag_id=tag_id)
            )
        by_arm[arm] = rows
    return by_arm


def _write_report_section(
    lines: list[str],
    *,
    dataset: str,
    by_arm: dict[str, list[dict[str, Any]]],
    arms: list[str],
) -> None:
    lines.extend(
        [
            f"## {dataset}",
            "",
            DATASET_NOTES.get(dataset, ""),
            "",
            "| Tag | " + " | ".join(arms) + " |",
            "|-----|" + "|".join(["--------"] * len(arms)) + "|",
        ]
    )
    tags = _tags_for_dataset(dataset)
    for tag_id, _qt in tags:
        cells = [tag_id]
        for arm in arms:
            r = next((x for x in by_arm[arm] if x["tag_id"] == tag_id), None)
            cells.append(f"{r['precision_mean']:.3f}" if r else "n/a")
        lines.append("| " + " | ".join(cells) + " |")


def build_combined_report(
    *,
    eval_root: Path,
    datasets: list[str],
    arms: list[str],
    seeds: list[int],
    top_k: int,
) -> dict[str, Any]:
    all_by_dataset: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for ds in datasets:
        eval_dir = eval_root / f"{ds}_public"
        all_by_dataset[ds] = build_dataset_summary(
            eval_dir=eval_dir,
            dataset=ds,
            arms=arms,
            seeds=seeds,
            top_k=top_k,
        )

    lines = [
        "# Public OOD retrieval report",
        "",
        f"- Eval root: `{eval_root}`",
        f"- Datasets: {', '.join(datasets)}",
        f"- Arms: {', '.join(arms)}",
        f"- Seeds: {', '.join(str(s) for s in seeds)}",
        f"- Top-K: {top_k}",
        "",
    ]
    for ds in datasets:
        _write_report_section(lines, dataset=ds, by_arm=all_by_dataset[ds], arms=arms)
        lines.append("")

    report_path = eval_root / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    meta: dict[str, Any] = {
        "eval_root": str(eval_root),
        "datasets": datasets,
        "arms": arms,
        "seeds": seeds,
        "top_k": top_k,
        "by_dataset": all_by_dataset,
        "report_md": str(report_path),
    }
    summary_json = eval_root / "summary.json"
    summary_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Combined public OOD retrieval report.")
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=settings.DATA_DIR / "eval",
    )
    parser.add_argument("--datasets", type=str, default="jamendo,mtat,openmic")
    parser.add_argument("--arms", type=str, default="pretrained,thesis_ft_v1")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    meta = build_combined_report(
        eval_root=args.eval_root.resolve(),
        datasets=_parse_list(args.datasets),
        arms=_parse_list(args.arms),
        seeds=_parse_seeds(args.seeds),
        top_k=args.top_k,
    )
    print(json.dumps({"report": meta["report_md"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
