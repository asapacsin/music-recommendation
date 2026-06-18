"""
Report for tag-only vs tag→LLM training ablation (symmetric per-checkpoint FAISS).

Expects CSVs under ``data/eval/tag_llm_ablation/``:
  - ``tag_meta_seed{N}.csv`` / ``tag_llm_meta_seed{N}.csv``
  - ``tag_caption_seed{N}.csv`` / ``tag_llm_caption_seed{N}.csv`` (optional)
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
    "Full-corpus primary-tag training text (gold join + fallback for unlabeled songs); "
    "LLM arm expands tag strings to short captions. "
    "Eval: per-checkpoint FAISS rebuild, human-gold tag queries."
)

ARM_TAG = "tag"
ARM_TAG_LLM = "tag_llm"
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
    tag_paths: list[Path],
    tag_llm_paths: list[Path],
    query_text: str,
    top_k: int,
    tag_id: str,
) -> dict[str, Any]:
    tag_row = _pick_row(_load_matrix_csv(tag_paths[0]), query_text, top_k)
    if tag_row is None:
        raise ValueError(f"Query {query_text!r} not found in {tag_paths[0]}")

    tp = _metric_lists(tag_paths, query_text, top_k, "precision_at_k")
    tn = _metric_lists(tag_paths, query_text, top_k, "ndcg_delta")
    lp = _metric_lists(tag_llm_paths, query_text, top_k, "precision_at_k")
    ln = _metric_lists(tag_llm_paths, query_text, top_k, "ndcg_delta")
    tpm, tps = _mean_stdev(tp)
    tnm, tns = _mean_stdev(tn)
    lpm, lps = _mean_stdev(lp)
    lnm, lns = _mean_stdev(ln)
    if tpm is None or lpm is None or tnm is None or lnm is None:
        raise ValueError(f"Insufficient metrics for {query_text}")

    return {
        "tag_id": tag_id,
        "query_text": query_text,
        "top_k": top_k,
        "n_positive": tag_row.get("n_positive", ""),
        "tag_precision_mean": tpm,
        "tag_precision_stdev": tps,
        "tag_ndcg_delta_mean": tnm,
        "tag_ndcg_delta_stdev": tns,
        "tag_llm_precision_mean": lpm,
        "tag_llm_precision_stdev": lps,
        "tag_llm_ndcg_delta_mean": lnm,
        "tag_llm_ndcg_delta_stdev": lns,
        "delta_precision": lpm - tpm,
        "delta_ndcg": lnm - tnm,
        "n_seeds": len(tp),
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
        tag_paths = _discover_csvs(ablation_dir, ARM_TAG, kind, seeds)
        tag_llm_paths = _discover_csvs(ablation_dir, ARM_TAG_LLM, kind, seeds)
        rows: list[dict[str, Any]] = []
        for tag_id, qt in PRIMARY_TAGS:
            rows.append(
                _summarize_arm_pair(
                    tag_paths=tag_paths,
                    tag_llm_paths=tag_llm_paths,
                    query_text=qt,
                    top_k=top_k,
                    tag_id=tag_id,
                )
            )
        by_index[kind] = rows

    primary_path = ablation_dir / "summary_primary.csv"
    _write_summary_csv(primary_path, by_index.get("meta", []))

    by_index_path = ablation_dir / "summary_by_index.csv"
    _write_by_index_csv(by_index_path, by_index)

    report_path = ablation_dir / "REPORT.md"
    _write_report_md(
        report_path,
        by_index=by_index,
        top_k=top_k,
        seeds=seeds,
        ablation_dir=ablation_dir,
    )

    meta: dict[str, Any] = {
        "ablation_dir": str(ablation_dir),
        "seeds": seeds,
        "top_k": top_k,
        "note": ABLATION_NOTE,
        "index_kinds": kinds,
        "arms": {"tag": ARM_TAG, "tag_llm": ARM_TAG_LLM},
        "by_index": {
            kind: [
                {
                    k: v
                    for k, v in r.items()
                    if k not in ("tag_precision_stdev", "tag_llm_precision_stdev")
                }
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
        "tag_precision_mean",
        "tag_precision_stdev",
        "tag_ndcg_delta_mean",
        "tag_ndcg_delta_stdev",
        "tag_llm_precision_mean",
        "tag_llm_precision_stdev",
        "tag_llm_ndcg_delta_mean",
        "tag_llm_ndcg_delta_stdev",
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
        "tag_precision_mean",
        "tag_llm_precision_mean",
        "delta_precision",
        "tag_ndcg_delta_mean",
        "tag_llm_ndcg_delta_mean",
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
                        "tag_precision_mean": f"{r['tag_precision_mean']:.6f}",
                        "tag_llm_precision_mean": f"{r['tag_llm_precision_mean']:.6f}",
                        "delta_precision": f"{r['delta_precision']:.6f}",
                        "tag_ndcg_delta_mean": f"{r['tag_ndcg_delta_mean']:.6f}",
                        "tag_llm_ndcg_delta_mean": f"{r['tag_llm_ndcg_delta_mean']:.6f}",
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
        "# Tag-only vs tag→LLM training ablation",
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
                "| Tag | Tag-only P@K | Tag→LLM P@K | Δ precision | Tag-only nDCG Δ | Tag→LLM nDCG Δ | Δ nDCG |",
                "|-----|--------------|-------------|-------------|-----------------|----------------|--------|",
            ]
        )
        for r in rows:
            lines.append(
                f"| {r['tag_id']} | {r['tag_precision_mean']:.3f} | {r['tag_llm_precision_mean']:.3f} "
                f"| {r['delta_precision']:+.3f} | {r['tag_ndcg_delta_mean']:.3f} "
                f"| {r['tag_llm_ndcg_delta_mean']:.3f} | {r['delta_ndcg']:+.3f} |"
            )
        lines.extend(["", "### Interpretation", ""])
        for r in rows:
            dp = r["delta_precision"]
            if dp > 0.01:
                verdict = "Tag→LLM arm better"
            elif dp < -0.01:
                verdict = "Tag-only arm better"
            else:
                verdict = "Roughly tied"
            lines.append(f"- **{r['tag_id']}** ({kind}): {verdict} (Δ precision {dp:+.3f})")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Tag vs tag→LLM ablation report.")
    parser.add_argument(
        "--ablation-dir",
        type=Path,
        default=settings.DATA_DIR / "eval" / "tag_llm_ablation",
    )
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--index-kinds",
        type=str,
        default="meta,caption",
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
