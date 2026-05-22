"""
Build ablation summary CSVs from retrieval-vs-random matrix exports.

Expects:
  - One pretrained matrix CSV (no fine-tune checkpoint used during eval).
  - One or more fine-tuned matrix CSVs (per seed), same columns as
    ``music_eval_retrieval_vs_random``.

Writes under ``data/eval/ablation/`` by default:
  - ``summary_primary.csv`` — inst_piano / inst_vocal / mood_relaxing prompts
  - ``summary_all_queries.csv`` — every query row in the pretrained file
  - ``summary.json`` — metadata + row counts
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

# query_text after " music" strip (must match music_eval_retrieval_vs_random)
PRIMARY_TAGS: list[tuple[str, str]] = [
    ("inst_piano", "piano"),
    ("inst_vocal", "vocal"),
    ("mood_relaxing", "relaxing"),
]

PRIMARY_QUERY_TEXTS = frozenset(q for _, q in PRIMARY_TAGS)


def _load_matrix_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pick_row(rows: list[dict[str, str]], query_text: str, top_k: int) -> dict[str, str] | None:
    for r in rows:
        if r.get("query_text") == query_text and int(r.get("top_k", 0)) == top_k:
            return r
    return None


def _ft_metric_lists(
    ft_paths: list[Path],
    query_text: str,
    top_k: int,
    field: str,
) -> list[float]:
    out: list[float] = []
    for p in ft_paths:
        row = _pick_row(_load_matrix_csv(p), query_text, top_k)
        if row is not None:
            out.append(float(row[field]))
    return out


def _mean_stdev(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    m = statistics.mean(values)
    if len(values) < 2:
        return m, 0.0
    return m, statistics.pstdev(values)


def _write_summary_csv(
    path: Path,
    *,
    query_texts: list[str],
    top_k: int,
    pretrained_rows: list[dict[str, str]],
    ft_paths: list[Path],
) -> int:
    fieldnames = [
        "tag_id",
        "query_text",
        "top_k",
        "n_positive",
        "pretrained_precision_at_k",
        "pretrained_ndcg_delta",
        "ft_precision_mean",
        "ft_precision_stdev",
        "ft_ndcg_delta_mean",
        "ft_ndcg_delta_stdev",
        "delta_precision_vs_pretrained",
        "delta_ndcg_vs_pretrained",
        "n_ft_seeds",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        tag_by_query = {q: tid for tid, q in PRIMARY_TAGS}
        for qt in query_texts:
            pre = _pick_row(pretrained_rows, qt, top_k)
            if pre is None:
                continue
            ps = _ft_metric_lists(ft_paths, qt, top_k, "precision_at_k")
            ns = _ft_metric_lists(ft_paths, qt, top_k, "ndcg_delta")
            pm, psd = _mean_stdev(ps)
            nm, nsd = _mean_stdev(ns)
            p0 = float(pre["precision_at_k"])
            n0 = float(pre["ndcg_delta"])
            w.writerow(
                {
                    "tag_id": tag_by_query.get(qt, ""),
                    "query_text": qt,
                    "top_k": top_k,
                    "n_positive": pre.get("n_positive", ""),
                    "pretrained_precision_at_k": f"{p0:.6f}",
                    "pretrained_ndcg_delta": f"{n0:.6f}",
                    "ft_precision_mean": "" if pm is None else f"{pm:.6f}",
                    "ft_precision_stdev": "" if psd is None else f"{psd:.6f}",
                    "ft_ndcg_delta_mean": "" if nm is None else f"{nm:.6f}",
                    "ft_ndcg_delta_stdev": "" if nsd is None else f"{nsd:.6f}",
                    "delta_precision_vs_pretrained": ""
                    if pm is None
                    else f"{pm - p0:.6f}",
                    "delta_ndcg_vs_pretrained": "" if nm is None else f"{nm - n0:.6f}",
                    "n_ft_seeds": len(ps),
                }
            )
            n_written += 1
    return n_written


def _discover_ft_csvs(
    ablation_dir: Path,
    *,
    run_id: str,
    legacy_dir: Path,
) -> list[Path]:
    patterns = [
        ablation_dir.glob("ft_seed*.csv"),
        legacy_dir.glob("retrieval_matrix_seed*.csv"),
        ablation_dir.glob("retrieval_matrix_seed*.csv"),
    ]
    seen: dict[str, Path] = {}
    for gen in patterns:
        for p in sorted(gen):
            seen[p.name] = p
    if seen:
        return list(seen.values())

    out: list[Path] = []
    for seed_dir in sorted(settings.FINETUNE_MODEL_DIR.glob(f"{run_id}/seed_*")):
        name = seed_dir.name.replace("seed_", "")
        for cand in (
            ablation_dir / f"ft_seed{name}.csv",
            legacy_dir / f"retrieval_matrix_seed{name}.csv",
        ):
            if cand.is_file():
                out.append(cand)
                break
    log_run = settings.finetune_log_run_dir(run_id)
    if log_run.is_dir():
        for seed_dir in sorted(log_run.glob("seed_*")):
            name = seed_dir.name.replace("seed_", "")
            if any(p.name.endswith(f"seed{name}.csv") for p in out):
                continue
            for cand in (
                ablation_dir / f"ft_seed{name}.csv",
                legacy_dir / f"retrieval_matrix_seed{name}.csv",
            ):
                if cand.is_file():
                    out.append(cand)
                    break
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize pretrained vs fine-tuned retrieval matrices.")
    parser.add_argument(
        "--ablation-dir",
        type=Path,
        default=settings.DATA_DIR / "eval" / "ablation",
        help="Directory with pretrained.csv and ft_seed*.csv (or legacy retrieval_matrix_seed*.csv).",
    )
    parser.add_argument(
        "--pretrained-csv",
        type=Path,
        default=None,
        help="Pretrained matrix CSV (default: <ablation-dir>/pretrained.csv).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="K value for summary tables (must exist in matrix CSVs).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="thesis_ft_v1",
        help="Fine-tune run id (used only to discover legacy per-seed CSV names).",
    )
    parser.add_argument(
        "--ft-csv",
        type=Path,
        action="append",
        default=None,
        help="Explicit fine-tuned matrix CSV (repeatable). Overrides auto-discovery.",
    )
    args = parser.parse_args()

    ablation_dir = args.ablation_dir
    eval_dir = settings.DATA_DIR / "eval"
    pretrained_path = args.pretrained_csv or (ablation_dir / "pretrained.csv")
    if not pretrained_path.is_file():
        raise FileNotFoundError(
            f"Pretrained matrix not found: {pretrained_path}. "
            "Run scripts/sbatch_clap_ablation.sh or music_eval_retrieval_vs_random first."
        )

    if args.ft_csv:
        ft_paths = list(args.ft_csv)
    else:
        ft_paths = _discover_ft_csvs(ablation_dir, run_id=args.run_id, legacy_dir=eval_dir)

    if not ft_paths:
        print(
            "warning: no fine-tuned matrix CSVs found; summary will have pretrained columns only",
            file=sys.stderr,
        )

    pretrained_rows = _load_matrix_csv(pretrained_path)
    all_query_texts: list[str] = []
    seen_q: set[str] = set()
    for r in pretrained_rows:
        qt = r.get("query_text", "")
        if qt and qt not in seen_q:
            seen_q.add(qt)
            all_query_texts.append(qt)

    primary_path = ablation_dir / "summary_primary.csv"
    all_path = ablation_dir / "summary_all_queries.csv"
    n_primary = _write_summary_csv(
        primary_path,
        query_texts=[q for _, q in PRIMARY_TAGS],
        top_k=args.top_k,
        pretrained_rows=pretrained_rows,
        ft_paths=ft_paths,
    )
    n_all = _write_summary_csv(
        all_path,
        query_texts=all_query_texts,
        top_k=args.top_k,
        pretrained_rows=pretrained_rows,
        ft_paths=ft_paths,
    )

    meta: dict[str, Any] = {
        "pretrained_csv": str(pretrained_path),
        "ft_csvs": [str(p) for p in ft_paths],
        "top_k": args.top_k,
        "primary_tags": [{"tag_id": t, "query_text": q} for t, q in PRIMARY_TAGS],
        "n_primary_rows": n_primary,
        "n_all_query_rows": n_all,
        "outputs": {
            "summary_primary": str(primary_path),
            "summary_all_queries": str(all_path),
        },
    }
    summary_json = ablation_dir / "summary.json"
    summary_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
