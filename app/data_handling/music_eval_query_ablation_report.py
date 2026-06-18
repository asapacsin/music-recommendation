"""
Query-set ablation report: expanding which multihot tags (and tempo) are used as CLAP queries.

Unlike ``music_eval_ablation_report`` (pretrained vs fine-tuned checkpoint comparison),
this module compares **query coverage tiers** and **per-tag** retrieval metrics from
existing ``music_eval_retrieval_vs_random`` matrix CSVs.

Writes under ``data/eval/ablation/`` by default:
  - ``query_ablation_per_tag.csv`` — one row per tag/query × model
  - ``query_ablation_tiers.csv`` — macro means over fixed query sets
  - ``query_ablation_cumulative.csv`` — macro mean as queries are added in order
  - ``query_ablation_report.md`` — thesis-ready summary
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

from app.data_handling.music_eval_prepare_gold_multihot_csv import MULTIHOT_COLUMNS
from app.data_handling.music_eval_retrieval_vs_random import _TEMPO_RETRIEVAL_QUERY_TEXTS

PRIMARY_TAG_IDS: list[str] = ["inst_piano", "inst_vocal", "mood_relaxing"]

TEMPO_QUERY_SPECS: list[tuple[str, str]] = [
    ("tempo_slow", _TEMPO_RETRIEVAL_QUERY_TEXTS[0]),
    ("tempo_mid", _TEMPO_RETRIEVAL_QUERY_TEXTS[1]),
    ("tempo_fast", _TEMPO_RETRIEVAL_QUERY_TEXTS[2]),
]


def _strip_music_suffix(text: str) -> str:
    t = text.strip()
    low = t.lower()
    suffix = " music"
    if low.endswith(suffix):
        return t[: -len(suffix)].strip()
    return t


def _load_matrix_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pick_row(rows: list[dict[str, str]], query_text: str, top_k: int) -> dict[str, str] | None:
    for r in rows:
        if r.get("query_text") == query_text and int(r.get("top_k", 0)) == top_k:
            return r
    return None


def _load_style_query_map(path: Path) -> dict[str, str]:
    """query_id -> stripped query_text (matches matrix CSV)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        qid = str(item.get("query_id", "")).strip()
        qt = str(item.get("query_text", "")).strip()
        if qid and qt:
            out[qid] = _strip_music_suffix(qt)
    return out


def _ordered_queries(style_map: Path) -> list[tuple[str, str, str]]:
    """(query_id, query_text, group) in cumulative ablation order."""
    smap = _load_style_query_map(style_map)
    ordered: list[tuple[str, str, str]] = []
    for qid in PRIMARY_TAG_IDS:
        if qid in smap:
            ordered.append((qid, smap[qid], "style_primary"))
    for qid in MULTIHOT_COLUMNS:
        if qid in PRIMARY_TAG_IDS or qid not in smap:
            continue
        ordered.append((qid, smap[qid], "style_extra"))
    for qid, qt in TEMPO_QUERY_SPECS:
        ordered.append((qid, qt, "tempo"))
    return ordered


def _discover_ft_csvs(ablation_dir: Path) -> list[Path]:
    return sorted(ablation_dir.glob("ft_seed*.csv"))


def _ft_mean_metric(
    ft_paths: list[Path],
    query_text: str,
    top_k: int,
    field: str,
) -> tuple[float | None, float | None]:
    vals: list[float] = []
    for p in ft_paths:
        row = _pick_row(_load_matrix_csv(p), query_text, top_k)
        if row is not None:
            vals.append(float(row[field]))
    if not vals:
        return None, None
    m = statistics.mean(vals)
    sd = 0.0 if len(vals) < 2 else statistics.pstdev(vals)
    return m, sd


def _macro_means(
    rows_by_model: dict[str, list[dict[str, Any]]],
    query_texts: list[str],
) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for model, rows in rows_by_model.items():
        prec: list[float] = []
        ndcg: list[float] = []
        for qt in query_texts:
            match = [r for r in rows if r["query_text"] == qt]
            if match:
                prec.append(float(match[0]["precision_at_k"]))
                ndcg.append(float(match[0]["ndcg_delta"]))
        if not prec:
            continue
        out[model] = {
            "n_queries": len(prec),
            "macro_precision_at_k": statistics.mean(prec),
            "macro_ndcg_delta": statistics.mean(ndcg),
        }
    return out


def _per_tag_rows(
    *,
    ordered: list[tuple[str, str, str]],
    top_k: int,
    pretrained_rows: list[dict[str, str]],
    ft_paths: list[Path],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for qid, qt, group in ordered:
        pre = _pick_row(pretrained_rows, qt, top_k)
        if pre is None:
            continue
        pm, psd = _ft_mean_metric(ft_paths, qt, top_k, "precision_at_k")
        nm, nsd = _ft_mean_metric(ft_paths, qt, top_k, "ndcg_delta")
        for model, p, n, p_sd in (
            ("pretrained", float(pre["precision_at_k"]), float(pre["ndcg_delta"]), None),
            ("fine_tuned", pm, nm, psd),
        ):
            if model == "fine_tuned" and p is None:
                continue
            out.append(
                {
                    "query_id": qid,
                    "query_text": qt,
                    "query_group": group,
                    "model": model,
                    "top_k": top_k,
                    "n_positive": int(pre.get("n_positive", 0)),
                    "prevalence": float(pre.get("prevalence", 0)),
                    "precision_at_k": f"{p:.6f}" if p is not None else "",
                    "ndcg_delta": f"{n:.6f}" if n is not None else "",
                    "precision_stdev_across_seeds": "" if p_sd is None else f"{p_sd:.6f}",
                    "in_primary_3": qid in PRIMARY_TAG_IDS,
                    "in_finetune_scope": qid in PRIMARY_TAG_IDS,
                }
            )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _build_markdown(
    *,
    top_k: int,
    tier_rows: list[dict[str, Any]],
    cumulative_rows: list[dict[str, Any]],
    per_tag_rows: list[dict[str, Any]],
    pretrained_csv: Path,
    ft_paths: list[Path],
) -> str:
    lines = [
        "# Query-set ablation report",
        "",
        "Expanding which **CLAP text queries** are evaluated against the metadata FAISS index "
        "(gold-labeled pool). This is **not** a checkpoint comparison table; see "
        "`summary_primary.csv` for pretrained vs fine-tuned on the same queries.",
        "",
        f"- **K:** {top_k}",
        f"- **Pretrained matrix:** `{pretrained_csv}`",
        f"- **Fine-tuned matrices:** {len(ft_paths)} seeds under `ft_seed*.csv`",
        "",
        "## Query-set tiers (macro mean over queries in set)",
        "",
        "| Tier | Queries | Model | Macro P@K | Macro ΔnDCG |",
        "|------|---------|-------|-------------|---------------|",
    ]
    for r in tier_rows:
        lines.append(
            f"| {r['tier_id']} | {r['n_queries']} | {r['model']} | "
            f"{r['macro_precision_at_k']} | {r['macro_ndcg_delta']} |"
        )
    lines.extend(
        [
            "",
            "## Cumulative add-query ablation (fixed order)",
            "",
            "Order: fine-tune **primary 3** → remaining **5 style** tags → **3 tempo** phrases.",
            "",
            "| # added | Last query_id | Model | Macro P@K | Macro ΔnDCG |",
            "|---------|---------------|-------|-------------|---------------|",
        ]
    )
    for r in cumulative_rows:
        lines.append(
            f"| {r['n_queries_in_set']} | {r['last_query_id']} | {r['model']} | "
            f"{r['macro_precision_at_k']} | {r['macro_ndcg_delta']} |"
        )
    lines.extend(
        [
            "",
            "## Per-tag results (pretrained vs fine-tuned)",
            "",
            "| query_id | group | n+ | Pretrained P@K | FT P@K | ΔP | Pretrained ΔnDCG | FT ΔnDCG |",
            "|----------|-------|-----|----------------|--------|-----|------------------|----------|",
        ]
    )
    by_id: dict[str, dict[str, dict[str, str]]] = {}
    for r in per_tag_rows:
        qid = r["query_id"]
        by_id.setdefault(qid, {})[r["model"]] = r
    for qid, _, group in [x for x in _ordered_queries(settings.DATA_DIR / "eval" / "style_queries.json")]:
        models = by_id.get(qid, {})
        pre = models.get("pretrained", {})
        ft = models.get("fine_tuned", {})
        if not pre:
            continue
        p0 = float(pre["precision_at_k"])
        n0 = float(pre["ndcg_delta"])
        p1 = float(ft["precision_at_k"]) if ft else None
        n1 = float(ft["ndcg_delta"]) if ft else None
        dp = "" if p1 is None else f"{p1 - p0:+.3f}"
        lines.append(
            f"| {qid} | {group} | {pre.get('n_positive', '')} | {p0:.3f} | "
            f"{'' if p1 is None else f'{p1:.3f}'} | {dp} | {n0:.3f} | "
            f"{'' if n1 is None else f'{n1:.3f}'} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Query-set ablation report from retrieval-vs-random matrix CSVs."
    )
    parser.add_argument(
        "--ablation-dir",
        type=Path,
        default=settings.DATA_DIR / "eval" / "ablation",
    )
    parser.add_argument(
        "--pretrained-csv",
        type=Path,
        default=None,
        help="Default: <ablation-dir>/pretrained.csv",
    )
    parser.add_argument(
        "--style-queries",
        type=Path,
        default=settings.DATA_DIR / "eval" / "style_queries.json",
    )
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    ablation_dir = args.ablation_dir
    pretrained_path = args.pretrained_csv or (ablation_dir / "pretrained.csv")
    if not pretrained_path.is_file():
        raise FileNotFoundError(
            f"Missing {pretrained_path}. Run music_eval_retrieval_vs_random or sbatch_clap_ablation.sh."
        )

    ft_paths = _discover_ft_csvs(ablation_dir)
    ordered = _ordered_queries(args.style_queries)
    pretrained_rows = _load_matrix_csv(pretrained_path)
    top_k = args.top_k

    per_tag = _per_tag_rows(
        ordered=ordered,
        top_k=top_k,
        pretrained_rows=pretrained_rows,
        ft_paths=ft_paths,
    )
    per_tag_fields = [
        "query_id",
        "query_text",
        "query_group",
        "model",
        "top_k",
        "n_positive",
        "prevalence",
        "precision_at_k",
        "ndcg_delta",
        "precision_stdev_across_seeds",
        "in_primary_3",
        "in_finetune_scope",
    ]
    per_tag_path = ablation_dir / "query_ablation_per_tag.csv"
    _write_csv(per_tag_path, per_tag, per_tag_fields)

    def model_rows(model: str) -> list[dict[str, Any]]:
        return [r for r in per_tag if r["model"] == model]

    style_qts = [qt for _, qt, g in ordered if g.startswith("style")]
    primary_qts = [qt for qid, qt, _ in ordered if qid in PRIMARY_TAG_IDS]
    all_qts = [qt for _, qt, _ in ordered]

    tier_specs = [
        ("primary_3_finetune_scope", primary_qts, "3 tags used for CLAP fine-tune"),
        ("all_style_8_multihot", style_qts, "All gold multihot style/instrument columns"),
        ("full_style_plus_tempo_11", all_qts, "8 style + 3 tempo retrieval phrases"),
    ]

    tier_rows: list[dict[str, Any]] = []
    for tier_id, qts, description in tier_specs:
        for model in ("pretrained", "fine_tuned"):
            subset = [r for r in model_rows(model) if r["query_text"] in qts]
            if not subset:
                continue
            tier_rows.append(
                {
                    "tier_id": tier_id,
                    "description": description,
                    "model": model,
                    "top_k": top_k,
                    "n_queries": len(subset),
                    "macro_precision_at_k": f"{statistics.mean(float(r['precision_at_k']) for r in subset):.6f}",
                    "macro_ndcg_delta": f"{statistics.mean(float(r['ndcg_delta']) for r in subset):.6f}",
                }
            )

    tier_path = ablation_dir / "query_ablation_tiers.csv"
    _write_csv(
        tier_path,
        tier_rows,
        [
            "tier_id",
            "description",
            "model",
            "top_k",
            "n_queries",
            "macro_precision_at_k",
            "macro_ndcg_delta",
        ],
    )

    cumulative_rows: list[dict[str, Any]] = []
    for n in range(1, len(ordered) + 1):
        qts_n = [qt for _, qt, _ in ordered[:n]]
        last_qid = ordered[n - 1][0]
        for model in ("pretrained", "fine_tuned"):
            subset = [r for r in model_rows(model) if r["query_text"] in qts_n]
            if not subset:
                continue
            cumulative_rows.append(
                {
                    "n_queries_in_set": n,
                    "last_query_id": last_qid,
                    "query_ids_in_set": ";".join(qid for qid, _, _ in ordered[:n]),
                    "model": model,
                    "top_k": top_k,
                    "macro_precision_at_k": f"{statistics.mean(float(r['precision_at_k']) for r in subset):.6f}",
                    "macro_ndcg_delta": f"{statistics.mean(float(r['ndcg_delta']) for r in subset):.6f}",
                }
            )

    cum_path = ablation_dir / "query_ablation_cumulative.csv"
    _write_csv(
        cum_path,
        cumulative_rows,
        [
            "n_queries_in_set",
            "last_query_id",
            "query_ids_in_set",
            "model",
            "top_k",
            "macro_precision_at_k",
            "macro_ndcg_delta",
        ],
    )

    md_path = ablation_dir / "query_ablation_report.md"
    md_path.write_text(
        _build_markdown(
            top_k=top_k,
            tier_rows=tier_rows,
            cumulative_rows=cumulative_rows,
            per_tag_rows=per_tag,
            pretrained_csv=pretrained_path,
            ft_paths=ft_paths,
        ),
        encoding="utf-8",
    )

    meta = {
        "report_type": "query_set_ablation",
        "top_k": top_k,
        "pretrained_csv": str(pretrained_path),
        "ft_csvs": [str(p) for p in ft_paths],
        "n_ft_seeds": len(ft_paths),
        "query_order": [{"query_id": q, "query_text": t, "group": g} for q, t, g in ordered],
        "outputs": {
            "per_tag": str(per_tag_path),
            "tiers": str(tier_path),
            "cumulative": str(cum_path),
            "markdown": str(md_path),
        },
    }
    meta_path = ablation_dir / "query_ablation_summary.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
