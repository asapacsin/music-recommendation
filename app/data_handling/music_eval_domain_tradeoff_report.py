"""
2×2 domain tradeoff report: in-domain gold vs public OOD for anime-only vs mixed FT.

Rows: anime_only (thesis_tag_only), mixed (thesis_tag_mixed)
Cols: gold (human multihot), ood_macro (mean P@K over public datasets)
"""
from __future__ import annotations

import argparse
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
from app.data_handling.music_eval_public_report import (
    _discover_csvs,
    _summarize_arm,
    _tags_for_dataset,
)

INTERPRETATION_RULES: list[tuple[str, str, str, str]] = [
    ("A", "≥ −0.05", "≥ +0.15 (vocal)", "Forgetting-dominated — mixed recovers OOD"),
    ("B", "≥ +0.10", "≤ −0.05", "Specialization — anime-only wins in-domain"),
    ("C", "≤ −0.05", "≤ −0.05", "Mixed hurts both — check ratio / leakage"),
    ("D", "≥ −0.05", "≥ −0.05", "No clear tradeoff — arms similar"),
]


def _parse_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for part in raw.replace(",", " ").split():
        part = part.strip()
        if part:
            seeds.append(int(part))
    return seeds or [42]


def _parse_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.replace(",", " ").split() if x.strip()]


def _summarize_gold_arm(
    trade_dir: Path,
    prefix: str,
    seeds: list[int],
    top_k: int,
) -> dict[str, dict[str, Any]]:
    """prefix: anime_only_gold or mixed_gold"""
    out: dict[str, dict[str, Any]] = {}
    for tag_id, query_text in PRIMARY_TAGS:
        prec: list[float] = []
        ndcg: list[float] = []
        for seed in seeds:
            p = trade_dir / f"{prefix}_seed{seed}.csv"
            if not p.is_file():
                raise FileNotFoundError(f"Missing gold CSV: {p}")
            row = _pick_row(_load_matrix_csv(p), query_text, top_k)
            if row is None:
                raise ValueError(f"Query {query_text!r} missing in {p}")
            prec.append(float(row["precision_at_k"]))
            ndcg.append(float(row["ndcg_delta"]))
        pm, ps = _mean_stdev(prec)
        nm, ns = _mean_stdev(ndcg)
        out[tag_id] = {
            "tag_id": tag_id,
            "query_text": query_text,
            "precision_mean": pm,
            "precision_stdev": ps,
            "ndcg_delta_mean": nm,
            "ndcg_delta_stdev": ns,
            "n_seeds": len(prec),
        }
    return out


def _summarize_public_arm(
    eval_root: Path,
    arm: str,
    datasets: list[str],
    seeds: list[int],
    top_k: int,
) -> dict[str, dict[str, Any]]:
    """Per tag: mean P@K over datasets (macro OOD)."""
    per_ds: dict[str, dict[str, dict[str, Any]]] = {}
    for ds in datasets:
        eval_dir = eval_root / f"{ds}_public"
        paths = _discover_csvs(eval_dir, arm, seeds)
        tags = _tags_for_dataset(ds)
        per_ds[ds] = {}
        for tag_id, qt in tags:
            per_ds[ds][tag_id] = _summarize_arm(
                paths, query_text=qt, top_k=top_k, tag_id=tag_id
            )

    out: dict[str, dict[str, Any]] = {}
    all_tag_ids = {t[0] for t in PRIMARY_TAGS}
    for tag_id in all_tag_ids:
        prec_vals: list[float] = []
        ndcg_vals: list[float] = []
        by_dataset: dict[str, float] = {}
        for ds in datasets:
            if tag_id not in per_ds.get(ds, {}):
                continue
            r = per_ds[ds][tag_id]
            prec_vals.append(float(r["precision_mean"]))
            ndcg_vals.append(float(r["ndcg_delta_mean"]))
            by_dataset[ds] = float(r["precision_mean"])
        if not prec_vals:
            continue
        pm, ps = _mean_stdev(prec_vals)
        nm, ns = _mean_stdev(ndcg_vals)
        out[tag_id] = {
            "tag_id": tag_id,
            "precision_mean": pm,
            "precision_stdev": ps,
            "ndcg_delta_mean": nm,
            "ndcg_delta_stdev": ns,
            "by_dataset": by_dataset,
            "n_datasets": len(prec_vals),
        }
    return out


def _classify_tradeoff(
    gold_delta: float,
    ood_delta: float,
    *,
    tag_id: str,
) -> str:
    vocal_boost = 0.15 if tag_id == "inst_vocal" else 0.10
    if gold_delta >= -0.05 and ood_delta >= vocal_boost:
        return "Forgetting-dominated (mixed recovers OOD)"
    if gold_delta >= 0.10 and ood_delta <= -0.05:
        return "Specialization (anime-only wins in-domain)"
    if gold_delta <= -0.05 and ood_delta <= -0.05:
        return "Mixed hurts both — check ratio / leakage"
    return "No clear tradeoff — arms similar"


def build_domain_tradeoff_report(
    *,
    trade_dir: Path,
    eval_root: Path,
    datasets: list[str],
    seeds: list[int],
    top_k: int,
    anime_arm: str,
    mixed_arm: str,
) -> dict[str, Any]:
    gold_anime = _summarize_gold_arm(trade_dir, "anime_only_gold", seeds, top_k)
    gold_mixed = _summarize_gold_arm(trade_dir, "mixed_gold", seeds, top_k)
    ood_anime = _summarize_public_arm(eval_root, anime_arm, datasets, seeds, top_k)
    ood_mixed = _summarize_public_arm(eval_root, mixed_arm, datasets, seeds, top_k)
    ood_pretrained = _summarize_public_arm(
        eval_root, "pretrained", datasets, seeds, top_k
    )

    rows: list[dict[str, Any]] = []
    for tag_id, _qt in PRIMARY_TAGS:
        ga = gold_anime[tag_id]["precision_mean"]
        gm = gold_mixed[tag_id]["precision_mean"]
        oa = ood_anime.get(tag_id, {}).get("precision_mean")
        om = ood_mixed.get(tag_id, {}).get("precision_mean")
        op = ood_pretrained.get(tag_id, {}).get("precision_mean")
        gold_delta = (gm - ga) if gm is not None and ga is not None else None
        ood_delta = (om - oa) if om is not None and oa is not None else None
        label = (
            _classify_tradeoff(gold_delta, ood_delta, tag_id=tag_id)
            if gold_delta is not None and ood_delta is not None
            else "n/a"
        )
        rows.append(
            {
                "tag_id": tag_id,
                "anime_only_gold": ga,
                "mixed_gold": gm,
                "gold_delta_mixed_minus_anime": gold_delta,
                "anime_only_ood_macro": oa,
                "mixed_ood_macro": om,
                "ood_delta_mixed_minus_anime": ood_delta,
                "pretrained_ood_macro": op,
                "interpretation": label,
                "ood_by_dataset_anime_only": ood_anime.get(tag_id, {}).get(
                    "by_dataset", {}
                ),
                "ood_by_dataset_mixed": ood_mixed.get(tag_id, {}).get("by_dataset", {}),
            }
        )

    lines = [
        "# Domain tradeoff report (2×2: training regime × eval domain)",
        "",
        "Tests **forgetting vs specialization** by comparing anime-only FT (`thesis_tag_only`) "
        f"vs mixed FT (`{mixed_arm}`: anime + MTAT + OpenMIC train, Jamendo OOD-only).",
        "",
        f"- Trade dir: `{trade_dir}`",
        f"- Public eval root: `{eval_root}`",
        f"- Datasets (OOD): {', '.join(datasets)}",
        f"- Seeds: {', '.join(str(s) for s in seeds)}",
        f"- Top-K: {top_k}",
        "",
        "## 2×2 P@K (mean over seeds)",
        "",
        "| Tag | Anime-only / Gold | Mixed / Gold | Anime-only / OOD | Mixed / OOD | "
        "Δ Gold | Δ OOD | Interpretation |",
        "|-----|-------------------|--------------|------------------|-------------|"
        "--------|-------|----------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['tag_id']} "
            f"| {r['anime_only_gold']:.3f} "
            f"| {r['mixed_gold']:.3f} "
            f"| {r['anime_only_ood_macro']:.3f} "
            f"| {r['mixed_ood_macro']:.3f} "
            f"| {r['gold_delta_mixed_minus_anime']:+.3f} "
            f"| {r['ood_delta_mixed_minus_anime']:+.3f} "
            f"| {r['interpretation']} |"
        )

    lines.extend(
        [
            "",
            "## Pretrained reference (OOD macro)",
            "",
            "| Tag | Pretrained OOD |",
            "|-----|----------------|",
        ]
    )
    for r in rows:
        op = r.get("pretrained_ood_macro")
        lines.append(f"| {r['tag_id']} | {op:.3f} |" if op is not None else f"| {r['tag_id']} | n/a |")

    lines.extend(["", "## OOD per dataset (mixed arm)", ""])
    for ds in datasets:
        lines.append(f"### {ds}")
        lines.append("")
        lines.append("| Tag | anime_only | mixed |")
        lines.append("|-----|--------------|-------|")
        for r in rows:
            ao = r["ood_by_dataset_anime_only"].get(ds)
            mx = r["ood_by_dataset_mixed"].get(ds)
            ao_s = f"{ao:.3f}" if ao is not None else "n/a"
            mx_s = f"{mx:.3f}" if mx is not None else "n/a"
            lines.append(f"| {r['tag_id']} | {ao_s} | {mx_s} |")
        lines.append("")

    lines.extend(
        [
            "## Interpretation key",
            "",
            "| Pattern | Gold Δ (mixed − anime) | OOD Δ (mixed − anime) | Label |",
            "|---------|------------------------|------------------------|-------|",
        ]
    )
    for code, g, o, label in INTERPRETATION_RULES:
        lines.append(f"| {code} | {g} | {o} | {label} |")

    report_path = trade_dir / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    meta: dict[str, Any] = {
        "trade_dir": str(trade_dir),
        "eval_root": str(eval_root),
        "datasets": datasets,
        "seeds": seeds,
        "top_k": top_k,
        "anime_arm": anime_arm,
        "mixed_arm": mixed_arm,
        "rows": rows,
        "report_md": str(report_path),
    }
    summary_path = trade_dir / "summary.json"
    summary_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description="2×2 domain tradeoff report.")
    parser.add_argument(
        "--trade-dir",
        type=Path,
        default=settings.DATA_DIR / "eval" / "domain_tradeoff",
    )
    parser.add_argument("--eval-root", type=Path, default=settings.DATA_DIR / "eval")
    parser.add_argument("--datasets", type=str, default="jamendo,mtat,openmic")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--anime-arm", type=str, default="thesis_tag_only")
    parser.add_argument("--mixed-arm", type=str, default="thesis_tag_mixed")
    args = parser.parse_args()

    meta = build_domain_tradeoff_report(
        trade_dir=args.trade_dir.resolve(),
        eval_root=args.eval_root.resolve(),
        datasets=_parse_list(args.datasets),
        seeds=_parse_seeds(args.seeds),
        top_k=args.top_k,
        anime_arm=args.anime_arm.strip(),
        mixed_arm=args.mixed_arm.strip(),
    )
    print(json.dumps({"report": meta["report_md"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
