"""
Summarize composite-query ablation CSVs: pretrained vs fine-tuned seeds.

Expects:
  composite/composite_pretrained.csv
  composite/composite_ft_seed<N>.csv
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


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pick(rows: list[dict[str, str]], n_tags: int, top_k: int) -> dict[str, str] | None:
    for r in rows:
        if int(r["n_tags_in_prompt"]) == n_tags and int(r["top_k"]) == top_k:
            return r
    return None


def _ft_paths(composite_dir: Path) -> list[Path]:
    return sorted(composite_dir.glob("composite_ft_seed*.csv"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Composite query ablation report (pretrained vs FT seeds).")
    parser.add_argument(
        "--composite-dir",
        type=Path,
        default=settings.DATA_DIR / "eval" / "ablation" / "composite",
    )
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    composite_dir = args.composite_dir
    pre_path = composite_dir / "composite_pretrained.csv"
    if not pre_path.is_file():
        raise FileNotFoundError(f"Missing {pre_path}. Run sbatch_composite_query_ablation.sh first.")

    top_k = args.top_k
    pre_rows = _load_csv(pre_path)
    ft_paths = _ft_paths(composite_dir)
    n_tags_list = sorted({int(r["n_tags_in_prompt"]) for r in pre_rows})

    summary_rows: list[dict[str, Any]] = []
    for n_tags in n_tags_list:
        pre = _pick(pre_rows, n_tags, top_k)
        if pre is None:
            continue
        ps = []
        ns = []
        for p in ft_paths:
            row = _pick(_load_csv(p), n_tags, top_k)
            if row:
                ps.append(float(row["precision_at_k"]))
                ns.append(float(row["ndcg_delta"]))
        pm, psd = (statistics.mean(ps), statistics.pstdev(ps) if len(ps) > 1 else 0.0) if ps else (None, None)
        nm, nsd = (statistics.mean(ns), statistics.pstdev(ns) if len(ns) > 1 else 0.0) if ns else (None, None)
        p0 = float(pre["precision_at_k"])
        n0 = float(pre["ndcg_delta"])
        summary_rows.append(
            {
                "n_tags_in_prompt": n_tags,
                "query_text": pre["query_text"],
                "query_ids": pre["query_ids"],
                "n_positive": pre["n_positive"],
                "prevalence": pre["prevalence"],
                "top_k": top_k,
                "pretrained_precision_at_k": f"{p0:.6f}",
                "pretrained_ndcg_delta": f"{n0:.6f}",
                "ft_precision_mean": "" if pm is None else f"{pm:.6f}",
                "ft_precision_stdev": "" if psd is None else f"{psd:.6f}",
                "ft_ndcg_delta_mean": "" if nm is None else f"{nm:.6f}",
                "ft_ndcg_delta_stdev": "" if nsd is None else f"{nsd:.6f}",
                "delta_precision_vs_pretrained": "" if pm is None else f"{pm - p0:.6f}",
                "delta_ndcg_vs_pretrained": "" if nm is None else f"{nm - n0:.6f}",
                "n_ft_seeds": len(ps),
            }
        )

    csv_out = composite_dir / "composite_query_summary.csv"
    fields = list(summary_rows[0].keys()) if summary_rows else []
    with csv_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)

    md_lines = [
        "# Composite query ablation (cumulative tags in one CLAP prompt)",
        "",
        f"- **K:** {top_k}",
        f"- **Tags:** piano → piano vocal → piano vocal relaxing (AND relevance on gold multihot)",
        f"- **No** trailing `music` in prompts",
        f"- **Fine-tuned seeds:** {len(ft_paths)}",
        "",
        "| n_tags | query_text | n⁺ | prev | Pretrained P@K | FT P@K (mean±std) | ΔP | Pretrained ΔnDCG | FT ΔnDCG |",
        "|--------|------------|-----|------|----------------|-------------------|-----|------------------|----------|",
    ]
    for r in summary_rows:
        pm = r["ft_precision_mean"]
        psd = r["ft_precision_stdev"]
        ft_p = f"{pm}±{psd}" if pm else "—"
        md_lines.append(
            f"| {r['n_tags_in_prompt']} | {r['query_text']} | {r['n_positive']} | {float(r['prevalence']):.3f} | "
            f"{r['pretrained_precision_at_k']} | {ft_p} | {r['delta_precision_vs_pretrained']} | "
            f"{r['pretrained_ndcg_delta']} | {r['ft_ndcg_delta_mean']} |"
        )
    md_lines.append("")
    md_out = composite_dir / "composite_query_report.md"
    md_out.write_text("\n".join(md_lines), encoding="utf-8")

    meta = {
        "pretrained_csv": str(pre_path),
        "ft_csvs": [str(p) for p in ft_paths],
        "top_k": top_k,
        "outputs": {"summary_csv": str(csv_out), "markdown": str(md_out)},
    }
    (composite_dir / "composite_query_summary.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
