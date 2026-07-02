"""Tests for domain tradeoff 2×2 report."""
from __future__ import annotations

import csv
from pathlib import Path

from app.data_handling.music_eval_domain_tradeoff_report import (
    _classify_tradeoff,
    build_domain_tradeoff_report,
)


def test_classify_forgetting() -> None:
    label = _classify_tradeoff(-0.02, 0.20, tag_id="inst_vocal")
    assert "Forgetting" in label


def test_classify_specialization() -> None:
    label = _classify_tradeoff(0.15, -0.10, tag_id="inst_piano")
    assert "Specialization" in label


def _write_gold_csv(path: Path, piano: float, vocal: float, relaxing: float) -> None:
    rows = [
        ("inst_piano", "piano", piano),
        ("inst_vocal", "vocal", vocal),
        ("mood_relaxing", "relaxing", relaxing),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "query_id",
                "query_text",
                "top_k",
                "precision_at_k",
                "ndcg_delta",
            ],
        )
        w.writeheader()
        for qid, qt, p in rows:
            w.writerow(
                {
                    "query_id": qid,
                    "query_text": qt,
                    "top_k": 10,
                    "precision_at_k": p,
                    "ndcg_delta": p * 0.5,
                }
            )


def _write_public_csv(path: Path, piano: float, vocal: float, relaxing: float) -> None:
    rows = [
        ("inst_piano", "piano", piano),
        ("inst_vocal", "vocal", vocal),
        ("mood_relaxing", "relaxing", relaxing),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "query_id",
                "query_text",
                "top_k",
                "n_positive",
                "n_pool",
                "precision_at_k",
                "ndcg_delta",
            ],
        )
        w.writeheader()
        for qid, qt, p in rows:
            w.writerow(
                {
                    "query_id": qid,
                    "query_text": qt,
                    "top_k": 10,
                    "n_positive": 10,
                    "n_pool": 100,
                    "precision_at_k": p,
                    "ndcg_delta": p * 0.5,
                }
            )


def test_build_domain_tradeoff_report(tmp_path: Path) -> None:
    trade = tmp_path / "domain_tradeoff"
    trade.mkdir()
    eval_root = tmp_path / "eval"
    for seed in (42, 43, 44):
        _write_gold_csv(trade / f"anime_only_gold_seed{seed}.csv", 0.5, 0.8, 0.4)
        _write_gold_csv(trade / f"mixed_gold_seed{seed}.csv", 0.55, 0.75, 0.45)
    for ds in ("jamendo", "mtat", "openmic"):
        pub = eval_root / f"{ds}_public"
        pub.mkdir(parents=True)
        for arm in ("pretrained", "thesis_tag_only", "thesis_tag_mixed"):
            for seed in (42, 43, 44):
                if arm == "thesis_tag_only":
                    p, v, r = 0.3, 0.0, 0.2
                elif arm == "thesis_tag_mixed":
                    p, v, r = 0.7, 0.5, 0.3
                else:
                    p, v, r = 0.9, 0.9, 0.4
                _write_public_csv(pub / f"{arm}_seed{seed}.csv", p, v, r)

    meta = build_domain_tradeoff_report(
        trade_dir=trade,
        eval_root=eval_root,
        datasets=["jamendo", "mtat", "openmic"],
        seeds=[42, 43, 44],
        top_k=10,
        anime_arm="thesis_tag_only",
        mixed_arm="thesis_tag_mixed",
    )
    assert (trade / "REPORT.md").is_file()
    assert len(meta["rows"]) == 3
    vocal_row = next(r for r in meta["rows"] if r["tag_id"] == "inst_vocal")
    assert vocal_row["ood_delta_mixed_minus_anime"] > 0


def test_build_grok_domain_tradeoff_report(tmp_path: Path) -> None:
    """Grok E v2 arm IDs produce same 2×2 schema as tag-only E."""
    trade = tmp_path / "domain_tradeoff_grok"
    trade.mkdir()
    eval_root = tmp_path / "eval"
    for seed in (42, 43, 44):
        _write_gold_csv(trade / f"anime_only_gold_seed{seed}.csv", 0.5, 0.8, 0.4)
        _write_gold_csv(trade / f"mixed_gold_seed{seed}.csv", 0.55, 0.75, 0.45)
    for ds in ("jamendo", "mtat", "openmic"):
        pub = eval_root / f"{ds}_public"
        pub.mkdir(parents=True)
        for arm in ("pretrained", "thesis_grok_only", "thesis_grok_mixed"):
            for seed in (42, 43, 44):
                p, v, r = (0.3, 0.4, 0.2) if arm == "thesis_grok_only" else (0.6, 0.5, 0.35)
                if arm == "pretrained":
                    p, v, r = 0.9, 0.9, 0.4
                _write_public_csv(pub / f"{arm}_seed{seed}.csv", p, v, r)

    meta = build_domain_tradeoff_report(
        trade_dir=trade,
        eval_root=eval_root,
        datasets=["jamendo", "mtat", "openmic"],
        seeds=[42, 43, 44],
        top_k=10,
        anime_arm="thesis_grok_only",
        mixed_arm="thesis_grok_mixed",
    )
    text = (trade / "REPORT.md").read_text(encoding="utf-8")
    assert "thesis_grok_only" in text
    assert "thesis_grok_mixed" in text
    assert "## 2×2 P@K (mean over seeds)" in text
    assert len(meta["rows"]) == 3
    for row in meta["rows"]:
        assert "anime_only_gold" in row
        assert "mixed_gold" in row
        assert "anime_only_ood_macro" in row
        assert "mixed_ood_macro" in row
        assert "gold_delta_mixed_minus_anime" in row
        assert "ood_delta_mixed_minus_anime" in row
