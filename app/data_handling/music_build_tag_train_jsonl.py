"""
Build train JSONL with primary-tag-derived ``text`` (full corpus + fallback).

Joins human gold multihot on ``source_path``; unlabeled songs get ``--fallback-text``.
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

from app.self_train.jsonl_io import resolve_project_path, write_jsonl_rows

# tag_id -> query token (aligned with music_eval_retrieval_vs_random / PRIMARY_TAGS)
DEFAULT_PRIMARY_TAGS: list[tuple[str, str]] = [
    ("inst_piano", "piano"),
    ("inst_vocal", "vocal"),
    ("mood_relaxing", "relaxing"),
]


def _norm_source_key(source_path: str) -> str:
    return str(resolve_project_path(source_path))


def load_gold_multihot_by_source(gold_jsonl: Path) -> dict[str, dict[str, int]]:
    """Map normalized source_path -> human_multihot dict."""
    if not gold_jsonl.is_file():
        raise FileNotFoundError(f"Gold JSONL not found: {gold_jsonl}")

    out: dict[str, dict[str, int]] = {}
    with gold_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            sp = row.get("source_path")
            hm = row.get("human_multihot")
            if not isinstance(sp, str) or not sp.strip():
                continue
            if not isinstance(hm, dict):
                continue
            labels: dict[str, int] = {}
            for k, v in hm.items():
                try:
                    labels[str(k)] = int(v)
                except (TypeError, ValueError):
                    continue
            key = _norm_source_key(sp)
            if key not in out:
                out[key] = labels
    return out


def format_tag_text(
    multihot: dict[str, int],
    *,
    primary_tags: list[tuple[str, str]],
    fallback_text: str,
) -> tuple[str, str]:
    """
    Return (caption_text, text_source).

    text_source is tag_primary (>=1 active tag), tag_fallback (gold but none active),
    or tag_no_gold (caller sets when no gold row).
    """
    parts: list[str] = []
    for tag_id, query_token in primary_tags:
        if multihot.get(tag_id, 0) == 1:
            parts.append(query_token)
    if parts:
        return ", ".join(parts), "tag_primary"
    return fallback_text, "tag_fallback"


def build_tag_train_jsonl(
    *,
    train_jsonl: Path,
    gold_jsonl: Path,
    out_jsonl: Path,
    fallback_text: str = "music",
    primary_tags: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    if not train_jsonl.is_file():
        raise FileNotFoundError(f"Train JSONL not found: {train_jsonl}")

    tags = primary_tags or DEFAULT_PRIMARY_TAGS
    gold_by_source = load_gold_multihot_by_source(gold_jsonl)

    out_rows: list[dict[str, Any]] = []
    n_primary = 0
    n_fallback = 0
    n_no_gold = 0
    n_songs_with_gold: set[str] = set()

    with train_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            source_path = row.get("source_path")
            if not isinstance(source_path, str) or not source_path.strip():
                continue

            out = dict(row)
            orig_text = out.get("text")
            if not isinstance(orig_text, str):
                orig_text = ""
            out["text_orig"] = orig_text

            key = _norm_source_key(source_path)
            hm = gold_by_source.get(key)
            if hm is None:
                out["text"] = fallback_text
                out["text_source"] = "tag_no_gold"
                n_no_gold += 1
            else:
                n_songs_with_gold.add(key)
                text, source = format_tag_text(
                    hm,
                    primary_tags=tags,
                    fallback_text=fallback_text,
                )
                out["text"] = text
                out["text_source"] = source
                if source == "tag_primary":
                    n_primary += 1
                else:
                    n_fallback += 1

            out_rows.append(out)

    if not out_rows:
        raise ValueError(f"No train rows read from {train_jsonl}")

    write_jsonl_rows(out_jsonl, out_rows)
    return {
        "train_jsonl": str(train_jsonl.resolve()),
        "gold_jsonl": str(gold_jsonl.resolve()),
        "out_jsonl": str(out_jsonl.resolve()),
        "n_total": len(out_rows),
        "n_tag_primary": n_primary,
        "n_tag_fallback": n_fallback,
        "n_tag_no_gold": n_no_gold,
        "n_unique_songs_with_gold": len(n_songs_with_gold),
        "n_gold_songs_in_file": len(gold_by_source),
        "fallback_text": fallback_text,
        "primary_tags": [t[0] for t in tags],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build CLAP train JSONL with primary-tag text (full corpus + fallback)."
    )
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        default=settings.CLAP_TRAIN_JSONL,
    )
    parser.add_argument(
        "--gold-jsonl",
        type=Path,
        default=settings.DATA_DIR / "eval" / "gold_merged.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=settings.MAPPING_DIR / "clap_train_tag.jsonl",
    )
    parser.add_argument(
        "--fallback-text",
        type=str,
        default="music",
        help="Text for clips without gold or with all primary tags off.",
    )
    parser.add_argument(
        "--primary-tags",
        type=str,
        default="",
        help="Comma-separated tag_ids (default: inst_piano,inst_vocal,mood_relaxing).",
    )
    args = parser.parse_args()

    primary_tags = DEFAULT_PRIMARY_TAGS
    if args.primary_tags.strip():
        tag_ids = [t.strip() for t in args.primary_tags.split(",") if t.strip()]
        lookup = dict(DEFAULT_PRIMARY_TAGS)
        primary_tags = []
        for tid in tag_ids:
            if tid not in lookup:
                raise ValueError(f"Unknown primary tag_id: {tid}")
            primary_tags.append((tid, lookup[tid]))

    summary = build_tag_train_jsonl(
        train_jsonl=args.train_jsonl.resolve(),
        gold_jsonl=args.gold_jsonl.resolve(),
        out_jsonl=args.out.resolve(),
        fallback_text=args.fallback_text.strip() or "music",
        primary_tags=primary_tags,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
