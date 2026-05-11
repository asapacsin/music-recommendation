"""
Per-tag positive counts and prevalence for human multihot gold labels.

Reads merged JSONL (`human_multihot`) or the raw gold CSV (same columns as prepare/merge).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import settings

from app.data_handling.music_eval_prepare_gold_multihot_csv import MULTIHOT_COLUMNS

MOOD_COLUMNS = [c for c in MULTIHOT_COLUMNS if c.startswith("mood_")]


def _parse_bin_cell(raw: str) -> int:
    v = (raw or "").strip()
    if not v:
        return 0
    return int(float(v))


def _counts_from_jsonl(path: Path) -> tuple[int, dict[str, int], int]:
    counts = {c: 0 for c in MULTIHOT_COLUMNS}
    n = 0
    all_mood_zero = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            h = r.get("human_multihot") if isinstance(r, dict) else None
            if not isinstance(h, dict):
                h = {}
            n += 1
            any_mood = False
            for c in MULTIHOT_COLUMNS:
                try:
                    v = int(h.get(c, 0))
                except (TypeError, ValueError):
                    v = 0
                if v != 0:
                    counts[c] += 1
                    if c in MOOD_COLUMNS:
                        any_mood = True
            if not any_mood:
                all_mood_zero += 1
    return n, counts, all_mood_zero


def _counts_from_csv(path: Path) -> tuple[int, dict[str, int], int]:
    counts = {c: 0 for c in MULTIHOT_COLUMNS}
    n = 0
    all_mood_zero = 0
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n += 1
            any_mood = False
            for c in MULTIHOT_COLUMNS:
                try:
                    v = _parse_bin_cell(str(row.get(c, "") or ""))
                except ValueError:
                    v = 0
                if v != 0:
                    counts[c] += 1
                    if c in MOOD_COLUMNS:
                        any_mood = True
            if not any_mood:
                all_mood_zero += 1
    return n, counts, all_mood_zero


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Count positives per multihot gold label (prevalence / class balance check)."
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=settings.DATA_DIR / "eval" / "gold_merged.jsonl",
        help="Merged gold JSONL (ignored if --csv is set).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Gold CSV (song_name + multihot). When set, reads this instead of --jsonl.",
    )
    args = parser.parse_args()

    use_csv = args.csv is not None
    path = args.csv if use_csv else args.jsonl
    if not path.is_file():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 1

    if use_csv:
        n, counts, all_mood_zero = _counts_from_csv(path)
        src = "csv"
    else:
        n, counts, all_mood_zero = _counts_from_jsonl(path)
        src = "jsonl"

    print(f"source\t{src}")
    print(f"path\t{path}")
    print(f"n_songs\t{n}")
    print(f"rows_no_positive_mood\t{all_mood_zero}")
    print("tag\tcount\tprevalence")
    for c in MULTIHOT_COLUMNS:
        prev = f"{counts[c] / n:.6f}" if n else ""
        print(f"{c}\t{counts[c]}\t{prev}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
