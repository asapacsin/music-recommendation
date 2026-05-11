"""
Export gold_merged.jsonl to a minimal UTF-8-BOM CSV for Excel review.

Columns: song_name, human multihot (0/1), tempo_bin_bpm (slow / mid-tempo / fast from program BPM).
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

from app.data_handling.music_eval_prepare_gold_multihot_csv import MULTIHOT_COLUMNS


def _tempo_bin_str(pt: Any) -> str:
    if not isinstance(pt, dict):
        return ""
    v = pt.get("tempo_bin_bpm")
    return str(v).strip() if isinstance(v, str) else ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Export gold_merged.jsonl to a minimal review CSV.")
    parser.add_argument(
        "--in-jsonl",
        type=Path,
        default=settings.DATA_DIR / "eval" / "gold_merged.jsonl",
        help="Merged gold JSONL (from music_eval_merge_gold).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=settings.DATA_DIR / "eval" / "gold_merged_review.csv",
        help="Output CSV path (UTF-8 BOM for Excel unless --no-bom).",
    )
    parser.add_argument(
        "--no-bom",
        action="store_true",
        help="Write UTF-8 without BOM.",
    )
    args = parser.parse_args()

    if not args.in_jsonl.is_file():
        raise FileNotFoundError(f"Input not found: {args.in_jsonl}")

    fieldnames = ["song_name", *MULTIHOT_COLUMNS, "tempo_bin_bpm"]

    rows_out: list[dict[str, str]] = []
    with args.in_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                continue
            song = (rec.get("song_name") or "").strip()
            hm = rec.get("human_multihot")
            if not isinstance(hm, dict):
                hm = {}
            pt = rec.get("program_tempo")

            row: dict[str, str] = {"song_name": song}
            for col in MULTIHOT_COLUMNS:
                v = hm.get(col, 0)
                try:
                    row[col] = "1" if int(v) != 0 else "0"
                except (TypeError, ValueError):
                    row[col] = "0"
            row["tempo_bin_bpm"] = _tempo_bin_str(pt)

            rows_out.append(row)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    enc = "utf-8" if args.no_bom else "utf-8-sig"
    with args.out_csv.open("w", encoding=enc, newline="") as cf:
        w = csv.DictWriter(cf, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows_out:
            w.writerow(row)

    print(json.dumps({"rows": len(rows_out), "out_csv": str(args.out_csv)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
