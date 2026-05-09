"""
Upgrade an existing human gold CSV when MULTIHOT_COLUMNS change.

Preserves row order (keeps alignment with .sidecar.jsonl line order).
Copies values for columns that exist in the old file; new columns default to 0.
Drops obsolete columns not in the current taxonomy.

Does NOT change the sidecar — regenerate sidecar only if the song list changed.

Usage:
  python -m app.data_handling.music_eval_upgrade_gold_csv --csv data/eval/gold_labels_multihot_template.csv --in-place
  python -m app.data_handling.music_eval_upgrade_gold_csv --csv old.csv --out new.csv
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from app.data_handling.music_eval_prepare_gold_multihot_csv import MULTIHOT_COLUMNS


def _parse_bin(raw: str) -> str:
    v = (raw or "").strip()
    if not v:
        return "0"
    try:
        n = int(float(v))
        return "1" if n != 0 else "0"
    except ValueError:
        return "0"


def main() -> int:
    parser = argparse.ArgumentParser(description="Upgrade gold CSV columns without losing filled cells.")
    parser.add_argument("--csv", type=Path, required=True, help="Input gold CSV (UTF-8-BOM ok).")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path. Default: <stem>_upgraded.csv next to input.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Write back to --csv after copying to .bak backup.",
    )
    parser.add_argument("--no-bom", action="store_true", help="Write UTF-8 without BOM.")
    args = parser.parse_args()

    if not args.csv.is_file():
        raise FileNotFoundError(f"Missing CSV: {args.csv}")

    fieldnames = ["song_name"] + MULTIHOT_COLUMNS
    encoding_in = "utf-8-sig"
    encoding_out = "utf-8" if args.no_bom else "utf-8-sig"

    out_path = args.csv if args.in_place else (args.out or args.csv.with_name(f"{args.csv.stem}_upgraded{args.csv.suffix}"))

    rows_out: list[dict[str, str]] = []
    skipped = 0
    with args.csv.open("r", encoding=encoding_in, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("song_name") or "").strip()
            if not name:
                skipped += 1
                continue
            out_row: dict[str, str] = {"song_name": name}
            for col in MULTIHOT_COLUMNS:
                out_row[col] = _parse_bin(str(row.get(col, "")))
            rows_out.append(out_row)

    if args.in_place:
        bak = args.csv.with_suffix(args.csv.suffix + ".bak")
        shutil.copy2(args.csv, bak)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding=encoding_out, newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    print(
        f"rows_written={len(rows_out)} skipped_no_name={skipped} "
        f"columns={len(MULTIHOT_COLUMNS)} output={out_path}"
    )
    if args.in_place:
        print(f"backup={args.csv.with_suffix(args.csv.suffix + '.bak')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
