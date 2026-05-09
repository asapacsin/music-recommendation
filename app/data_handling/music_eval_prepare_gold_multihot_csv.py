"""
Build a minimal human gold-label CSV: song filename + multi-hot 0/1 columns only.

Instrumentation: inst_piano, inst_orchestral, inst_vocal
Mood / character: mood_sad_melancholic, mood_relaxing, mood_dark_tense, mood_exciting, mood_elegant, mood_epic

Tempo omitted (program BPM). No uncertain/notes/eval_clip_paths/source_path in CSV.

Writes sidecar JSONL (song_name + source_path) beside CSV unless --no-sidecar.

Rows come from song_eval_manifest.jsonl by default.
CSV is UTF-8 with BOM for Excel on Windows.
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

MULTIHOT_COLUMNS: list[str] = [
    "inst_piano",
    "inst_orchestral",
    "inst_vocal",
    "mood_sad_melancholic",
    "mood_relaxing",
    "mood_dark_tense",
    "mood_exciting",
    "mood_elegant",
    "mood_epic",
]


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare minimal multi-hot gold label CSV (song name only).")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=settings.DATA_DIR / "eval" / "song_eval_manifest.jsonl",
        help="Song manifest JSONL (source_path + eval_audio_paths).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=settings.DATA_DIR / "eval" / "gold_labels_multihot_template.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--default-value",
        choices=("0", "empty"),
        default="0",
        help="Initial value for label columns.",
    )
    parser.add_argument(
        "--no-bom",
        action="store_true",
        help="Write UTF-8 without BOM (default: UTF-8 BOM for Excel).",
    )
    parser.add_argument(
        "--no-sidecar",
        action="store_true",
        help="Do not write song_name↔source_path sidecar JSONL.",
    )
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    if not manifest:
        raise ValueError("Manifest is empty.")

    fieldnames = ["song_name"] + MULTIHOT_COLUMNS
    default_cell = "" if args.default_value == "empty" else "0"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    encoding = "utf-8" if args.no_bom else "utf-8-sig"
    sidecar_path = args.out.with_suffix(args.out.suffix + ".sidecar.jsonl")

    rows_written = 0
    with args.out.open("w", encoding=encoding, newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        sf = (
            None
            if args.no_sidecar
            else sidecar_path.open("w", encoding="utf-8")
        )
        try:
            for row in manifest:
                sp = row.get("source_path")
                if not isinstance(sp, str) or not sp.strip():
                    continue
                sp = sp.strip()
                song_name = Path(sp.replace("\\", "/")).name
                out_row: dict[str, str] = {"song_name": song_name}
                for col in MULTIHOT_COLUMNS:
                    out_row[col] = default_cell
                writer.writerow(out_row)
                rows_written += 1
                if sf is not None:
                    sf.write(
                        json.dumps({"song_name": song_name, "source_path": sp}, ensure_ascii=False)
                        + "\n"
                    )
        finally:
            if sf is not None:
                sf.close()

    payload = {
        "rows_written": rows_written,
        "columns": fieldnames,
        "output": str(args.out),
        "encoding": encoding,
        "sidecar": None if args.no_sidecar else str(sidecar_path),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
