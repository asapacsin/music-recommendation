"""
Merge human-reviewed human_pass_way.json with the process layer and music_metadata.json.

Output: process_meta.json

Merge order (later steps override earlier ones by ``audio`` key):

1. Optional existing process_meta.json — baseline "process" snapshot (all rows kept).
2. Rows from music_metadata.json with confidence > threshold (auto layer; only fills/updates
   where useful; see merge logic below).
3. All rows from human_pass_way.json — human-verified records always win, regardless of their
   confidence value.

This ensures human-reviewed entries are never dropped or skipped because of low confidence.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import settings


def _confidence(record: dict[str, Any]) -> float | None:
    c = record.get("confidence")
    if isinstance(c, (int, float)) and not isinstance(c, bool):
        return float(c)
    return None


def _audio_key(record: dict[str, Any]) -> str | None:
    a = record.get("audio")
    if isinstance(a, str) and a.strip():
        return a.strip()
    return None


def load_json_array(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return [x for x in data if isinstance(x, dict)]


def load_music_metadata(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing metadata file: {path}")
    return load_json_array(path)


def merge_process_meta(
    human_pass_way_path: Path,
    music_metadata_path: Path,
    *,
    process_meta_path: Path | None,
    music_confidence_min: float,
) -> list[dict[str, Any]]:
    """
    Build merged records keyed by ``audio``.

    Human rows are applied last and are never filtered by confidence.
    """
    by_audio: dict[str, dict[str, Any]] = {}

    if process_meta_path is not None and process_meta_path.is_file():
        for row in load_json_array(process_meta_path):
            key = _audio_key(row)
            if key is not None:
                by_audio[key] = row
        logging.info(
            "Seeded %d rows from process file %s",
            len(by_audio),
            process_meta_path,
        )

    music_rows = load_music_metadata(music_metadata_path)
    auto_added = 0
    for row in music_rows:
        c = _confidence(row)
        if c is None or c <= music_confidence_min:
            continue
        key = _audio_key(row)
        if key is None:
            continue
        by_audio[key] = row
        auto_added += 1
    logging.info(
        "Applied %d music_metadata rows with confidence > %s",
        auto_added,
        music_confidence_min,
    )

    human_rows = load_json_array(human_pass_way_path)
    human_applied = 0
    for row in human_rows:
        key = _audio_key(row)
        if key is None:
            logging.warning("Skipping human row without valid audio key: %s", row)
            continue
        # Human-verified: always merge, regardless of confidence on the row.
        by_audio[key] = row
        human_applied += 1
    logging.info(
        "Applied %d human_pass_way rows (no confidence filter)",
        human_applied,
    )

    return [by_audio[k] for k in sorted(by_audio.keys())]


def save_json(path: Path, payload: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge process_meta (optional seed), music_metadata (confidence threshold), and "
            "human_pass_way (always wins, any confidence) into process_meta.json."
        )
    )
    parser.add_argument(
        "--human",
        type=Path,
        default=settings.HUMAN_PASS_WAY_FILE,
        help="Human-reviewed human_pass_way.json",
    )
    parser.add_argument(
        "--music-metadata",
        type=Path,
        default=settings.MUSIC_METADATA_FILE,
        help="Source music_metadata.json",
    )
    parser.add_argument(
        "--process",
        type=Path,
        default=settings.PROCESS_META_FILE,
        help=(
            "Existing process_meta.json to seed the merge (optional; use --no-process-seed to skip). "
            "Default: same as --output path."
        ),
    )
    parser.add_argument(
        "--no-process-seed",
        action="store_true",
        help="Do not load existing process_meta.json as the first layer.",
    )
    parser.add_argument(
        "--music-confidence-min",
        type=float,
        default=0.7,
        help=(
            "Include rows from music_metadata only when confidence is strictly greater than this "
            "(default: 0.7). Human_pass_way is never filtered by confidence."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.PROCESS_META_FILE,
        help="Output process_meta.json",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if args.music_confidence_min < 0 or args.music_confidence_min > 1:
        raise ValueError("--music-confidence-min must be between 0 and 1")

    process_seed: Path | None = None
    if not args.no_process_seed:
        process_seed = args.process

    merged = merge_process_meta(
        args.human,
        args.music_metadata,
        process_meta_path=process_seed,
        music_confidence_min=args.music_confidence_min,
    )
    save_json(args.output, merged)
    logging.info("Wrote %d rows to %s", len(merged), args.output)


if __name__ == "__main__":
    main()
