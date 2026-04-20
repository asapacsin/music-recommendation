"""
Merge human-reviewed human_pass_way.json with high-confidence rows from music_metadata.json.

Output: process_meta.json

- All rows from human_pass_way.json (post human edit) are included and override by audio key.
- Rows from music_metadata.json with confidence > 0.7 are included when not overridden.
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
) -> list[dict[str, Any]]:
    human_rows = load_json_array(human_pass_way_path)
    music_rows = load_music_metadata(music_metadata_path)

    by_audio: dict[str, dict[str, Any]] = {}

    for row in music_rows:
        c = _confidence(row)
        if c is None or c <= 0.7:
            continue
        key = _audio_key(row)
        if key is None:
            continue
        by_audio[key] = row

    for row in human_rows:
        key = _audio_key(row)
        if key is None:
            logging.warning("Skipping human row without valid audio key: %s", row)
            continue
        by_audio[key] = row

    return [by_audio[k] for k in sorted(by_audio.keys())]


def save_json(path: Path, payload: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge human_pass_way.json with music_metadata (confidence > 0.7) into process_meta.json."
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

    merged = merge_process_meta(args.human, args.music_metadata)
    save_json(args.output, merged)
    logging.info("Wrote %d rows to %s", len(merged), args.output)


if __name__ == "__main__":
    main()
