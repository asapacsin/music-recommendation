"""
Evaluate confidence in data/mapping/music_metadata.json.

Rows with 0.1 <= confidence <= 0.4 are written to human_pass_way.json for human review.
Rows with confidence > threshold (default 0.35) can be collected to a separate file.
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


def load_music_metadata(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing metadata file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return [x for x in data if isinstance(x, dict)]


def save_json(path: Path, payload: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def evaluate_confidence(
    music_metadata_path: Path,
    human_pass_way_path: Path,
) -> dict[str, int]:
    records = load_music_metadata(music_metadata_path)
    human_queue: list[dict[str, Any]] = []
    counts = {
        "total": len(records),
        "human_pass_way": 0,
        "high_gt_0_4": 0,
        "low_lt_0_1": 0,
        "other": 0,
    }

    for row in records:
        c = _confidence(row)
        if c is None:
            counts["other"] += 1
            continue
        if 0.1 <= c <= 0.4:
            human_queue.append(row)
            counts["human_pass_way"] += 1
        elif c > 0.4:
            counts["high_gt_0_4"] += 1
        elif c < 0.1:
            counts["low_lt_0_1"] += 1
        else:
            counts["other"] += 1

    save_json(human_pass_way_path, human_queue)
    return counts


def collect_high_confidence(
    music_metadata_path: Path,
    high_confidence_path: Path,
    *,
    threshold: float = 0.35,
) -> dict[str, int]:
    """
    Collect rows with confidence strictly greater than threshold.
    """
    records = load_music_metadata(music_metadata_path)
    high_rows: list[dict[str, Any]] = []

    for row in records:
        c = _confidence(row)
        if c is not None and c > threshold:
            high_rows.append(row)

    save_json(high_confidence_path, high_rows)
    return {
        "total": len(records),
        "high_confidence": len(high_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Route 0.1–0.4 confidence rows from music_metadata.json to human_pass_way.json."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=settings.MUSIC_METADATA_FILE,
        help="Source music_metadata.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.HUMAN_PASS_WAY_FILE,
        help="Output human_pass_way.json (mid-confidence queue)",
    )
    parser.add_argument(
        "--collect-high",
        action="store_true",
        help="Also collect rows with confidence > --high-threshold into --high-output.",
    )
    parser.add_argument(
        "--high-threshold",
        type=float,
        default=0.35,
        help="Threshold for high-confidence collector (strictly greater than this value).",
    )
    parser.add_argument(
        "--high-output",
        type=Path,
        default=settings.MAPPING_DIR / "music_metadata_gt_0_35.json",
        help="Output file for high-confidence collector.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    stats = evaluate_confidence(args.input, args.output)
    logging.info(
        "total=%(total)d human_pass_way(0.1–0.4)=%(human_pass_way)d "
        "high(>0.4)=%(high_gt_0_4)d low(<0.1)=%(low_lt_0_1)d other=%(other)d",
        stats,
    )
    logging.info("Wrote %d rows to %s", stats["human_pass_way"], args.output)

    if args.collect_high:
        if not 0.0 <= args.high_threshold <= 1.0:
            raise ValueError("--high-threshold must be between 0 and 1")
        high_stats = collect_high_confidence(
            args.input,
            args.high_output,
            threshold=args.high_threshold,
        )
        logging.info(
            "Collected %d/%d rows with confidence > %s to %s",
            high_stats["high_confidence"],
            high_stats["total"],
            args.high_threshold,
            args.high_output,
        )


if __name__ == "__main__":
    main()
