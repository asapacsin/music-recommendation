"""
Build CLAP train/val manifests from 15-second segment mapping using grouped split by source song.

Default inputs:
  - data/mapping/music_15s_map.json
  - data/mapping/music_metadata_gt_0_35.json

Default outputs:
  - data/mapping/clap_train_15s.jsonl
  - data/mapping/clap_val_15s.jsonl
  - data/mapping/clap_split_summary.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import settings


def _read_json_array(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON array in {path}")
    return [row for row in payload if isinstance(row, dict)]


def _norm_audio_key(value: str) -> str:
    s = value.strip().replace("\\", "/").lstrip("./")
    prefix = "data/music_db/"
    if s.startswith(prefix):
        s = s[len(prefix) :]
    return s


def _metadata_index(metadata_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in metadata_rows:
        audio = row.get("audio")
        if isinstance(audio, str) and audio.strip():
            index[_norm_audio_key(audio)] = row
    return index


def _group_rows_by_source(
    map_rows: list[dict[str, Any]],
    meta_index: dict[str, dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    groups: dict[str, list[dict[str, Any]]] = {}
    unmatched = 0
    for row in map_rows:
        source = row.get("source_path")
        segment = row.get("segment_path")
        if not isinstance(source, str) or not isinstance(segment, str):
            continue

        source_key = _norm_audio_key(source)
        meta = meta_index.get(source_key)
        if meta is None:
            unmatched += 1
            continue

        sample = {
            "audio_path": segment,
            "text": meta.get("text") if isinstance(meta.get("text"), str) else "",
            "mood": meta.get("mood"),
            "confidence": meta.get("confidence"),
            "source_path": source,
            "segment_index": row.get("segment_index"),
            "start_sec": row.get("start_sec"),
            "end_sec": row.get("end_sec"),
            "duration_sec": row.get("duration_sec"),
        }
        groups.setdefault(source_key, []).append(sample)
    return groups, unmatched


def _split_grouped_by_source(
    groups: dict[str, list[dict[str, Any]]],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    keys = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    total_segments = sum(len(groups[k]) for k in keys)
    target_val = max(1, int(round(total_segments * val_ratio))) if total_segments > 1 else 0

    val_keys: set[str] = set()
    val_count = 0
    for key in keys:
        if val_count >= target_val:
            break
        val_keys.add(key)
        val_count += len(groups[key])

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for key in sorted(keys):
        if key in val_keys:
            val_rows.extend(groups[key])
        else:
            train_rows.extend(groups[key])
    return train_rows, val_rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_train_val(
    map_path: Path,
    metadata_path: Path,
    train_out: Path,
    val_out: Path,
    summary_out: Path,
    *,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, int]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")

    map_rows = _read_json_array(map_path)
    metadata_rows = _read_json_array(metadata_path)
    meta_index = _metadata_index(metadata_rows)
    groups, unmatched = _group_rows_by_source(map_rows, meta_index)
    train_rows, val_rows = _split_grouped_by_source(groups, val_ratio=val_ratio, seed=seed)

    _write_jsonl(train_out, train_rows)
    _write_jsonl(val_out, val_rows)

    summary = {
        "mapping_rows": len(map_rows),
        "metadata_rows": len(metadata_rows),
        "matched_segment_rows": len(train_rows) + len(val_rows),
        "unmatched_segment_rows": unmatched,
        "group_count_sources": len(groups),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "val_ratio": val_ratio,
        "seed": seed,
        "train_out": str(train_out),
        "val_out": str(val_out),
    }
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "group_count_sources": len(groups),
        "unmatched_segment_rows": unmatched,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build grouped train/val manifests from 15s map + metadata."
    )
    parser.add_argument(
        "--map",
        type=Path,
        default=settings.MAPPING_DIR / "music_15s_map.json",
        help="Segment mapping JSON (default: data/mapping/music_15s_map.json).",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=settings.MAPPING_DIR / "music_metadata_gt_0_35.json",
        help="Metadata JSON to provide text/mood labels (default: data/mapping/music_metadata_gt_0_35.json).",
    )
    parser.add_argument(
        "--train-out",
        type=Path,
        default=settings.MAPPING_DIR / "clap_train_15s.jsonl",
        help="Train manifest output JSONL.",
    )
    parser.add_argument(
        "--val-out",
        type=Path,
        default=settings.MAPPING_DIR / "clap_val_15s.jsonl",
        help="Val manifest output JSONL.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=settings.MAPPING_DIR / "clap_split_summary.json",
        help="Summary JSON output.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation ratio at grouped source level target (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for group shuffle (default: 42).",
    )
    args = parser.parse_args()

    stats = build_train_val(
        map_path=args.map,
        metadata_path=args.metadata,
        train_out=args.train_out,
        val_out=args.val_out,
        summary_out=args.summary_out,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(
        "summary: "
        + ", ".join(f"{k}={v}" for k, v in stats.items())
        + f", train_out={args.train_out}, val_out={args.val_out}, summary_out={args.summary_out}"
    )


if __name__ == "__main__":
    main()
