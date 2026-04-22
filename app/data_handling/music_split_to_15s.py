"""
Split audio files in data/music_db into fixed-length clips (default 15s)
and write a mapping file from each segment back to its source track.

By default, only tracks with confidence > 0.35 from
data/mapping/music_metadata_gt_0_35.json are split.

Default outputs:
  - Segments: data/music_db_15s
  - Mapping:  data/mapping/music_15s_map.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from tqdm import tqdm

from config import settings


def _ffmpeg_exe() -> str | None:
    return os.environ.get("FFMPEG_PATH") or shutil.which("ffmpeg")


def _ffprobe_exe() -> str | None:
    return os.environ.get("FFPROBE_PATH") or shutil.which("ffprobe")


def _duration_seconds(ffprobe: str, path: Path) -> float:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    value = proc.stdout.strip()
    return float(value)


def _extract_segment(ffmpeg: str, src: Path, dst: Path, start_s: float, dur_s: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_s:.6f}",
        "-i",
        str(src),
        "-t",
        f"{dur_s:.6f}",
        "-codec:a",
        "libmp3lame",
        "-qscale:a",
        "2",
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _safe_stem(rel_path: str) -> str:
    stem = Path(rel_path).with_suffix("")
    return "__".join(stem.parts)


def _clip_name(rel_path: str, seg_idx: int) -> str:
    return f"{_safe_stem(rel_path)}__seg{seg_idx:03d}.mp3"


def _rel_posix(path: Path, base: Path) -> str:
    return str(path.relative_to(base)).replace("\\", "/")


def _normalize_audio_key(value: str) -> str:
    """Normalize metadata audio path to music_db-relative posix path."""
    normalized = value.strip().replace("\\", "/")
    normalized = normalized.lstrip("./")
    prefix = "data/music_db/"
    if normalized.startswith(prefix):
        normalized = normalized[len(prefix) :]
    return normalized


def _load_allowed_audio(
    metadata_path: Path,
    *,
    confidence_threshold: float,
) -> set[str]:
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata filter file: {metadata_path}")

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON array in {metadata_path}")

    allowed: set[str] = set()
    for row in payload:
        if not isinstance(row, dict):
            continue
        audio = row.get("audio")
        conf = row.get("confidence")
        if not isinstance(audio, str) or not audio.strip():
            continue
        if not isinstance(conf, (int, float)) or isinstance(conf, bool):
            continue
        if float(conf) > confidence_threshold:
            allowed.add(_normalize_audio_key(audio))
    return allowed


def _load_existing_mapping(mapping_path: Path) -> dict[str, dict[str, Any]]:
    """
    Load existing mapping rows keyed by segment_path.
    Returns empty dict when file does not exist or invalid.
    """
    if not mapping_path.is_file():
        return {}
    try:
        payload = json.loads(mapping_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            return {}
    except (OSError, json.JSONDecodeError):
        return {}

    out: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        seg = row.get("segment_path")
        if isinstance(seg, str) and seg.strip():
            out[seg] = row
    return out


def _save_mapping_rows(mapping_path: Path, by_segment_path: dict[str, dict[str, Any]]) -> None:
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [by_segment_path[k] for k in sorted(by_segment_path.keys())]
    mapping_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def split_music_db(
    source_dir: Path,
    target_dir: Path,
    mapping_path: Path,
    *,
    metadata_filter_path: Path,
    confidence_threshold: float = 0.35,
    segment_seconds: float = 15.0,
    min_tail_seconds: float = 5.0,
    overwrite: bool = False,
    limit: int | None = None,
    progress: bool = True,
) -> dict[str, int]:
    if not source_dir.is_dir():
        raise NotADirectoryError(f"source_dir does not exist: {source_dir}")
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be > 0")
    if min_tail_seconds < 0:
        raise ValueError("min_tail_seconds must be >= 0")
    if not 0.0 < confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in (0, 1]")

    ffmpeg = _ffmpeg_exe()
    ffprobe = _ffprobe_exe()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH. Install FFmpeg and/or set FFMPEG_PATH.")
    if not ffprobe:
        raise RuntimeError("ffprobe not found on PATH. Install FFmpeg and/or set FFPROBE_PATH.")

    allowed_audio = _load_allowed_audio(
        metadata_filter_path,
        confidence_threshold=confidence_threshold,
    )

    files = sorted(
        p
        for p in source_dir.rglob("*")
        if p.is_file() and _rel_posix(p, source_dir) in allowed_audio
    )
    if limit is not None:
        files = files[:limit]

    stats = {
        "sources": len(files),
        "allowed_audio": len(allowed_audio),
        "segments_written": 0,
        "segments_existing": 0,
        "segments_skipped_short_tail": 0,
        "errors": 0,
        "mapping_loaded": 0,
        "mapping_rows_final": 0,
    }
    mapping_by_segment = _load_existing_mapping(mapping_path)
    stats["mapping_loaded"] = len(mapping_by_segment)
    files_since_checkpoint = 0

    iterator = tqdm(
        files,
        desc="Splitting to 15s",
        unit="file",
        disable=not progress,
        dynamic_ncols=True,
    )
    for src in iterator:
        rel_src = _rel_posix(src, source_dir)
        try:
            dur = _duration_seconds(ffprobe, src)
            if dur <= 0:
                stats["errors"] += 1
                continue

            seg_count = int(math.floor(dur / segment_seconds))
            tail = dur - (seg_count * segment_seconds)
            if tail >= min_tail_seconds:
                seg_count += 1
            elif tail > 0:
                stats["segments_skipped_short_tail"] += 1

            for seg_idx in range(seg_count):
                start_s = seg_idx * segment_seconds
                end_s = min((seg_idx + 1) * segment_seconds, dur)
                clip_dur = max(0.0, end_s - start_s)
                if clip_dur <= 0:
                    continue

                clip_name = _clip_name(rel_src, seg_idx)
                dst = target_dir / clip_name
                if dst.exists() and not overwrite:
                    stats["segments_existing"] += 1
                else:
                    _extract_segment(ffmpeg, src, dst, start_s, clip_dur)
                    stats["segments_written"] += 1

                row = {
                    "segment_path": _rel_posix(dst, settings.BASE_DIR),
                    "source_path": _rel_posix(src, settings.BASE_DIR),
                    "segment_index": seg_idx,
                    "start_sec": round(start_s, 3),
                    "end_sec": round(end_s, 3),
                    "duration_sec": round(clip_dur, 3),
                }
                mapping_by_segment[row["segment_path"]] = row
        except Exception as exc:  # noqa: BLE001
            stats["errors"] += 1
            tqdm.write(f"ERROR: {rel_src} :: {exc}")
        finally:
            files_since_checkpoint += 1
            if files_since_checkpoint >= 25:
                _save_mapping_rows(mapping_path, mapping_by_segment)
                files_since_checkpoint = 0

    _save_mapping_rows(mapping_path, mapping_by_segment)
    stats["mapping_rows_final"] = len(mapping_by_segment)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split data/music_db into fixed-length clips and write source->segment mapping. "
            "By default only metadata rows with confidence > 0.35 are included."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=settings.MUSIC_DB_DIR,
        help="Source directory with original tracks (default: data/music_db).",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=settings.DATA_DIR / "music_db_15s",
        help="Output directory for segments (default: data/music_db_15s).",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=settings.MAPPING_DIR / "music_15s_map.json",
        help="Output mapping JSON (default: data/mapping/music_15s_map.json).",
    )
    parser.add_argument(
        "--metadata-filter",
        type=Path,
        default=settings.MAPPING_DIR / "music_metadata_gt_0_35.json",
        help=(
            "Metadata JSON used to select source tracks by confidence "
            "(default: data/mapping/music_metadata_gt_0_35.json)."
        ),
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.35,
        help=(
            "Only split tracks with confidence strictly greater than this threshold "
            "in --metadata-filter (default: 0.35)."
        ),
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=15.0,
        help="Segment duration in seconds (default: 15).",
    )
    parser.add_argument(
        "--min-tail-seconds",
        type=float,
        default=5.0,
        help="Keep final tail segment only if >= this duration (default: 5).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing segment files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N source files (for testing).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress display.",
    )
    args = parser.parse_args()

    stats = split_music_db(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        mapping_path=args.mapping,
        metadata_filter_path=args.metadata_filter,
        confidence_threshold=args.confidence_threshold,
        segment_seconds=args.segment_seconds,
        min_tail_seconds=args.min_tail_seconds,
        overwrite=args.overwrite,
        limit=args.limit,
        progress=not args.no_progress,
    )
    print(
        "summary: "
        + ", ".join(f"{k}={v}" for k, v in stats.items())
        + f", mapping={args.mapping}"
    )


if __name__ == "__main__":
    main()
