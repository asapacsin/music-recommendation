"""
Convert non-MP3 audio files in data/music_db to MP3 in place.

Each file is transcoded to a temporary MP3 next to the source, then the
original is deleted and the temp is renamed to <stem>.mp3 (same directory).
Already-.mp3 files are left unchanged.

Requires the ffmpeg executable on PATH (same as MoviePy audio export).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import settings


def _ffmpeg_exe() -> str | None:
    return os.environ.get("FFMPEG_PATH") or shutil.which("ffmpeg")


def _run_ffmpeg(ffmpeg: str, src: Path, dst: Path) -> None:
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-codec:a",
        "libmp3lame",
        "-qscale:a",
        "2",
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def convert_file_in_place(src: Path, *, ffmpeg: str) -> None:
    if not src.is_file():
        raise FileNotFoundError(str(src))
    if src.suffix.lower() == ".mp3":
        return

    dest = src.with_suffix(".mp3")

    fd, tmp_str = tempfile.mkstemp(
        suffix=".mp3",
        prefix=".conv_",
        dir=str(src.parent),
    )
    os.close(fd)
    tmp = Path(tmp_str)

    try:
        _run_ffmpeg(ffmpeg, src, tmp)
        src.unlink()
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def convert_music_db(
    music_dir: Path | None = None,
    *,
    dry_run: bool = False,
    limit: int | None = None,
    log_file: Path | None = None,
) -> dict[str, int]:
    music_dir = music_dir or settings.MUSIC_DB_DIR
    stats: dict[str, int] = {
        "skipped_mp3": 0,
        "dry_run": 0,
        "ok": 0,
        "error": 0,
    }
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        log_lines.append(msg)

    if not music_dir.is_dir():
        log(f"ERROR: not a directory: {music_dir}")
        return stats

    all_files = sorted(p for p in music_dir.iterdir() if p.is_file())
    stats["skipped_mp3"] = sum(1 for p in all_files if p.suffix.lower() == ".mp3")
    candidates = [p for p in all_files if p.suffix.lower() != ".mp3"]
    if limit is not None:
        candidates = candidates[:limit]

    ffmpeg: str | None = None
    if not dry_run and candidates:
        ffmpeg = _ffmpeg_exe()
        if not ffmpeg:
            raise RuntimeError(
                "ffmpeg not found on PATH. Install FFmpeg and/or set FFMPEG_PATH."
            )

    log(f"start_utc={datetime.now(timezone.utc).isoformat()}")
    log(f"music_dir={music_dir.resolve()}")
    log(f"ffmpeg={ffmpeg or '(not required for dry-run)'}")
    log(
        f"dry_run={dry_run} "
        f"to_convert={len(candidates)} skipped_mp3={stats['skipped_mp3']}"
    )

    for path in candidates:
        try:
            if dry_run:
                stats["dry_run"] += 1
                log(f"DRY_RUN would convert: {path.name}")
                continue
            assert ffmpeg is not None
            convert_file_in_place(path, ffmpeg=ffmpeg)
            stats["ok"] += 1
            log(f"OK: {path.name} -> {path.stem}.mp3")
        except subprocess.CalledProcessError as e:
            stats["error"] += 1
            err = (e.stderr or e.stdout or str(e)).strip()
            log(f"ERROR: {path.name} :: {err[:500]}")
        except OSError as e:
            stats["error"] += 1
            log(f"ERROR: {path.name} :: {e}")

    log(
        "summary: "
        + ", ".join(f"{k}={v}" for k, v in stats.items() if v or k in ("ok", "error"))
    )

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        print(f"Wrote log: {log_file}")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert audio in data/music_db to MP3 in place (non-mp3 only)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be converted; do not change disk.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N non-mp3 files (for testing).",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Do not write data/log/music_to_mp3_<timestamp>.log",
    )
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = None if args.no_log_file else settings.LOG_DIR / f"music_to_mp3_{stamp}.log"

    convert_music_db(
        dry_run=args.dry_run,
        limit=args.limit,
        log_file=log_path,
    )


if __name__ == "__main__":
    main()
