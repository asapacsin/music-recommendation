"""
Scan data/music_db for audio files and summarize file suffix usage.
Writes a report under data/log (see config.settings.LOG_DIR).
"""
from __future__ import annotations

import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import settings


def suffix_key(path: Path) -> str:
    s = path.suffix
    return s.lower() if s else "<no extension>"


def scan_music_db(music_dir: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not music_dir.is_dir():
        return counts
    for p in music_dir.iterdir():
        if p.is_file():
            counts[suffix_key(p)] += 1
    return counts


def format_report(counts: Counter[str], music_dir: Path) -> str:
    lines = [
        f"generated_utc: {datetime.now(timezone.utc).isoformat()}",
        f"music_dir: {music_dir.resolve()}",
        f"distinct_suffix_types: {len(counts)}",
        f"total_files: {sum(counts.values())}",
        "",
        "suffix\tcount",
    ]
    for suffix, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"{suffix}\t{n}")
    return "\n".join(lines) + "\n"


def main() -> None:
    music_dir = settings.MUSIC_DB_DIR
    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)

    counts = scan_music_db(music_dir)
    body = format_report(counts, music_dir)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = settings.LOG_DIR / f"music_suffix_check_{stamp}.log"
    log_path.write_text(body, encoding="utf-8")

    print(body)
    print(f"Wrote: {log_path}")


if __name__ == "__main__":
    main()
