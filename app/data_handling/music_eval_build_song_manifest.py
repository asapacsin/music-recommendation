"""
Build a song-level evaluation manifest: K representative 15s clips per source track.

Clip selection (deterministic):
  - segment_index 0
  - middle: floor((n - 1) / 2)  [middle of 0..n-1]
  - last: n - 1

Duplicates are removed if n < 3.

Outputs JSONL rows:
  source_path, num_segments, eval_segment_indices, eval_audio_paths (segment_path values)
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
        raise FileNotFoundError(f"Missing file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array: {path}")
    return [x for x in data if isinstance(x, dict)]


def _unique_sources_from_val_jsonl(path: Path) -> set[str]:
    out: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            sp = row.get("source_path")
            if isinstance(sp, str) and sp.strip():
                out.add(sp.strip())
    return out


def _pick_eval_indices(n: int) -> list[int]:
    if n <= 0:
        return []
    if n == 1:
        return [0]
    mid = (n - 1) // 2
    last = n - 1
    indices = [0, mid, last]
    seen: list[int] = []
    for i in indices:
        if i not in seen:
            seen.append(i)
    return sorted(seen)


def _group_map_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        src = row.get("source_path")
        seg = row.get("segment_path")
        if not isinstance(src, str) or not isinstance(seg, str):
            continue
        groups.setdefault(src.strip(), []).append(row)
    for src in groups:
        groups[src].sort(key=lambda r: int(r.get("segment_index", 0)))
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description="Build song-level multi-clip eval manifest.")
    parser.add_argument(
        "--map-json",
        type=Path,
        default=settings.MAPPING_DIR / "music_15s_map.json",
        help="music_15s_map.json",
    )
    parser.add_argument(
        "--filter-val-jsonl",
        type=Path,
        default=None,
        help="If set, only include songs whose source_path appears in this JSONL (e.g. clap_val_15s.jsonl).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=settings.DATA_DIR / "eval" / "song_eval_manifest.jsonl",
        help="Output manifest JSONL.",
    )
    parser.add_argument(
        "--max-songs",
        type=int,
        default=0,
        help="If >0, cap number of songs (stable order: sorted source_path).",
    )
    parser.add_argument(
        "--random-sample",
        type=int,
        default=0,
        help="If >0, randomly sample this many songs after filtering (use with --seed). "
        "Recommended ~150 for gold labeling when val pool is large.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for --random-sample.",
    )
    args = parser.parse_args()

    map_rows = _read_json_array(args.map_json)
    groups = _group_map_rows(map_rows)

    allowed: set[str] | None = None
    if args.filter_val_jsonl is not None:
        allowed = _unique_sources_from_val_jsonl(args.filter_val_jsonl)

    manifest_rows: list[dict[str, Any]] = []
    for source_path in sorted(groups.keys()):
        if allowed is not None and source_path not in allowed:
            continue
        segs = groups[source_path]
        n = len(segs)
        positions = _pick_eval_indices(n)
        paths: list[str] = []
        seg_indices: list[int] = []
        for pos in positions:
            if not (0 <= pos < len(segs)):
                continue
            row = segs[pos]
            spath = row.get("segment_path")
            if isinstance(spath, str):
                paths.append(spath)
            try:
                seg_indices.append(int(row.get("segment_index", pos)))
            except (TypeError, ValueError):
                seg_indices.append(pos)
        manifest_rows.append(
            {
                "source_path": source_path,
                "num_segments": n,
                "eval_positions_in_sorted_list": positions,
                "eval_segment_index": seg_indices,
                "eval_audio_paths": paths,
            }
        )

    if args.max_songs > 0:
        manifest_rows = manifest_rows[: args.max_songs]

    if args.random_sample > 0:
        if args.random_sample >= len(manifest_rows):
            pass  # keep all
        else:
            rng = random.Random(args.seed)
            manifest_rows = rng.sample(manifest_rows, args.random_sample)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "songs": len(manifest_rows),
                "output": str(args.out),
                "filtered_by_val": str(args.filter_val_jsonl) if args.filter_val_jsonl else None,
                "random_sample": args.random_sample if args.random_sample > 0 else None,
                "seed": args.seed if args.random_sample > 0 else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
