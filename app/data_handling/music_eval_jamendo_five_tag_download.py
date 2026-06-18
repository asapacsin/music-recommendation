"""
Build a capped MTG-Jamendo eval manifest for five public tags and download matching audio.

Uses split-0 **test** TSVs (instrument + moodtheme), 30 s ``raw_30s`` MP3s from the MTG CDN
(``https://cdn.freesound.org/mtg-jamendo/raw_30s/audio/<PATH>``).

Default tags (Jamendo tag strings):
  pub_piano    -> instrument---piano
  pub_guitar   -> instrument---guitar
  pub_vocal    -> instrument---voice
  pub_relaxing -> mood/theme---relaxing
  pub_epic     -> mood/theme---epic

Example:
  python -m app.data_handling.music_eval_jamendo_five_tag_download --max-per-tag 60
  python -m app.data_handling.music_eval_jamendo_five_tag_download --manifest-only
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import settings

JAMENDO_REPO = settings.DATA_DIR / "public_eval" / "jamendo" / "mtg-jamendo-dataset"
SPLIT_DIR = JAMENDO_REPO / "data" / "splits" / "split-0"
DEFAULT_AUDIO_DIR = settings.DATA_DIR / "public_eval" / "jamendo" / "audio_five_tag"
DEFAULT_MANIFEST = settings.DATA_DIR / "eval" / "jamendo_five_tag_manifest.jsonl"
CDN_AUDIO_URL = "https://cdn.freesound.org/mtg-jamendo/raw_30s/audio/{path}"

PUB_TAG_JAMENDO: dict[str, list[str]] = {
    "pub_piano": ["instrument---piano"],
    "pub_guitar": ["instrument---guitar"],
    "pub_vocal": ["instrument---voice"],
    "pub_relaxing": ["mood/theme---relaxing"],
    "pub_epic": ["mood/theme---epic"],
}

CHUNK_SIZE = 512 * 1024


def _read_split_tsv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for parts in reader:
            if len(parts) < 6:
                continue
            tags = [t.strip() for t in parts[5:] if t.strip()]
            rows.append(
                {
                    "track_id": parts[0].strip(),
                    "path": parts[3].strip().replace("\\", "/"),
                    "duration": float(parts[4]) if parts[4] else 0.0,
                    "tags": tags,
                }
            )
    return rows


def _load_test_pool(split_dir: Path) -> dict[str, dict[str, Any]]:
    """Merge instrument + moodtheme test rows by track_id."""
    paths = [
        split_dir / "autotagging_instrument-test.tsv",
        split_dir / "autotagging_moodtheme-test.tsv",
    ]
    by_id: dict[str, dict[str, Any]] = {}
    for p in paths:
        for row in _read_split_tsv(p):
            tid = row["track_id"]
            if tid not in by_id:
                by_id[tid] = {
                    "track_id": tid,
                    "path": row["path"],
                    "duration": row["duration"],
                    "tags": set(),
                }
            by_id[tid]["tags"].update(row["tags"])
    for v in by_id.values():
        v["tags"] = sorted(v["tags"])
    return by_id


def _active_pub_tags(jamendo_tags: list[str]) -> dict[str, bool]:
    tag_set = set(jamendo_tags)
    out: dict[str, bool] = {}
    for pub_id, needles in PUB_TAG_JAMENDO.items():
        out[pub_id] = any(n in tag_set for n in needles)
    return out


def build_manifest(
    *,
    split_dir: Path,
    max_per_tag: int,
    seed: int,
) -> list[dict[str, Any]]:
    pool = _load_test_pool(split_dir)
    per_tag_ids: dict[str, list[str]] = {k: [] for k in PUB_TAG_JAMENDO}
    for tid, row in pool.items():
        active = _active_pub_tags(row["tags"])
        for pub_id, is_on in active.items():
            if is_on:
                per_tag_ids[pub_id].append(tid)

    rng = random.Random(seed)
    selected: set[str] = set()
    for pub_id, ids in per_tag_ids.items():
        rng.shuffle(ids)
        selected.update(ids[:max_per_tag])

    manifest: list[dict[str, Any]] = []
    for tid in sorted(selected):
        row = pool[tid]
        manifest.append(
            {
                "track_id": tid,
                "path": row["path"],
                "duration": row["duration"],
                "jamendo_tags": row["tags"],
                **{f"gold_{k}": int(v) for k, v in _active_pub_tags(row["tags"]).items()},
            }
        )
    return manifest


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _download_file(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return True
    try:
        with requests.get(url, stream=True, timeout=120) as res:
            res.raise_for_status()
            with dest.open("wb") as out:
                for chunk in res.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        out.write(chunk)
        return True
    except Exception as exc:
        if dest.is_file():
            dest.unlink(missing_ok=True)
        print(f"warning: failed {url}: {exc}", flush=True)
        return False


def download_audio(
    manifest: list[dict[str, Any]],
    audio_dir: Path,
) -> dict[str, int]:
    stats = {"ok": 0, "skip": 0, "fail": 0}
    for row in tqdm(manifest, desc="Jamendo five-tag audio"):
        rel = row["path"]
        dest = audio_dir / rel
        if dest.is_file() and dest.stat().st_size > 0:
            stats["skip"] += 1
            row["audio_path"] = str(dest.resolve())
            continue
        url = CDN_AUDIO_URL.format(path=rel)
        if _download_file(url, dest):
            stats["ok"] += 1
            row["audio_path"] = str(dest.resolve())
        else:
            stats["fail"] += 1
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Download capped MTG-Jamendo audio for five public eval tags.")
    parser.add_argument(
        "--jamendo-repo",
        type=Path,
        default=JAMENDO_REPO,
        help="Path to cloned mtg-jamendo-dataset repo.",
    )
    parser.add_argument(
        "--max-per-tag",
        type=int,
        default=60,
        help="Max tracks per pub tag before union (default 60, ~300 total cap).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=DEFAULT_AUDIO_DIR,
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Write manifest only; do not download audio.",
    )
    args = parser.parse_args()

    jamendo_repo = args.jamendo_repo.resolve()
    split_dir = jamendo_repo / "data" / "splits" / "split-0"

    if not split_dir.is_dir():
        print(f"ERROR: missing {split_dir}. Clone mtg-jamendo-dataset first.", file=sys.stderr)
        return 1

    manifest = build_manifest(
        split_dir=split_dir,
        max_per_tag=args.max_per_tag,
        seed=args.seed,
    )
    _write_jsonl(args.manifest_out.resolve(), manifest)

    counts = {k: sum(1 for r in manifest if r.get(f"gold_{k}")) for k in PUB_TAG_JAMENDO}
    summary = {
        "manifest": str(args.manifest_out.resolve()),
        "n_tracks": len(manifest),
        "max_per_tag": args.max_per_tag,
        "positives_per_tag": counts,
        "audio_dir": str(args.audio_dir.resolve()),
    }
    summary_path = args.manifest_out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)

    if args.manifest_only:
        return 0

    stats = download_audio(manifest, args.audio_dir.resolve())
    _write_jsonl(args.manifest_out.resolve(), manifest)
    print(f"download stats: {stats}", flush=True)
    return 0 if stats["fail"] == 0 or stats["ok"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
