"""
Build capped MagnaTagATune manifest for public post-train retrieval test.

Maps MTAT tag columns to thesis-aligned ``gold_pub_*`` labels:
  piano -> ``piano``
  vocal -> any of vocals, male vocal, female voice, singer, voice
  relaxing -> calm OR mellow (no ``relaxing`` column in MTAT)

Example:
  python -m app.data_handling.music_build_mtat_manifest --max-per-tag 60
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import settings

DEFAULT_MTAT_DIR = settings.DATA_DIR / "public_eval" / "magnatagatune"
DEFAULT_MANIFEST = settings.DATA_DIR / "eval" / "mtat_manifest.jsonl"
ANNOTATIONS = "annotations_final.csv"

MTAT_VOCAL_COLS = ("vocals", "male vocal", "female voice", "singer", "voice")
MTAT_RELAX_COLS = ("calm", "mellow")


def _tag_positive(row: dict[str, str], col: str) -> bool:
    v = row.get(col, "0")
    try:
        return int(float(v)) == 1
    except (ValueError, TypeError):
        return False


def _any_positive(row: dict[str, str], cols: tuple[str, ...]) -> bool:
    return any(_tag_positive(row, c) for c in cols)


def resolve_mtat_mp3(mtat_dir: Path, mp3_rel: str) -> Path | None:
    rel = mp3_rel.strip()
    if not rel:
        return None
    for sub in ("mp3", "mp3s", "input"):
        cand = mtat_dir / sub / rel
        if cand.is_file() and cand.stat().st_size > 0:
            return cand.resolve()
    cand = mtat_dir / rel
    if cand.is_file() and cand.stat().st_size > 0:
        return cand.resolve()
    return None


def build_mtat_manifest(
    *,
    mtat_dir: Path,
    max_per_tag: int,
    seed: int,
) -> list[dict[str, Any]]:
    ann_path = mtat_dir / ANNOTATIONS
    if not ann_path.is_file():
        raise FileNotFoundError(f"Missing {ann_path}; run scripts/download_public_eval.sh")

    rows_by_tag: dict[str, list[dict[str, str]]] = {
        "gold_pub_piano": [],
        "gold_pub_vocal": [],
        "gold_pub_relaxing": [],
    }

    with ann_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames or "clip_id" not in reader.fieldnames:
            raise ValueError(f"Unexpected TSV header in {ann_path}")
        if "mp3_path" not in reader.fieldnames:
            raise ValueError(f"Expected mp3_path column in {ann_path}")
        if "piano" not in reader.fieldnames:
            raise ValueError("Expected piano column in MTAT annotations")

        for row in reader:
            mp3_rel = row.get("mp3_path", "")
            if not mp3_rel:
                continue
            rec = dict(row)
            if _tag_positive(rec, "piano"):
                rows_by_tag["gold_pub_piano"].append(rec)
            if _any_positive(rec, MTAT_VOCAL_COLS):
                rows_by_tag["gold_pub_vocal"].append(rec)
            if _any_positive(rec, MTAT_RELAX_COLS):
                rows_by_tag["gold_pub_relaxing"].append(rec)

    rng = random.Random(seed)
    selected_ids: set[str] = set()
    for _col, pool in rows_by_tag.items():
        rng.shuffle(pool)
        for rec in pool[:max_per_tag]:
            selected_ids.add(rec["clip_id"])

    manifest: list[dict[str, Any]] = []
    with ann_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            cid = row.get("clip_id", "")
            if cid not in selected_ids:
                continue
            mp3_rel = row.get("mp3_path", "")
            audio = resolve_mtat_mp3(mtat_dir, mp3_rel)
            entry: dict[str, Any] = {
                "clip_id": cid,
                "mp3_path": mp3_rel,
                "gold_pub_piano": int(_tag_positive(row, "piano")),
                "gold_pub_vocal": int(_any_positive(row, MTAT_VOCAL_COLS)),
                "gold_pub_relaxing": int(_any_positive(row, MTAT_RELAX_COLS)),
                "dataset": "mtat",
            }
            if audio is not None:
                entry["audio_path"] = str(audio)
            manifest.append(entry)

    manifest.sort(key=lambda r: str(r.get("clip_id", "")))
    return manifest


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build MTAT public test manifest.")
    parser.add_argument("--mtat-dir", type=Path, default=DEFAULT_MTAT_DIR)
    parser.add_argument("--max-per-tag", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    mtat_dir = args.mtat_dir.resolve()
    manifest = build_mtat_manifest(
        mtat_dir=mtat_dir,
        max_per_tag=args.max_per_tag,
        seed=args.seed,
    )
    n_audio = sum(1 for r in manifest if r.get("audio_path"))
    summary = {
        "manifest": str(args.manifest_out.resolve()),
        "n_rows": len(manifest),
        "n_with_audio": n_audio,
        "max_per_tag": args.max_per_tag,
        "vocal_cols": list(MTAT_VOCAL_COLS),
        "relax_cols": list(MTAT_RELAX_COLS),
    }
    _write_jsonl(args.manifest_out.resolve(), manifest)
    summary_path = args.manifest_out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if n_audio == 0:
        print(
            "warning: no local MP3s found; extract mp3.zip.* under magnatagatune/",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
