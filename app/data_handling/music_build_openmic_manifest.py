"""
Build capped OpenMIC-2018 manifest for public post-train retrieval test.

Reads long-format ``openmic-2018-aggregated-labels.csv`` (sample_key, instrument, relevance).
Maps:
  piano -> instrument ``piano`` (relevance >= threshold)
  vocal -> instrument ``voice``
  relaxing -> not applicable (always 0; eval uses piano + vocal only)

Example:
  python -m app.data_handling.music_build_openmic_manifest
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import settings

DEFAULT_OPENMIC_DIR = settings.DATA_DIR / "public_eval" / "openmic"
DEFAULT_MANIFEST = settings.DATA_DIR / "eval" / "openmic_manifest.jsonl"
LABELS_NAME = "openmic-2018-aggregated-labels.csv"
AUDIO_SUBDIR = "audio"


def resolve_openmic_audio(openmic_dir: Path, sample_key: str) -> Path | None:
    key = sample_key.strip()
    if not key:
        return None
    prefix = key[:3]
    root = openmic_dir / "openmic-2018"
    for ext in (".ogg", ".wav", ".mp3"):
        cand = root / AUDIO_SUBDIR / prefix / f"{key}{ext}"
        if cand.is_file() and cand.stat().st_size > 0:
            return cand.resolve()
    return None


def _load_label_rows(
    labels_path: Path,
    *,
    label_threshold: float,
) -> dict[str, dict[str, bool]]:
    """sample_key -> {piano: bool, voice: bool}."""
    flags: dict[str, dict[str, bool]] = defaultdict(
        lambda: {"piano": False, "voice": False}
    )
    with labels_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Empty labels file: {labels_path}")
        cols = {c.lower(): c for c in reader.fieldnames}
        if "sample_key" not in cols or "instrument" not in cols or "relevance" not in cols:
            raise ValueError(
                f"Expected sample_key, instrument, relevance in {labels_path}; "
                f"got {reader.fieldnames}"
            )
        sk_col = cols["sample_key"]
        inst_col = cols["instrument"]
        rel_col = cols["relevance"]
        for row in reader:
            key = (row.get(sk_col) or "").strip()
            inst = (row.get(inst_col) or "").strip().lower()
            if not key:
                continue
            try:
                rel = float(row.get(rel_col) or 0)
            except (TypeError, ValueError):
                rel = 0.0
            if rel < label_threshold:
                continue
            if inst == "piano":
                flags[key]["piano"] = True
            elif inst == "voice":
                flags[key]["voice"] = True
    return flags


def build_openmic_manifest(
    *,
    openmic_dir: Path,
    max_per_tag: int,
    seed: int,
    label_threshold: float,
) -> list[dict[str, Any]]:
    root = openmic_dir / "openmic-2018"
    labels_path = root / LABELS_NAME
    if not labels_path.is_file():
        raise FileNotFoundError(
            f"Missing {labels_path}; download OpenMIC: "
            "SKIP_JAMENDO=1 SKIP_MTAT=1 bash scripts/download_public_eval.sh"
        )

    flags = _load_label_rows(labels_path, label_threshold=label_threshold)
    piano_keys = [k for k, v in flags.items() if v["piano"]]
    voice_keys = [k for k, v in flags.items() if v["voice"]]

    rng = random.Random(seed)
    selected: set[str] = set()
    for pool in (piano_keys, voice_keys):
        rng.shuffle(pool)
        selected.update(pool[:max_per_tag])

    manifest: list[dict[str, Any]] = []
    for key in sorted(selected):
        f = flags[key]
        audio = resolve_openmic_audio(openmic_dir, key)
        entry: dict[str, Any] = {
            "sample_key": key,
            "gold_pub_piano": int(f["piano"]),
            "gold_pub_vocal": int(f["voice"]),
            "gold_pub_relaxing": 0,
            "dataset": "openmic",
        }
        if audio is not None:
            entry["audio_path"] = str(audio)
        manifest.append(entry)

    return manifest


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build OpenMIC public test manifest.")
    parser.add_argument("--openmic-dir", type=Path, default=DEFAULT_OPENMIC_DIR)
    parser.add_argument("--max-per-tag", type=int, default=60)
    parser.add_argument("--label-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    manifest = build_openmic_manifest(
        openmic_dir=args.openmic_dir.resolve(),
        max_per_tag=args.max_per_tag,
        seed=args.seed,
        label_threshold=args.label_threshold,
    )
    n_audio = sum(1 for r in manifest if r.get("audio_path"))
    summary = {
        "manifest": str(args.manifest_out.resolve()),
        "n_rows": len(manifest),
        "n_with_audio": n_audio,
        "max_per_tag": args.max_per_tag,
        "label_threshold": args.label_threshold,
        "note": "mood_relaxing not applicable; eval uses piano + vocal only",
    }
    _write_jsonl(args.manifest_out.resolve(), manifest)
    summary_path = args.manifest_out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if n_audio == 0:
        print("warning: no local .ogg files; extract OpenMIC tarball first", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
