"""
Build mixed-domain CLAP train JSONL: anime (tag-only) + MTAT + OpenMIC public clips.

Excludes eval holdout paths (MTAT/OpenMIC manifests). Jamendo is never included in training.
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

from app.data_handling.music_build_mtat_manifest import (
    MTAT_RELAX_COLS,
    MTAT_VOCAL_COLS,
    resolve_mtat_mp3,
)
from app.data_handling.music_build_openmic_manifest import (
    _load_label_rows,
    resolve_openmic_audio,
)
from app.data_handling.music_build_tag_train_jsonl import (
    DEFAULT_PRIMARY_TAGS,
    format_tag_text,
)
from app.self_train.jsonl_io import write_jsonl_rows

DEFAULT_ANIME_JSONL = settings.MAPPING_DIR / "clap_train_tag.jsonl"
DEFAULT_OUT_JSONL = settings.MAPPING_DIR / "clap_train_tag_mixed.jsonl"
DEFAULT_HOLDOUT_TXT = settings.MAPPING_DIR / "public_eval_holdout_paths.txt"
DEFAULT_MTAT_DIR = settings.DATA_DIR / "public_eval" / "magnatagatune"
DEFAULT_OPENMIC_DIR = settings.DATA_DIR / "public_eval" / "openmic"
MTAT_ANNOTATIONS = "annotations_final.csv"
OPENMIC_LABELS = "openmic-2018-aggregated-labels.csv"


def _norm_path(p: str) -> str:
    return str(Path(p).expanduser().resolve())


def load_holdout_paths(manifest_paths: list[Path]) -> set[str]:
    """Normalized absolute audio_path strings excluded from mixed training."""
    holdout: set[str] = set()
    for mp in manifest_paths:
        if not mp.is_file():
            raise FileNotFoundError(f"Holdout manifest not found: {mp}")
        with mp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                ap = row.get("audio_path")
                if isinstance(ap, str) and ap.strip():
                    holdout.add(_norm_path(ap))
    return holdout


def gold_pub_to_multihot(entry: dict[str, Any]) -> dict[str, int]:
    return {
        "inst_piano": int(entry.get("gold_pub_piano", 0) or 0),
        "inst_vocal": int(entry.get("gold_pub_vocal", 0) or 0),
        "mood_relaxing": int(entry.get("gold_pub_relaxing", 0) or 0),
    }


def _tag_buckets(multihot: dict[str, int]) -> list[str]:
    """Primary tag buckets for stratified sampling (order: piano, vocal, relaxing)."""
    buckets: list[str] = []
    if multihot.get("inst_piano", 0) == 1:
        buckets.append("inst_piano")
    if multihot.get("inst_vocal", 0) == 1:
        buckets.append("inst_vocal")
    if multihot.get("mood_relaxing", 0) == 1:
        buckets.append("mood_relaxing")
    return buckets or ["none"]


def _mtat_tag_positive(row: dict[str, str], col: str) -> bool:
    v = row.get(col, "0")
    try:
        return int(float(v)) == 1
    except (ValueError, TypeError):
        return False


def _mtat_any_positive(row: dict[str, str], cols: tuple[str, ...]) -> bool:
    return any(_mtat_tag_positive(row, c) for c in cols)


def build_mtat_train_candidates(
    *,
    mtat_dir: Path,
    holdout: set[str],
) -> list[dict[str, Any]]:
    ann_path = mtat_dir / MTAT_ANNOTATIONS
    if not ann_path.is_file():
        raise FileNotFoundError(f"Missing MTAT annotations: {ann_path}")

    candidates: list[dict[str, Any]] = []
    with ann_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            mp3_rel = row.get("mp3_path", "")
            if not mp3_rel:
                continue
            audio = resolve_mtat_mp3(mtat_dir, mp3_rel)
            if audio is None:
                continue
            ap = _norm_path(str(audio))
            if ap in holdout:
                continue
            gold = {
                "gold_pub_piano": int(_mtat_tag_positive(row, "piano")),
                "gold_pub_vocal": int(_mtat_any_positive(row, MTAT_VOCAL_COLS)),
                "gold_pub_relaxing": int(_mtat_any_positive(row, MTAT_RELAX_COLS)),
            }
            if not any(gold.values()):
                continue
            mh = gold_pub_to_multihot(gold)
            text, text_source = format_tag_text(
                mh, primary_tags=DEFAULT_PRIMARY_TAGS, fallback_text="music"
            )
            candidates.append(
                {
                    "audio_path": ap,
                    "text": text,
                    "text_source": text_source,
                    "domain": "mtat",
                    "clip_id": row.get("clip_id", ""),
                    "gold_pub_piano": gold["gold_pub_piano"],
                    "gold_pub_vocal": gold["gold_pub_vocal"],
                    "gold_pub_relaxing": gold["gold_pub_relaxing"],
                }
            )
    return candidates


def build_openmic_train_candidates(
    *,
    openmic_dir: Path,
    holdout: set[str],
    label_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    labels_path = openmic_dir / "openmic-2018" / OPENMIC_LABELS
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing OpenMIC labels: {labels_path}")

    flags = _load_label_rows(labels_path, label_threshold=label_threshold)
    candidates: list[dict[str, Any]] = []
    for key, f in flags.items():
        if not f["piano"] and not f["voice"]:
            continue
        audio = resolve_openmic_audio(openmic_dir, key)
        if audio is None:
            continue
        ap = _norm_path(str(audio))
        if ap in holdout:
            continue
        gold = {
            "gold_pub_piano": int(f["piano"]),
            "gold_pub_vocal": int(f["voice"]),
            "gold_pub_relaxing": 0,
        }
        mh = gold_pub_to_multihot(gold)
        text, text_source = format_tag_text(
            mh, primary_tags=DEFAULT_PRIMARY_TAGS, fallback_text="music"
        )
        candidates.append(
            {
                "audio_path": ap,
                "text": text,
                "text_source": text_source,
                "domain": "openmic",
                "sample_key": key,
                **gold,
            }
        )
    return candidates


def sample_public_rows(
    candidates: list[dict[str, Any]],
    *,
    target: int,
    seed: int,
) -> list[dict[str, Any]]:
    if target <= 0 or not candidates:
        return []
    if len(candidates) <= target:
        return list(candidates)

    rng = random.Random(seed)
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in candidates:
        mh = gold_pub_to_multihot(rec)
        for bucket in _tag_buckets(mh):
            by_bucket[bucket].append(rec)

    per_bucket = max(1, target // max(1, len(by_bucket)))
    selected: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    for bucket in sorted(by_bucket.keys()):
        pool = by_bucket[bucket][:]
        rng.shuffle(pool)
        n = 0
        for rec in pool:
            ap = rec["audio_path"]
            if ap in seen_paths:
                continue
            selected.append(rec)
            seen_paths.add(ap)
            n += 1
            if n >= per_bucket:
                break

    if len(selected) < target:
        remaining = [c for c in candidates if c["audio_path"] not in seen_paths]
        rng.shuffle(remaining)
        for rec in remaining:
            if len(selected) >= target:
                break
            selected.append(rec)
            seen_paths.add(rec["audio_path"])

    rng.shuffle(selected)
    return selected[:target]


def load_anime_rows(anime_jsonl: Path) -> list[dict[str, Any]]:
    if not anime_jsonl.is_file():
        raise FileNotFoundError(f"Anime JSONL not found: {anime_jsonl}")
    rows: list[dict[str, Any]] = []
    with anime_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            out = dict(row)
            out["domain"] = "anime"
            rows.append(out)
    return rows


def compute_public_target(
    n_anime: int,
    *,
    public_clip_target: int | None,
    mix_ratio: float | None,
) -> int:
    if public_clip_target is not None and public_clip_target > 0:
        return public_clip_target
    if mix_ratio is not None and 0.0 < mix_ratio < 1.0:
        return int(n_anime * mix_ratio / (1.0 - mix_ratio))
    return n_anime


def build_mixed_domain_train_jsonl(
    *,
    anime_jsonl: Path,
    holdout_manifests: list[Path],
    out_jsonl: Path,
    holdout_txt: Path,
    mtat_dir: Path,
    openmic_dir: Path,
    public_clip_target: int | None = None,
    mix_ratio: float | None = 0.5,
    seed: int = 42,
    label_threshold: float = 0.5,
) -> dict[str, Any]:
    holdout = load_holdout_paths(holdout_manifests)
    holdout_txt.parent.mkdir(parents=True, exist_ok=True)
    holdout_txt.write_text("\n".join(sorted(holdout)) + "\n", encoding="utf-8")

    anime_rows = load_anime_rows(anime_jsonl)
    mtat_pool = build_mtat_train_candidates(mtat_dir=mtat_dir, holdout=holdout)
    openmic_pool = build_openmic_train_candidates(
        openmic_dir=openmic_dir,
        holdout=holdout,
        label_threshold=label_threshold,
    )
    public_pool = mtat_pool + openmic_pool
    pub_target = compute_public_target(
        len(anime_rows),
        public_clip_target=public_clip_target,
        mix_ratio=mix_ratio,
    )
    pub_target_requested = pub_target
    pub_target = min(pub_target, len(public_pool))
    public_rows = sample_public_rows(public_pool, target=pub_target, seed=seed)

    combined = anime_rows + public_rows
    rng = random.Random(seed + 1)
    rng.shuffle(combined)

    write_jsonl_rows(out_jsonl, combined)

    domain_counts: dict[str, int] = defaultdict(int)
    for r in combined:
        domain_counts[str(r.get("domain", "?"))] += 1

    return {
        "out_jsonl": str(out_jsonl.resolve()),
        "holdout_txt": str(holdout_txt.resolve()),
        "n_anime": len(anime_rows),
        "n_public_sampled": len(public_rows),
        "n_total": len(combined),
        "n_mtat_candidates": len(mtat_pool),
        "n_openmic_candidates": len(openmic_pool),
        "n_holdout_paths": len(holdout),
        "public_clip_target_requested": pub_target_requested,
        "public_clip_target": pub_target,
        "mix_ratio": mix_ratio,
        "domain_counts": dict(domain_counts),
        "seed": seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build mixed-domain tag-only train JSONL (anime + MTAT + OpenMIC)."
    )
    parser.add_argument("--anime-jsonl", type=Path, default=DEFAULT_ANIME_JSONL)
    parser.add_argument(
        "--holdout-manifests",
        type=Path,
        nargs="+",
        default=[
            settings.DATA_DIR / "eval" / "mtat_manifest.jsonl",
            settings.DATA_DIR / "eval" / "openmic_manifest.jsonl",
        ],
    )
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT_JSONL)
    parser.add_argument("--holdout-txt", type=Path, default=DEFAULT_HOLDOUT_TXT)
    parser.add_argument("--mtat-dir", type=Path, default=DEFAULT_MTAT_DIR)
    parser.add_argument("--openmic-dir", type=Path, default=DEFAULT_OPENMIC_DIR)
    parser.add_argument(
        "--public-clip-target",
        type=int,
        default=0,
        help="Explicit public row count (0 = use mix-ratio).",
    )
    parser.add_argument(
        "--mix-ratio",
        type=float,
        default=0.5,
        help="Public fraction when target unset (0.5 = 50/50 anime/public rows).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-threshold", type=float, default=0.5)
    args = parser.parse_args()

    pub_target = args.public_clip_target if args.public_clip_target > 0 else None
    mix_ratio = args.mix_ratio if pub_target is None else None

    summary = build_mixed_domain_train_jsonl(
        anime_jsonl=args.anime_jsonl.resolve(),
        holdout_manifests=[p.resolve() for p in args.holdout_manifests],
        out_jsonl=args.out_jsonl.resolve(),
        holdout_txt=args.holdout_txt.resolve(),
        mtat_dir=args.mtat_dir.resolve(),
        openmic_dir=args.openmic_dir.resolve(),
        public_clip_target=pub_target,
        mix_ratio=mix_ratio,
        seed=args.seed,
        label_threshold=args.label_threshold,
    )
    summary_path = args.out_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
