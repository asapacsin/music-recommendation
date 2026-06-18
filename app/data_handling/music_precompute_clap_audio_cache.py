"""
One-time (per backbone) precompute of CLAP HTSAT backbone features for JSONL manifests.

Usage:
  python -m app.data_handling.music_precompute_clap_audio_cache \\
    --jsonl data/mapping/clap_train_15s.jsonl \\
    --jsonl data/mapping/clap_val_15s.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import settings

from app.clap_audio_cache import (
    ClapAudioBackboneCache,
    default_backbone_cache_dir,
    get_backbone_audio_features_from_filelist,
    normalize_audio_manifest_key,
)
from app.init_model import _load_clap_train_jsonl, load_original_model


def _collect_rows(jsonl_paths: list[Path]) -> dict[str, str]:
    """Map cache_key -> absolute audio path (first manifest wins)."""
    by_key: dict[str, str] = {}
    for jsonl_path in jsonl_paths:
        paths, _texts, keys = _load_clap_train_jsonl(jsonl_path.resolve(), with_cache_keys=True)
        for abs_path, key in zip(paths, keys, strict=True):
            by_key.setdefault(key, abs_path)
    return by_key


def ensure_clap_audio_cache(
    *,
    jsonl_paths: list[Path],
    cache_dir: Path | None = None,
    batch_size: int = 32,
) -> Path:
    """
    One-time MP3 decode + backbone encode; reuse on every fine-tune epoch/seed.

    Builds only missing keys. Sets ``RAGWEB_CLAP_AUDIO_CACHE`` for child processes.
    """
    import os

    resolved_jsonls = [p.resolve() for p in jsonl_paths if p.is_file()]
    if not resolved_jsonls:
        raise FileNotFoundError(
            "No JSONL manifests found for audio cache "
            f"(checked: {[str(p) for p in jsonl_paths]})"
        )

    out_dir = (cache_dir or default_backbone_cache_dir()).resolve()
    by_key = _collect_rows(resolved_jsonls)
    manifest_keys = list(by_key.keys())
    cache = ClapAudioBackboneCache(out_dir)
    present, total = cache.keys_for_manifest(manifest_keys)

    if present < total:
        print(
            f"CLAP audio cache incomplete ({present}/{total}); "
            f"precomputing missing clips to {out_dir} ...",
            flush=True,
        )
        stats = precompute_cache(
            jsonl_paths=resolved_jsonls,
            cache_dir=out_dir,
            batch_size=batch_size,
            skip_existing=True,
        )
        present = int(stats["cached_keys"])
        total = int(stats["manifest_keys"])
    else:
        print(
            f"Using CLAP audio cache: {out_dir} ({total} clips, no MP3 re-decode)",
            flush=True,
        )

    if present < total:
        raise RuntimeError(
            f"CLAP audio cache still incomplete after precompute: {present}/{total} at {out_dir}"
        )

    os.environ["RAGWEB_CLAP_AUDIO_CACHE"] = str(out_dir)
    return out_dir


def precompute_cache(
    *,
    jsonl_paths: list[Path],
    cache_dir: Path,
    batch_size: int = 32,
    skip_existing: bool = True,
) -> dict[str, Any]:
    cache = ClapAudioBackboneCache(cache_dir)
    by_key = _collect_rows(jsonl_paths)

    todo: list[tuple[str, str]] = []
    for key, abs_path in sorted(by_key.items()):
        if skip_existing and cache.has_key(key):
            continue
        todo.append((key, abs_path))

    model = load_original_model()
    if hasattr(model, "to"):
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

    n_written = 0
    for start in tqdm(range(0, len(todo), batch_size), desc="Precompute backbone", unit="batch"):
        batch = todo[start : start + batch_size]
        if not batch:
            continue
        keys = [k for k, _ in batch]
        paths = [p for _, p in batch]
        feats = get_backbone_audio_features_from_filelist(model, paths, use_tensor=False)
        feats = np.asarray(feats, dtype=np.float32)
        if feats.ndim == 1:
            feats = feats.reshape(1, -1)
        for i, key in enumerate(keys):
            cache.write_entry(key, feats[i])
            n_written += 1

    cache.save_index()
    cache.write_meta()
    present, total = cache.keys_for_manifest(list(by_key.keys()))
    return {
        "cache_dir": str(cache_dir),
        "manifest_keys": total,
        "cached_keys": present,
        "newly_written": n_written,
        "jsonl_paths": [str(p) for p in jsonl_paths],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Precompute CLAP backbone audio features for CLAP JSONL manifests."
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        action="append",
        required=True,
        help="Train/val manifest (repeatable). Default train+val if omitted in wrapper scripts.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=default_backbone_cache_dir(),
        help="Output cache root (features/ + index.json).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-encode rows even when cache entry exists.",
    )
    args = parser.parse_args()

    stats = precompute_cache(
        jsonl_paths=args.jsonl,
        cache_dir=args.cache_dir.resolve(),
        batch_size=args.batch_size,
        skip_existing=not args.no_skip_existing,
    )
    print(json.dumps(stats, indent=2))
    if stats["cached_keys"] < stats["manifest_keys"]:
        raise SystemExit(
            f"Cache incomplete: {stats['cached_keys']}/{stats['manifest_keys']} keys"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
