"""
Multi-seed CLAP fine-tune driver (calls ``init_model.model_creation`` per seed).

Writes ``data/log/finetune_runs/<run_id>/seed_<n>/`` with ``best_model.pt``,
``params.json``, ``metrics.jsonl``, and a run-level ``summary.json`` (mean/std of
best train-time similarity — extend with retrieval metrics as needed).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import settings

from app.init_model import model_creation


def _default_params() -> dict[str, Any]:
    return {
        "learning_rate": 1e-4,
        "num_epochs": 5,
        "batch_size": 128,
        "temperature": 100,
        "unfreeze_layers": {
            "audio_projection": True,
            "audio_transform": True,
            "text_projection": True,
            "text_transform": True,
        },
        "early_stopping": {
            "enabled": True,
            "metric": "similarity",
            "mode": "max",
        },
        "train_jsonl": str(settings.CLAP_TRAIN_JSONL),
    }


def _parse_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    return [args.base_seed + i for i in range(args.n_seeds)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CLAP fine-tune for multiple RNG seeds.")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Subfolder under data/log/finetune_runs (default: UTC timestamp).",
    )
    parser.add_argument("--base-seed", type=int, default=42, help="First seed when using --n-seeds.")
    parser.add_argument("--n-seeds", type=int, default=3, help="Number of consecutive seeds from base-seed.")
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds (overrides --base-seed and --n-seeds).",
    )
    parser.add_argument(
        "--params-json",
        type=Path,
        default=None,
        help="JSON object merged on top of defaults (same keys as app/main.py params).",
    )
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        default=None,
        help=f"15s train manifest (default: {settings.CLAP_TRAIN_JSONL}).",
    )
    parser.add_argument(
        "--use-music-db-fallback",
        action="store_true",
        help="Train on data/music_db/* instead of clap_train_15s.jsonl.",
    )
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = settings.LOG_DIR / "finetune_runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    base = _default_params()
    if args.train_jsonl is not None:
        base["train_jsonl"] = str(args.train_jsonl)
    if args.use_music_db_fallback:
        base["use_music_db_fallback"] = True
    if args.params_json is not None:
        if not args.params_json.is_file():
            raise FileNotFoundError(f"--params-json not found: {args.params_json}")
        extra = json.loads(args.params_json.read_text(encoding="utf-8"))
        if not isinstance(extra, dict):
            raise ValueError("--params-json must be a JSON object")
        base.update(extra)

    seeds = _parse_seeds(args)
    if not seeds:
        raise ValueError("No seeds to run.")

    per_seed: list[dict[str, Any]] = []
    sims: list[float] = []

    for seed in seeds:
        seed_dir = run_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        save_path = seed_dir / "best_model.pt"
        params_path = seed_dir / "params.json"
        merged = {**base, "seed": seed, "save_path": str(save_path)}
        params_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

        out = model_creation(merged)
        per_seed.append(out)
        sims.append(float(out["best_similarity"]))
        if not math.isfinite(sims[-1]):
            print(f"warning: seed {seed} has non-finite best_similarity", file=sys.stderr)

    finite = [s for s in sims if math.isfinite(s)]
    summary: dict[str, Any] = {
        "run_id": run_id,
        "run_root": str(run_root),
        "seeds": seeds,
        "per_seed": per_seed,
        "best_similarity_mean": statistics.mean(finite) if finite else None,
        "best_similarity_stdev": statistics.stdev(finite) if len(finite) > 1 else 0.0,
        "n_seeds_finite": len(finite),
    }
    summary_path = run_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "best_similarity_mean": summary["best_similarity_mean"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
