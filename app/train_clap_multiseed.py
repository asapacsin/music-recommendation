"""
Multi-seed CLAP fine-tune driver (calls ``init_model.model_creation`` per seed).

Checkpoints: ``model/clap/finetune/<run_id>/seed_<n>/best_model.pt``
Logs: ``data/log/finetune_runs/<run_id>/seed_<n>/`` (params.json, metrics.jsonl)
Run summary: ``data/log/finetune_runs/<run_id>/summary.json``
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

import torch

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


def _resolve_checkpoint_path(model_seed_dir: Path, log_seed_dir: Path) -> Path | None:
    for candidate in (
        model_seed_dir / "best_model.pt",
        log_seed_dir / "best_model.pt",
    ):
        if candidate.is_file():
            return candidate
    return None


def _seed_training_complete(log_seed_dir: Path) -> bool:
    if (log_seed_dir / "training_complete.json").is_file():
        return True
    metrics_path = log_seed_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        return False
    lines = [ln for ln in metrics_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return False
    last = json.loads(lines[-1])
    epoch = int(last.get("epoch", 0))
    # Legacy runs before training_complete.json (val early-stop, min_epochs=5).
    return epoch >= 5 and last.get("val_similarity") is not None


def _load_skipped_seed_result(
    *,
    seed: int,
    save_path: Path,
    metrics_path: Path,
) -> dict[str, Any]:
    """Rebuild per-seed summary from an existing checkpoint (resume after Slurm timeout)."""
    best_metric: float | None = None
    best_epoch = 0
    best_train: float | None = None
    best_val: float | None = None
    monitor = "val_similarity"

    if metrics_path.is_file():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cm = rec.get("checkpoint_metric")
            if cm is None:
                cm = rec.get("val_similarity")
            if cm is None:
                cm = rec.get("train_similarity")
            if cm is None:
                continue
            cm_f = float(cm)
            if best_metric is None or cm_f > best_metric:
                best_metric = cm_f
                best_epoch = int(rec.get("epoch", 0))
                ts = rec.get("train_similarity")
                best_train = float(ts) if ts is not None else None
                vs = rec.get("val_similarity")
                best_val = float(vs) if vs is not None else None
                monitor = str(rec.get("monitor", monitor))

    if best_metric is None and save_path.is_file():
        ckpt = torch.load(str(save_path), map_location="cpu")
        if isinstance(ckpt, dict):
            cm = ckpt.get("checkpoint_metric")
            if cm is None:
                cm = ckpt.get("val_similarity")
            if cm is None:
                cm = ckpt.get("train_similarity")
            if cm is not None:
                best_metric = float(cm)
                best_epoch = int(ckpt.get("epoch", 0)) + 1
                monitor = str(ckpt.get("monitor", monitor))
                vs = ckpt.get("val_similarity")
                best_val = float(vs) if vs is not None else None
                ts = ckpt.get("train_similarity")
                best_train = float(ts) if ts is not None else None

    if best_metric is None:
        raise RuntimeError(
            f"seed {seed}: --skip-existing set but could not read metrics from {metrics_path}"
        )

    print(
        f"Skipping seed {seed}: checkpoint exists at {save_path} "
        f"(best_epoch={best_epoch}, metric={best_metric:.4f})",
        flush=True,
    )
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_similarity": best_metric,
        "best_train_similarity": best_train if best_train is not None else float("-inf"),
        "best_val_similarity": best_val,
        "monitor": monitor,
        "save_path": str(save_path),
        "metrics_path": str(metrics_path),
        "skipped": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CLAP fine-tune for multiple RNG seeds.")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Experiment id (subfolder under model/clap/finetune/ and data/log/finetune_runs/).",
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
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip seeds that already have best_model.pt (default: true).",
    )
    parser.add_argument(
        "--no-audio-cache",
        action="store_true",
        help="Disable backbone audio cache (re-decode MP3 every batch; slow).",
    )
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_run_root = settings.finetune_log_run_dir(run_id)
    model_run_root = settings.FINETUNE_MODEL_DIR / run_id
    log_run_root.mkdir(parents=True, exist_ok=True)
    model_run_root.mkdir(parents=True, exist_ok=True)

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

    if not args.no_audio_cache and not base.get("use_music_db_fallback"):
        from app.data_handling.music_precompute_clap_audio_cache import ensure_clap_audio_cache

        cache_jsonls: list[Path] = []
        train_path = Path(base.get("train_jsonl", settings.CLAP_TRAIN_JSONL))
        if train_path.is_file():
            cache_jsonls.append(train_path.resolve())
        val_raw = base.get("val_jsonl")
        if val_raw:
            val_path = Path(val_raw)
            if val_path.is_file():
                cache_jsonls.append(val_path.resolve())
        if cache_jsonls:
            explicit_cache = base.get("audio_cache_dir")
            cache_dir = ensure_clap_audio_cache(
                jsonl_paths=cache_jsonls,
                cache_dir=Path(explicit_cache) if explicit_cache else None,
            )
            base["audio_cache_dir"] = str(cache_dir)

    per_seed: list[dict[str, Any]] = []
    sims: list[float] = []

    for seed in seeds:
        log_seed_dir = settings.finetune_log_seed_dir(run_id, seed)
        model_seed_dir = model_run_root / f"seed_{seed}"
        log_seed_dir.mkdir(parents=True, exist_ok=True)
        model_seed_dir.mkdir(parents=True, exist_ok=True)

        save_path = model_seed_dir / "best_model.pt"
        params_path = log_seed_dir / "params.json"
        metrics_path = log_seed_dir / "metrics.jsonl"

        if (
            args.skip_existing
            and _seed_training_complete(log_seed_dir)
            and _resolve_checkpoint_path(model_seed_dir, log_seed_dir) is not None
        ):
            ckpt_path = _resolve_checkpoint_path(model_seed_dir, log_seed_dir)
            assert ckpt_path is not None
            out = _load_skipped_seed_result(
                seed=seed,
                save_path=ckpt_path,
                metrics_path=metrics_path,
            )
            per_seed.append(out)
            sims.append(float(out["best_similarity"]))
            continue

        merged = {
            **base,
            "seed": seed,
            "save_path": str(save_path),
            "metrics_path": str(metrics_path),
        }
        params_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

        out = model_creation(merged)
        per_seed.append(out)
        sims.append(float(out["best_similarity"]))
        if not math.isfinite(sims[-1]):
            print(f"warning: seed {seed} has non-finite best_similarity", file=sys.stderr)

    finite = [s for s in sims if math.isfinite(s)]
    summary: dict[str, Any] = {
        "run_id": run_id,
        "log_run_root": str(log_run_root),
        "model_run_root": str(model_run_root),
        "seeds": seeds,
        "per_seed": per_seed,
        "best_similarity_mean": statistics.mean(finite) if finite else None,
        "best_similarity_stdev": statistics.stdev(finite) if len(finite) > 1 else 0.0,
        "n_seeds_finite": len(finite),
    }
    summary_path = log_run_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "model_run_root": str(model_run_root),
                "best_similarity_mean": summary["best_similarity_mean"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
