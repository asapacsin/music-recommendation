"""
CLAP iterative self-training driver (local 15s JSONL).

Mine hard pairs → optional LLM refine + gate → mixed JSONL → fine-tune → val eval.

Example::

    python -m app.train_clap_self_loop --run-id thesis_self_v2 --n-iters 2 --refine
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import settings

from app.init_model import model_creation
from app.self_train.eval_iter import eval_val_mean_similarity, run_gold_retrieval_eval, write_iter_metrics
from app.self_train.gate import default_gate_params
from app.self_train.manifest import build_mixed_manifest
from app.self_train.mine import mine_hard_pairs
from app.self_train.refine import get_refiner, refine_hard_manifest


def _default_train_params() -> dict[str, Any]:
    return {
        "learning_rate": 1e-4,
        "num_epochs": 20,
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
            "monitor": "val_similarity",
            "mode": "max",
            "patience": 2,
            "min_epochs": 5,
        },
    }


def run_iteration(
    *,
    run_id: str,
    iter_n: int,
    train_jsonl: Path,
    val_jsonl: Path,
    hard_frac: float,
    seed: int,
    no_refine: bool,
    train_params: dict[str, Any],
    max_samples: int | None,
    embed_batch_size: int,
    run_gold_eval: bool,
    gate_params: dict[str, float] | None,
    llm_params: dict[str, Any] | None,
    refine_max_hard: int | None,
) -> dict[str, Any]:
    data_dir = settings.self_train_iter_data_dir(run_id, iter_n)
    log_dir = settings.self_train_log_iter_dir(run_id, iter_n)
    data_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    init_checkpoint: Path | None = None
    if iter_n > 0:
        prev = settings.self_train_checkpoint_path(run_id, iter_n - 1)
        if not prev.is_file():
            raise FileNotFoundError(
                f"Previous iter checkpoint required for iter {iter_n}: {prev}"
            )
        init_checkpoint = prev

    hard_path = data_dir / "hard_mined.jsonl"
    mine_summary = mine_hard_pairs(
        train_jsonl=train_jsonl,
        out_path=hard_path,
        init_checkpoint=init_checkpoint,
        hard_frac=hard_frac,
        batch_size=embed_batch_size,
        max_samples=max_samples,
    )

    refined_path = data_dir / "refined.jsonl"
    refiner = None
    try:
        if no_refine:
            refiner = get_refiner("noop")
            refine_summary = refine_hard_manifest(
                hard_jsonl=hard_path,
                out_path=refined_path,
                refiner=refiner,
                iter_n=iter_n,
                text_source="grok",
                refine_max_hard=refine_max_hard,
            )
        else:
            refiner = get_refiner(
                "llm",
                init_checkpoint=init_checkpoint,
                gate_params=gate_params,
                llm_params=llm_params,
            )
            refine_summary = refine_hard_manifest(
                hard_jsonl=hard_path,
                out_path=refined_path,
                refiner=refiner,
                iter_n=iter_n,
                text_source="llm_refined",
                refine_max_hard=refine_max_hard,
            )
    finally:
        if refiner is not None:
            refiner.close()

    mixed_path = data_dir / "train_mixed.jsonl"
    manifest_summary = build_mixed_manifest(
        train_jsonl=train_jsonl,
        hard_jsonl=hard_path,
        refined_jsonl=refined_path,
        out_path=mixed_path,
        iter_n=iter_n,
        max_samples=max_samples,
    )

    ckpt_path = settings.self_train_checkpoint_path(run_id, iter_n)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / "metrics.jsonl"
    merged_params = {
        **train_params,
        "seed": seed,
        "train_jsonl": str(mixed_path),
        "val_jsonl": str(val_jsonl),
        "save_path": str(ckpt_path),
        "metrics_path": str(metrics_path),
    }
    if init_checkpoint is not None:
        merged_params["init_checkpoint"] = str(init_checkpoint)

    params_path = log_dir / "params.json"
    params_path.write_text(
        json.dumps(merged_params, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    train_out = model_creation(merged_params)

    val_sim = eval_val_mean_similarity(
        val_jsonl=val_jsonl,
        checkpoint=ckpt_path,
        max_samples=max_samples,
        batch_size=embed_batch_size,
    )

    iter_metrics: dict[str, Any] = {
        "run_id": run_id,
        "iter": iter_n,
        "mine": mine_summary,
        "refine": refine_summary,
        "manifest": manifest_summary,
        "train": train_out,
        "val_mean_similarity": val_sim,
        "checkpoint": str(ckpt_path),
        "no_refine": no_refine,
    }

    if run_gold_eval:
        gold_result = run_gold_retrieval_eval(checkpoint=ckpt_path, repo_root=_root)
        iter_metrics["gold_retrieval_eval"] = gold_result

    metrics_file = data_dir / "iter_metrics.json"
    write_iter_metrics(metrics_file, iter_metrics)

    return iter_metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CLAP self-training loop on local clap_train_15s.jsonl."
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Experiment id (subdirs under data/self_train and model/clap/self_train/).",
    )
    parser.add_argument("--n-iters", type=int, default=2, help="Number of loop iterations.")
    parser.add_argument("--hard-frac", type=float, default=0.2, help="Bottom fraction to mine.")
    parser.add_argument("--seed", type=int, default=42)
    refine_group = parser.add_mutually_exclusive_group()
    refine_group.add_argument(
        "--no-refine",
        dest="no_refine",
        action="store_true",
        default=True,
        help="Skip LLM refine (NoOp pass-through; default).",
    )
    refine_group.add_argument(
        "--refine",
        dest="no_refine",
        action="store_false",
        help="Enable LLM refine + CLAP gate (v2).",
    )
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        default=None,
        help=f"Base train manifest (default: {settings.CLAP_TRAIN_JSONL}).",
    )
    parser.add_argument(
        "--val-jsonl",
        type=Path,
        default=None,
        help=f"Val manifest for iter metrics (default: {settings.CLAP_VAL_JSONL}).",
    )
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="Val similarity patience for inner early stop (default 2).",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=None,
        help="Inner loop: do not val-early-stop before this many epochs (default 5).",
    )
    parser.add_argument(
        "--min-epochs-before-outer",
        type=int,
        default=5,
        help="Outer loop: only allow outer early-stop if prior iter trained >= N epochs.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap rows for dev smoke tests (mine + manifest + val eval).",
    )
    parser.add_argument("--embed-batch-size", type=int, default=8)
    parser.add_argument(
        "--params-json",
        type=Path,
        default=None,
        help="JSON merged on top of default train params.",
    )
    parser.add_argument(
        "--run-gold-eval",
        action="store_true",
        help="After each iter, run music_eval_retrieval_vs_random (needs gold + FAISS).",
    )
    parser.add_argument(
        "--gate-min-text-cos",
        type=float,
        default=None,
        help="Min cosine(text_orig, text_new) for LLM gate (default env or 0.85).",
    )
    parser.add_argument(
        "--gate-min-sim-gain",
        type=float,
        default=None,
        help="Min CLAP sim gain for accepted refinement (default 0).",
    )
    parser.add_argument(
        "--llm-max-new-tokens",
        type=int,
        default=256,
        help="Max tokens for Llama caption refine.",
    )
    parser.add_argument(
        "--refine-max-hard",
        type=int,
        default=None,
        help="Cap number of hard rows sent to LLM (dev smoke).",
    )
    parser.add_argument(
        "--min-val-delta",
        type=float,
        default=0.001,
        help="Min val sim improvement between iters for outer loop stop.",
    )
    parser.add_argument(
        "--outer-patience-iters",
        type=int,
        default=2,
        help="Stop outer loop after this many iters without val improvement.",
    )
    parser.add_argument(
        "--no-outer-early-stop",
        action="store_true",
        help="Disable outer-loop val plateau stop (run all n-iters).",
    )
    parser.add_argument(
        "--no-audio-cache",
        action="store_true",
        help="Disable backbone audio cache (re-decode MP3 every batch; slow).",
    )
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    train_jsonl = (args.train_jsonl or settings.CLAP_TRAIN_JSONL).resolve()
    val_jsonl = (args.val_jsonl or settings.CLAP_VAL_JSONL).resolve()

    if not train_jsonl.is_file():
        raise FileNotFoundError(f"Train JSONL not found: {train_jsonl}")
    if not val_jsonl.is_file():
        raise FileNotFoundError(f"Val JSONL not found: {val_jsonl}")

    train_params = _default_train_params()
    if args.num_epochs is not None:
        train_params["num_epochs"] = args.num_epochs
    if args.batch_size is not None:
        train_params["batch_size"] = args.batch_size
    if args.early_stop_patience is not None:
        train_params.setdefault("early_stopping", {})["patience"] = args.early_stop_patience
    if args.min_epochs is not None:
        train_params.setdefault("early_stopping", {})["min_epochs"] = args.min_epochs
    if args.params_json is not None:
        extra = json.loads(args.params_json.read_text(encoding="utf-8"))
        if not isinstance(extra, dict):
            raise ValueError("--params-json must be a JSON object")
        train_params.update(extra)

    gate_params = default_gate_params()
    if args.gate_min_text_cos is not None:
        gate_params["min_text_cos"] = args.gate_min_text_cos
    if args.gate_min_sim_gain is not None:
        gate_params["min_sim_gain"] = args.gate_min_sim_gain

    llm_params: dict[str, Any] = {
        "max_new_tokens": args.llm_max_new_tokens,
        "temperature": 0.2,
    }

    if not args.no_audio_cache:
        from app.data_handling.music_precompute_clap_audio_cache import ensure_clap_audio_cache

        cache_dir = ensure_clap_audio_cache(jsonl_paths=[train_jsonl, val_jsonl])
        train_params["audio_cache_dir"] = str(cache_dir)

    per_iter: list[dict[str, Any]] = []
    prev_val_sim: float | None = None
    stagnant_iters = 0

    for iter_n in range(args.n_iters):
        print(f"=== self-train iter {iter_n}/{args.n_iters - 1} run_id={run_id} ===", flush=True)
        summary = run_iteration(
            run_id=run_id,
            iter_n=iter_n,
            train_jsonl=train_jsonl,
            val_jsonl=val_jsonl,
            hard_frac=args.hard_frac,
            seed=args.seed,
            no_refine=args.no_refine,
            train_params=train_params,
            max_samples=args.max_samples,
            embed_batch_size=args.embed_batch_size,
            run_gold_eval=args.run_gold_eval,
            gate_params=gate_params,
            llm_params=llm_params,
            refine_max_hard=args.refine_max_hard,
        )
        per_iter.append(summary)
        val_sim = float(summary["val_mean_similarity"])
        print(
            json.dumps(
                {
                    "iter": iter_n,
                    "val_mean_similarity": val_sim,
                    "checkpoint": summary["checkpoint"],
                },
                indent=2,
            ),
            flush=True,
        )

        if not args.no_outer_early_stop and prev_val_sim is not None:
            inner_epochs = int(summary.get("train", {}).get("best_epoch", 0))
            inner_ok = inner_epochs >= args.min_epochs_before_outer
            if val_sim - prev_val_sim < args.min_val_delta:
                stagnant_iters += 1
            else:
                stagnant_iters = 0
            if stagnant_iters >= args.outer_patience_iters:
                if inner_ok:
                    print(
                        f"Outer early stop: val plateau for {stagnant_iters} iters "
                        f"(min_delta={args.min_val_delta}, "
                        f"inner_epochs={inner_epochs}>={args.min_epochs_before_outer})",
                        flush=True,
                    )
                    break
                print(
                    f"Outer early stop skipped: inner best_epoch={inner_epochs} "
                    f"< min_epochs_before_outer={args.min_epochs_before_outer}; "
                    f"continuing to next iter.",
                    flush=True,
                )
                stagnant_iters = 0
        prev_val_sim = val_sim

    log_run = settings.self_train_log_run_dir(run_id)
    log_run.mkdir(parents=True, exist_ok=True)
    run_summary = {
        "run_id": run_id,
        "n_iters": args.n_iters,
        "hard_frac": args.hard_frac,
        "seed": args.seed,
        "no_refine": args.no_refine,
        "train_jsonl": str(train_jsonl),
        "val_jsonl": str(val_jsonl),
        "max_samples": args.max_samples,
        "gate_params": gate_params,
        "refine_max_hard": args.refine_max_hard,
        "per_iter": per_iter,
        "outer_stopped_early": len(per_iter) < args.n_iters,
    }
    summary_path = log_run / "summary.json"
    summary_path.write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"summary": str(summary_path)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
