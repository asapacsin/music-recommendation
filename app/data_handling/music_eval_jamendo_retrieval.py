"""
Jamendo five-tag public OOD test (thin wrapper around ``music_eval_public_retrieval``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import settings

from app.data_handling.music_eval_public_retrieval import (
    PRIMARY_PUB_EVAL_FULL as PRIMARY_PUB_EVAL,
    DEFAULT_MANIFEST_BY_DATASET,
    eval_public_retrieval as eval_jamendo_retrieval,
    load_public_manifest as load_jamendo_manifest,
    run_public_retrieval_cli,
    write_matrix_csv as _write_csv,
)

__all__ = [
    "PRIMARY_PUB_EVAL",
    "load_jamendo_manifest",
    "eval_jamendo_retrieval",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Jamendo five-tag audio–text retrieval (public OOD, no FAISS)."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_BY_DATASET["jamendo"],
    )
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ndcg-random-iters", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audio-batch-size", type=int, default=16)
    parser.add_argument("--arm", type=str, default="")
    args = parser.parse_args()

    return run_public_retrieval_cli(
        dataset="jamendo",
        manifest=args.manifest,
        out_csv=args.out_csv,
        out_json=args.out_json,
        top_k=args.top_k,
        ndcg_random_iters=args.ndcg_random_iters,
        seed=args.seed,
        audio_batch_size=args.audio_batch_size,
        arm=args.arm,
    )


if __name__ == "__main__":
    raise SystemExit(main())
