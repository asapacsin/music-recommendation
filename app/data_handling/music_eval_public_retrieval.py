"""
Post-train audio–text retrieval on public OOD manifests (no FAISS, no training).

Shared by Jamendo, MagnaTagATune, and OpenMIC test scripts.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import settings

from app.data_handling.music_eval_retrieval_vs_random import (
    _ndcg_at_k,
    _random_ndcg_mc,
)

# (tag_id, gold_column, query_text) — aligned with in-domain PRIMARY_TAGS
PRIMARY_PUB_EVAL_FULL: list[tuple[str, str, str]] = [
    ("inst_piano", "gold_pub_piano", "piano"),
    ("inst_vocal", "gold_pub_vocal", "vocal"),
    ("mood_relaxing", "gold_pub_relaxing", "relaxing"),
]

PRIMARY_PUB_EVAL_OPENMIC: list[tuple[str, str, str]] = [
    ("inst_piano", "gold_pub_piano", "piano"),
    ("inst_vocal", "gold_pub_vocal", "vocal"),
]

PRIMARY_BY_DATASET: dict[str, list[tuple[str, str, str]]] = {
    "jamendo": PRIMARY_PUB_EVAL_FULL,
    "mtat": PRIMARY_PUB_EVAL_FULL,
    "openmic": PRIMARY_PUB_EVAL_OPENMIC,
}

DEFAULT_MANIFEST_BY_DATASET: dict[str, Path] = {
    "jamendo": settings.DATA_DIR / "eval" / "jamendo_five_tag_manifest.jsonl",
    "mtat": settings.DATA_DIR / "eval" / "mtat_manifest.jsonl",
    "openmic": settings.DATA_DIR / "eval" / "openmic_manifest.jsonl",
}


def _normalize_embeddings(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def load_public_manifest(manifest_jsonl: Path) -> list[dict[str, Any]]:
    """Load rows with a non-empty local ``audio_path`` file."""
    if not manifest_jsonl.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_jsonl}")

    rows: list[dict[str, Any]] = []
    skipped = 0
    with manifest_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                continue
            ap = rec.get("audio_path")
            if not isinstance(ap, str) or not ap.strip():
                skipped += 1
                continue
            path = Path(ap)
            if not path.is_file() or path.stat().st_size <= 0:
                skipped += 1
                continue
            rows.append(rec)

    if not rows:
        raise ValueError(
            f"No evaluable rows in {manifest_jsonl} (need audio_path files on disk)."
        )
    if skipped:
        print(
            f"warning: skipped {skipped} manifest rows without local audio",
            file=sys.stderr,
        )
    return rows


def _embed_audio_batch(
    model: Any,
    paths: list[str],
    *,
    batch_size: int,
    desc: str = "Embed public audio",
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in tqdm(
        range(0, len(paths), batch_size),
        desc=desc,
        unit="batch",
    ):
        batch = paths[start : start + batch_size]
        emb = model.get_audio_embedding_from_filelist(x=batch, use_tensor=False)
        chunks.append(np.asarray(emb, dtype=np.float32))
    out = np.vstack(chunks).astype(np.float32)
    return _normalize_embeddings(out)


def _relevance_vector(rows: list[dict[str, Any]], gold_col: str) -> np.ndarray:
    rel = []
    for rec in rows:
        val = rec.get(gold_col, 0)
        try:
            rel.append(1.0 if int(val) == 1 else 0.0)
        except (TypeError, ValueError):
            rel.append(0.0)
    return np.asarray(rel, dtype=np.float32)


def eval_public_retrieval(
    *,
    rows: list[dict[str, Any]],
    model: Any,
    primary_eval: list[tuple[str, str, str]],
    top_k: int,
    ndcg_random_iters: int,
    seed: int,
    audio_batch_size: int = 16,
    embed_desc: str = "Embed public audio",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return CSV-style metric rows and summary metadata."""
    paths = [str(rec["audio_path"]) for rec in rows]
    audio_emb = _embed_audio_batch(
        model, paths, batch_size=audio_batch_size, desc=embed_desc
    )
    n_pool = len(rows)

    csv_rows: list[dict[str, Any]] = []
    for tag_id, gold_col, query_text in primary_eval:
        rel = _relevance_vector(rows, gold_col)
        r_pos = int(rel.sum())
        if r_pos == 0:
            print(
                f"warning: no positives for {tag_id} ({gold_col}); metrics may be zero",
                file=sys.stderr,
            )
        prevalence = (r_pos / n_pool) if n_pool else 0.0

        q_raw = model.get_text_embedding(x=[query_text], use_tensor=False)
        q_emb = _normalize_embeddings(np.asarray(q_raw, dtype=np.float32)).reshape(-1)
        scores = audio_emb @ q_emb.astype(np.float32)
        order = np.argsort(-scores)

        k_eff = min(top_k, n_pool)
        p_model = float(rel[order[:k_eff]].sum()) / float(k_eff) if k_eff else 0.0
        ndcg_model = _ndcg_at_k(rel, order, top_k)

        p_rand = prevalence
        ndcg_rand = _random_ndcg_mc(
            rel,
            top_k,
            n_iters=ndcg_random_iters,
            seed=seed + sum(ord(c) for c in tag_id) % 10_007,
        )

        csv_rows.append(
            {
                "query_id": tag_id,
                "query_text": query_text,
                "gold_column": gold_col,
                "top_k": top_k,
                "n_pool": n_pool,
                "n_positive": r_pos,
                "prevalence": prevalence,
                "precision_at_k": p_model,
                "precision_delta": p_model - p_rand,
                "ndcg_at_k": ndcg_model,
                "ndcg_random_mean": ndcg_rand,
                "ndcg_delta": ndcg_model - ndcg_rand,
            }
        )

    meta = {
        "n_pool": n_pool,
        "n_tracks_manifest": n_pool,
        "primary_tags": [t[0] for t in primary_eval],
        "positives_per_tag": {
            t[0]: int(_relevance_vector(rows, t[1]).sum()) for t in primary_eval
        },
    }
    return csv_rows, meta


def write_matrix_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run_public_retrieval_cli(
    *,
    dataset: str,
    manifest: Path,
    out_csv: Path,
    out_json: Path | None,
    top_k: int,
    ndcg_random_iters: int,
    seed: int,
    audio_batch_size: int,
    arm: str,
) -> int:
    ds = dataset.strip().lower()
    if ds not in PRIMARY_BY_DATASET:
        raise ValueError(f"Unknown dataset {dataset!r}; use: {list(PRIMARY_BY_DATASET)}")

    from app.data_handling.music_eval_topk_prepare import _load_model

    primary = PRIMARY_BY_DATASET[ds]
    rows = load_public_manifest(manifest.resolve())
    model = _load_model()

    csv_rows, meta = eval_public_retrieval(
        rows=rows,
        model=model,
        primary_eval=primary,
        top_k=top_k,
        ndcg_random_iters=ndcg_random_iters,
        seed=seed,
        audio_batch_size=audio_batch_size,
        embed_desc=f"Embed {ds} audio",
    )

    ckpt = os.environ.get("RAGWEB_CLAP_CHECKPOINT", "")
    out_csv = out_csv.resolve()
    out_json_path = out_json.resolve() if out_json else out_csv.with_suffix(".json")

    payload = {
        "manifest": str(manifest.resolve()),
        "dataset": ds,
        "arm": arm or None,
        "checkpoint_env": ckpt or None,
        "top_k": top_k,
        **meta,
        "rows": csv_rows,
    }
    write_matrix_csv(out_csv, csv_rows)
    out_json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"written_csv": str(out_csv), "written_json": str(out_json_path)},
            indent=2,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Public OOD audio–text retrieval (post-train test, no FAISS)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=sorted(PRIMARY_BY_DATASET),
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ndcg-random-iters", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audio-batch-size", type=int, default=16)
    parser.add_argument("--arm", type=str, default="")
    args = parser.parse_args()

    ds = args.dataset.strip().lower()
    manifest = args.manifest or DEFAULT_MANIFEST_BY_DATASET[ds]
    return run_public_retrieval_cli(
        dataset=ds,
        manifest=manifest,
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
