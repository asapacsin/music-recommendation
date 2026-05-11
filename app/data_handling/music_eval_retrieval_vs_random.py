"""
Gold-labeled retrieval vs random baseline (metadata FAISS text index).

Eval pool = FAISS rows whose metadata audio basename appears in gold_merged.jsonl.
Scores CLAP text queries against indexed embeddings (same as top-k prepare).

For each query_id matching a MULTIHOT_COLUMNS key, computes precision@K and nDCG@K
vs random baselines (expected precision = prevalence; Monte Carlo nDCG).

CSV columns: query_text, top_k, n_pool, n_positive, prevalence, precision_at_k,
precision_delta, ndcg_at_k, ndcg_random_mean, ndcg_delta.

With ``--include-tempo``, three columns are appended (same values on every row):
tempo_accuracy, tempo_macro_f1, tempo_n_songs — from song-level BPM vs CLAP zero-shot
already stored in ``gold_merged.jsonl`` (``program_tempo``), intersected with the
metadata pool basenames (deduped per song).

CLAP prompts strip a trailing " music" suffix (case-insensitive); CSV query_text matches
that same stripped phrase.

Outputs CSV + JSON under data/eval/ by default.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import settings

from app.data_handling.music_eval_prepare_gold_multihot_csv import MULTIHOT_COLUMNS


def _norm_sp(value: str) -> str:
    return value.strip().replace("\\", "/")


def _basename_key(path_str: str) -> str:
    return Path(_norm_sp(path_str)).name


def _strip_music_suffix(text: str) -> str:
    """Remove trailing ' music' for CLAP prompts and CSV labels (case-insensitive)."""
    t = text.strip()
    low = t.lower()
    suffix = " music"
    if low.endswith(suffix):
        return t[: -len(suffix)].strip()
    return t


def _normalize_embeddings(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def _load_gold_by_basename(path: Path) -> dict[str, dict[str, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"Gold JSONL not found: {path}")
    out: dict[str, dict[str, int]] = {}
    dupes = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                continue
            sp = rec.get("source_path")
            if not isinstance(sp, str) or not sp.strip():
                continue
            bn = _basename_key(sp)
            hm = rec.get("human_multihot")
            if not isinstance(hm, dict):
                continue
            labels = {c: int(bool(hm.get(c))) for c in MULTIHOT_COLUMNS}
            if bn in out:
                dupes += 1
            out[bn] = labels
    if dupes:
        print(f"warning: {dupes} duplicate basename keys in gold (last row wins)", file=sys.stderr)
    return out


def _load_tempo_by_basename_from_gold(
    path: Path,
    *,
    tempo_labels: frozenset[str],
) -> dict[str, tuple[str, str]]:
    """basename -> (gt_tempo, pred_tempo) from program_tempo; last JSONL row wins per basename."""
    out: dict[str, tuple[str, str]] = {}
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                continue
            sp = rec.get("source_path")
            if not isinstance(sp, str) or not sp.strip():
                continue
            pt = rec.get("program_tempo")
            if not isinstance(pt, dict):
                continue
            gt = pt.get("tempo_bin_bpm")
            pred = pt.get("tempo_clap_zeroshot")
            if not isinstance(gt, str) or not isinstance(pred, str):
                continue
            if gt not in tempo_labels or pred not in tempo_labels:
                continue
            bn = _basename_key(sp)
            out[bn] = (gt, pred)
    return out


def _dcg_from_ranked(rels: list[float]) -> float:
    total = 0.0
    for i, rel in enumerate(rels, start=1):
        total += (2.0**float(rel) - 1.0) / math.log2(i + 1)
    return total


def _ndcg_at_k(rel: np.ndarray, order: np.ndarray, k: int) -> float:
    n = len(rel)
    if n <= 0:
        return 0.0
    k_eff = min(k, n)
    ranked = rel[order[:k_eff]].astype(float).tolist()
    dcg = _dcg_from_ranked(ranked)
    ideal = sorted(float(x) for x in rel.flat)
    ideal.sort(reverse=True)
    ideal = ideal[:k_eff]
    idcg = _dcg_from_ranked(ideal)
    return (dcg / idcg) if idcg > 0 else 0.0


def _random_hit_at_k(*, n_pool: int, n_pos: int, k: int) -> float:
    """P(at least one relevant in top-K) under uniform random ranking (kept for tests)."""
    if n_pool <= 0 or k <= 0:
        return 0.0
    k_eff = min(k, n_pool)
    r = int(n_pos)
    if r <= 0:
        return 0.0
    if r >= n_pool:
        return 1.0
    nr = n_pool - r
    if k_eff > nr:
        return 1.0
    return 1.0 - math.comb(nr, k_eff) / math.comb(n_pool, k_eff)


def _random_ndcg_mc(
    rel: np.ndarray,
    k: int,
    *,
    n_iters: int,
    seed: int,
) -> float:
    n = len(rel)
    if n <= 0 or k <= 0:
        return 0.0
    k_eff = min(k, n)
    ideal = sorted(float(x) for x in rel.flat)
    ideal.sort(reverse=True)
    ideal = ideal[:k_eff]
    idcg = _dcg_from_ranked(ideal)
    if idcg <= 0:
        return 0.0
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(n_iters):
        perm = rng.permutation(n)
        ranked = rel[perm][:k_eff].astype(float).tolist()
        total += _dcg_from_ranked(ranked) / idcg
    return total / float(n_iters)


def _build_eval_pool(
    *,
    index: Any,
    mapping: dict[str, dict[str, Any]],
    gold_by_basename: dict[str, dict[str, int]],
) -> tuple[list[int], dict[int, str]]:
    pool_ids: list[int] = []
    basename_for_id: dict[int, str] = {}
    ntotal = int(index.ntotal)
    for i in range(ntotal):
        row = mapping.get(str(i))
        if not isinstance(row, dict):
            continue
        audio = row.get("audio")
        if not isinstance(audio, str) or not audio.strip():
            continue
        bn = _basename_key(audio)
        if bn not in gold_by_basename:
            continue
        pool_ids.append(i)
        basename_for_id[i] = bn
    pool_ids.sort()
    return pool_ids, basename_for_id


def _score_pool(
    *,
    index: Any,
    pool_ids: list[int],
    query_emb: np.ndarray,
) -> np.ndarray:
    """Inner product scores (indexed vectors are L2-normalized at build time)."""
    d = int(index.d)
    mat = np.zeros((len(pool_ids), d), dtype=np.float32)
    for j, fid in enumerate(pool_ids):
        mat[j] = np.asarray(index.reconstruct(int(fid)), dtype=np.float32)
    q = query_emb.astype(np.float32).reshape(-1)
    return mat @ q


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retrieval metrics vs random baseline per class (gold multihot + metadata FAISS)."
    )
    parser.add_argument(
        "--gold-jsonl",
        type=Path,
        default=settings.DATA_DIR / "eval" / "gold_merged.jsonl",
        help="Merged gold JSONL (human_multihot + source_path).",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=settings.DATA_DIR / "eval" / "style_queries.json",
        help="Style queries JSON (query_id must match a multihot column name).",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=settings.METADATA_TEXT_INDEX_FILE,
        help="Metadata FAISS index path.",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=settings.METADATA_ID_MAPPING_FILE,
        help="Metadata id mapping JSON path.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[10],
        help="One or more K values (e.g. --top-k 5 10 20).",
    )
    parser.add_argument(
        "--ndcg-random-iters",
        type=int,
        default=256,
        help="Monte Carlo draws for mean nDCG under random ranking.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for nDCG Monte Carlo.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=settings.DATA_DIR / "eval" / "retrieval_vs_random_matrix.json",
        help="JSON output path.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=settings.DATA_DIR / "eval" / "retrieval_vs_random_matrix.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--include-tempo",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append tempo_accuracy, tempo_macro_f1, tempo_n_songs from gold_merged program_tempo.",
    )
    args = parser.parse_args()

    import faiss  # noqa: PLC0415 — defer heavy deps until CLI runs

    from app.data_handling.music_eval_topk_prepare import (  # noqa: PLC0415
        _load_model,
        _load_queries,
    )

    ks = [int(k) for k in args.top_k]
    if any(k <= 0 for k in ks):
        raise ValueError("--top-k values must be positive")

    gold_by_basename = _load_gold_by_basename(args.gold_jsonl)
    if not gold_by_basename:
        raise ValueError(f"No gold rows loaded from {args.gold_jsonl}")

    queries = _load_queries(args.queries)
    index = faiss.read_index(str(args.index))
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise ValueError("Mapping JSON must be an object")

    pool_ids, basename_for_id = _build_eval_pool(
        index=index, mapping=mapping, gold_by_basename=gold_by_basename
    )
    n_pool = len(pool_ids)
    if n_pool == 0:
        raise ValueError(
            "Eval pool is empty: no FAISS rows matched gold basenames. "
            "Check metadata audio paths vs gold source_path (basename match)."
        )

    pool_basenames = frozenset(basename_for_id.values())

    tempo_meta: dict[str, Any] | None = None
    tempo_csv_fields: dict[str, str] = {}
    if args.include_tempo:
        from app.data_handling.music_eval_zeroshot_tempo import (  # noqa: PLC0415
            TEMPO_LABELS,
            _compute_metrics,
        )

        labels_set = frozenset(TEMPO_LABELS)
        tempo_by_bn = _load_tempo_by_basename_from_gold(args.gold_jsonl, tempo_labels=labels_set)
        y_true: list[str] = []
        y_pred: list[str] = []
        for bn in sorted(pool_basenames):
            pair = tempo_by_bn.get(bn)
            if pair is None:
                continue
            y_true.append(pair[0])
            y_pred.append(pair[1])

        n_tempo = len(y_true)
        if n_tempo == 0:
            print(
                "warning: --include-tempo set but no songs in pool have valid "
                "program_tempo.tempo_bin_bpm and tempo_clap_zeroshot in gold_merged.jsonl; "
                "run song-level tempo eval and music_eval_merge_gold first.",
                file=sys.stderr,
            )
            tempo_meta = {
                "n_songs": 0,
                "accuracy": None,
                "macro_f1": None,
                "source": "gold_merged.program_tempo",
            }
            tempo_csv_fields = {
                "tempo_accuracy": "",
                "tempo_macro_f1": "",
                "tempo_n_songs": "",
            }
        else:
            metrics = _compute_metrics(y_true, y_pred)
            tempo_meta = {
                "n_songs": n_tempo,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "labels": metrics["labels"],
                "per_class": metrics["per_class"],
                "confusion_matrix": metrics["confusion_matrix"],
                "source": "gold_merged.program_tempo (tempo_bin_bpm vs tempo_clap_zeroshot)",
            }
            tempo_csv_fields = {
                "tempo_accuracy": f"{metrics['accuracy']:.6f}",
                "tempo_macro_f1": f"{metrics['macro_f1']:.6f}",
                "tempo_n_songs": str(n_tempo),
            }

    model = _load_model()

    csv_rows: list[dict[str, Any]] = []
    skipped: list[str] = []

    for query in queries:
        qid = query["query_id"]
        if qid not in MULTIHOT_COLUMNS:
            skipped.append(qid)
            continue

        rel = np.array(
            [gold_by_basename[basename_for_id[i]][qid] for i in pool_ids],
            dtype=np.float32,
        )
        r_pos = int(rel.sum())
        prevalence = (r_pos / n_pool) if n_pool else 0.0

        phrase = _strip_music_suffix(query["query_text"])
        q_raw = model.get_text_embedding(x=[phrase], use_tensor=False)
        q_emb = _normalize_embeddings(np.asarray(q_raw, dtype=np.float32))
        scores = _score_pool(index=index, pool_ids=pool_ids, query_emb=q_emb.squeeze(0))
        order = np.argsort(-scores)

        for k in ks:
            k_eff = min(k, n_pool)
            p_model = float(rel[order[:k_eff]].sum()) / float(k_eff)
            ndcg_model = _ndcg_at_k(rel, order, k)

            p_rand = prevalence
            ndcg_rand = _random_ndcg_mc(
                rel,
                k,
                n_iters=args.ndcg_random_iters,
                seed=args.seed + k * 100_003 + sum(ord(c) for c in qid) % 10_007,
            )

            row: dict[str, Any] = {
                "query_text": phrase,
                "top_k": k,
                "n_pool": n_pool,
                "n_positive": r_pos,
                "prevalence": prevalence,
                "precision_at_k": p_model,
                "precision_delta": p_model - p_rand,
                "ndcg_at_k": ndcg_model,
                "ndcg_random_mean": ndcg_rand,
                "ndcg_delta": ndcg_model - ndcg_rand,
            }
            if args.include_tempo:
                row.update(tempo_csv_fields)
            csv_rows.append(row)

    csv_columns = [
        "query_text",
        "top_k",
        "n_pool",
        "n_positive",
        "prevalence",
        "precision_at_k",
        "precision_delta",
        "ndcg_at_k",
        "ndcg_random_mean",
        "ndcg_delta",
    ]
    if args.include_tempo:
        csv_columns.extend(["tempo_accuracy", "tempo_macro_f1", "tempo_n_songs"])

    meta: dict[str, Any] = {
        "gold_jsonl": str(args.gold_jsonl),
        "queries_file": str(args.queries),
        "index": str(args.index),
        "mapping": str(args.mapping),
        "n_faiss_total": int(index.ntotal),
        "n_pool": n_pool,
        "top_k_values": ks,
        "ndcg_random_iters": args.ndcg_random_iters,
        "seed": args.seed,
        "skipped_query_ids_not_in_multihot": skipped,
        "csv_columns": csv_columns,
        "include_tempo": bool(args.include_tempo),
    }
    if tempo_meta is not None:
        meta["tempo"] = tempo_meta

    payload = {
        "meta": meta,
        "rows": csv_rows,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = list(csv_columns)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in csv_rows:
            w.writerow(row)

    print(json.dumps({"written_json": str(args.out_json), "written_csv": str(args.out_csv)}, indent=2))
    if skipped:
        print(f"warning: skipped query_id(s) not in MULTIHOT_COLUMNS: {skipped}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
