"""
Composite CLAP text queries: cumulative tags in one prompt (no trailing " music").

Evaluates retrieval on gold multihot with AND relevance — a song is relevant only if
all tags in the current prompt set are 1.

Default tags (reliable / fine-tune scope only): inst_piano, inst_vocal, mood_relaxing.
Prompts: "piano" -> "piano vocal" -> "piano vocal relaxing".

Run once per checkpoint (unset RAGWEB_CLAP_CHECKPOINT = pretrained; set for each seed).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import settings

# Fine-tune / headline tags only (drop low-reliability columns).
COMPOSITE_TAG_ORDER: list[tuple[str, str]] = [
    ("inst_piano", "piano"),
    ("inst_vocal", "vocal"),
    ("mood_relaxing", "relaxing"),
]


def _import_retrieval_helpers() -> tuple[Any, ...]:
    from app.data_handling.music_eval_retrieval_vs_random import (  # noqa: PLC0415
        _build_eval_pool,
        _load_gold_multihot_and_tempo_bins,
        _ndcg_at_k,
        _normalize_embeddings,
        _random_ndcg_mc,
        _score_pool,
    )
    from app.data_handling.music_eval_topk_prepare import _load_model  # noqa: PLC0415

    return (
        _build_eval_pool,
        _load_gold_multihot_and_tempo_bins,
        _ndcg_at_k,
        _normalize_embeddings,
        _random_ndcg_mc,
        _score_pool,
        _load_model,
    )


def _and_relevance(
    gold_by_basename: dict[str, dict[str, int]],
    basename_for_id: dict[int, str],
    pool_ids: list[int],
    tag_ids: list[str],
) -> np.ndarray:
    rel = np.zeros(len(pool_ids), dtype=np.float32)
    for j, fid in enumerate(pool_ids):
        bn = basename_for_id[fid]
        labels = gold_by_basename[bn]
        if all(int(labels.get(tid, 0)) != 0 for tid in tag_ids):
            rel[j] = 1.0
    return rel


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Composite-tag CLAP retrieval (cumulative prompt, AND multihot relevance)."
    )
    parser.add_argument(
        "--gold-jsonl",
        type=Path,
        default=settings.DATA_DIR / "eval" / "gold_merged.jsonl",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=settings.DATA_DIR / "index" / "metadata_text_index.faiss",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=settings.MAPPING_DIR / "metadata_id_mapping.json",
    )
    parser.add_argument("--top-k", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--ndcg-random-iters", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=settings.DATA_DIR / "eval" / "ablation" / "composite" / "composite_pretrained.csv",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional JSON export (default: <out-csv>.json).",
    )
    parser.add_argument(
        "--model-label",
        type=str,
        default="pretrained",
        help="Label stored in JSON meta (e.g. pretrained, ft_seed42).",
    )
    args = parser.parse_args()

    ks = [int(k) for k in args.top_k]
    if any(k <= 0 for k in ks):
        raise ValueError("--top-k values must be positive")

    (
        _build_eval_pool,
        _load_gold_multihot_and_tempo_bins,
        _ndcg_at_k,
        _normalize_embeddings,
        _random_ndcg_mc,
        _score_pool,
        _load_model,
    ) = _import_retrieval_helpers()

    import faiss  # noqa: PLC0415

    from app.data_handling.music_eval_zeroshot_tempo import TEMPO_LABELS  # noqa: PLC0415

    gold_load = _load_gold_multihot_and_tempo_bins(
        args.gold_jsonl, tempo_labels=frozenset(TEMPO_LABELS)
    )
    gold_by_basename = gold_load.multihot_by_basename
    if not gold_by_basename:
        raise ValueError(f"No gold rows in {args.gold_jsonl}")

    index = faiss.read_index(str(args.index))
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise ValueError("Mapping JSON must be an object")

    pool_ids, basename_for_id = _build_eval_pool(
        index=index, mapping=mapping, gold_by_basename=gold_by_basename
    )
    n_pool = len(pool_ids)
    if n_pool == 0:
        raise ValueError("Eval pool is empty (no FAISS rows matched gold basenames).")

    model = _load_model()
    csv_rows: list[dict[str, Any]] = []

    for k in range(1, len(COMPOSITE_TAG_ORDER) + 1):
        tag_slice = COMPOSITE_TAG_ORDER[:k]
        tag_ids = [tid for tid, _ in tag_slice]
        phrase = " ".join(word for _, word in tag_slice)
        rel = _and_relevance(gold_by_basename, basename_for_id, pool_ids, tag_ids)
        r_pos = int(rel.sum())
        prevalence = (r_pos / n_pool) if n_pool else 0.0

        q_raw = model.get_text_embedding(x=[phrase], use_tensor=False)
        q_emb = _normalize_embeddings(np.asarray(q_raw, dtype=np.float32))
        scores = _score_pool(index=index, pool_ids=pool_ids, query_emb=q_emb.squeeze(0))
        order = np.argsort(-scores)

        seed_tag = sum(ord(c) for c in "".join(tag_ids)) % 10_007

        for top_k in ks:
            k_eff = min(top_k, n_pool)
            p_model = float(rel[order[:k_eff]].sum()) / float(k_eff)
            ndcg_model = _ndcg_at_k(rel, order, top_k)
            p_rand = prevalence
            ndcg_rand = _random_ndcg_mc(
                rel,
                top_k,
                n_iters=args.ndcg_random_iters,
                seed=args.seed + top_k * 100_003 + seed_tag,
            )
            csv_rows.append(
                {
                    "n_tags_in_prompt": k,
                    "query_ids": ";".join(tag_ids),
                    "query_text": phrase,
                    "relevance": "and",
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

    out_csv = args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(csv_rows[0].keys()) if csv_rows else []
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)

    out_json = args.out_json or out_csv.with_suffix(".json")
    meta = {
        "report_type": "composite_query_ablation",
        "model_label": args.model_label,
        "gold_jsonl": str(args.gold_jsonl),
        "index": str(args.index),
        "mapping": str(args.mapping),
        "n_pool": n_pool,
        "top_k_values": ks,
        "tag_order": [{"query_id": t, "prompt_word": w} for t, w in COMPOSITE_TAG_ORDER],
        "relevance": "and",
        "prompt_suffix_music": False,
        "checkpoint_env": __import__("os").environ.get("RAGWEB_CLAP_CHECKPOINT"),
        "written_csv": str(out_csv),
    }
    payload = {"meta": meta, "rows": csv_rows}
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"written_csv": str(out_csv), "written_json": str(out_json), "n_rows": len(csv_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
