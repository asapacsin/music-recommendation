"""
Song-level LLM expansion of tag-derived training text (full corpus, resumable).

Reads ``clap_train_tag.jsonl``; writes progress per song then merged ``clap_train_tag_llm.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import settings

from app.data_handling.music_refine_full_corpus_captions import (
    _norm_source_key,
    collect_unique_songs,
    load_progress_map,
)
from app.llm_local import chat_generate, load_llm, model_is_ready
from app.self_train.jsonl_io import write_jsonl_rows

_REFINE_SYSTEM = (
    "You write short English captions for music retrieval. "
    "Do not invent artist names, song titles, or lyrics. "
    "One or two sentences."
)


def build_refine_user_prompt_from_tags(row: dict[str, Any]) -> str:
    tag_text = str(row.get("text") or "").strip()
    parts = [
        "Expand this tag-based music description into a short natural-language caption "
        "for retrieval. Keep the same tags and facts; do not add new instruments or moods.",
        f"Tag description: {tag_text}",
    ]
    text_source = row.get("text_source")
    if isinstance(text_source, str) and text_source.strip():
        parts.append(f"Source: {text_source.strip()}")
    parts.append("Return only the revised caption.")
    return "\n".join(parts)


def refine_tag_songs(
    *,
    train_jsonl: Path,
    progress_jsonl: Path,
    max_songs: int | None = None,
    llm_max_new_tokens: int = 256,
    llm_temperature: float = 0.2,
) -> dict[str, Any]:
    if not model_is_ready():
        raise FileNotFoundError(
            "Local LLM not ready. Run: bash scripts/download_llama31_8b.sh"
        )

    songs = collect_unique_songs(train_jsonl)
    done = load_progress_map(progress_jsonl)
    todo_keys = [k for k in sorted(songs) if k not in done]
    if max_songs is not None:
        todo_keys = todo_keys[: max(0, int(max_songs))]

    progress_jsonl.parent.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_llm()
    n_written = 0
    n_errors = 0

    try:
        with progress_jsonl.open("a", encoding="utf-8") as out_f:
            for key in todo_keys:
                row = songs[key]
                source_path = row["source_path"]
                try:
                    user_prompt = build_refine_user_prompt_from_tags(row)
                    llm_text = chat_generate(
                        user_prompt,
                        model,
                        tokenizer,
                        system_prompt=_REFINE_SYSTEM,
                        max_new_tokens=llm_max_new_tokens,
                        temperature=llm_temperature,
                    )
                    if not llm_text.strip():
                        n_errors += 1
                        continue
                    out_row = {
                        "source_path": source_path,
                        "text": llm_text.strip(),
                        "text_orig": row.get("text"),
                        "text_source": "tag_llm_refined",
                    }
                    out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                    out_f.flush()
                    done[key] = llm_text.strip()
                    n_written += 1
                    if n_written % 50 == 0:
                        print(
                            f"tag refine {n_written} new songs "
                            f"({len(done)}/{len(songs)} total)",
                            flush=True,
                        )
                except Exception as exc:
                    n_errors += 1
                    print(f"warning: tag refine failed for {source_path}: {exc}", flush=True)
    finally:
        if hasattr(model, "cpu"):
            model.cpu()
        del model
        del tokenizer
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    return {
        "train_jsonl": str(train_jsonl.resolve()),
        "progress_jsonl": str(progress_jsonl.resolve()),
        "n_songs_total": len(songs),
        "n_songs_done": len(done),
        "n_songs_new": n_written,
        "n_errors": n_errors,
        "n_songs_remaining": len(songs) - len(done),
    }


def merge_tag_llm_train_jsonl(
    *,
    train_jsonl: Path,
    progress_jsonl: Path,
    out_jsonl: Path,
) -> dict[str, Any]:
    song_text = load_progress_map(progress_jsonl)
    if not song_text:
        raise ValueError(f"No refined songs in {progress_jsonl}")

    out_rows: list[dict[str, Any]] = []
    n_replaced = 0
    n_missing_song = 0

    with train_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            source_path = row.get("source_path")
            if not isinstance(source_path, str) or not source_path.strip():
                continue

            out = dict(row)
            key = _norm_source_key(source_path)
            llm_text = song_text.get(key)
            if llm_text is None:
                n_missing_song += 1
                out.setdefault("text_source", row.get("text_source", "tag_primary"))
            else:
                if "text_orig" not in out:
                    out["text_orig_tag"] = out.get("text")
                out["text"] = llm_text
                out["text_source"] = "tag_llm_refined"
                n_replaced += 1
            out_rows.append(out)

    if not out_rows:
        raise ValueError(f"No train rows read from {train_jsonl}")

    write_jsonl_rows(out_jsonl, out_rows)
    return {
        "train_jsonl": str(train_jsonl.resolve()),
        "progress_jsonl": str(progress_jsonl.resolve()),
        "out_jsonl": str(out_jsonl.resolve()),
        "n_total": len(out_rows),
        "n_replaced": n_replaced,
        "n_missing_song": n_missing_song,
        "n_songs_in_progress": len(song_text),
    }


def check_tag_llm_refine_complete(
    *,
    train_jsonl: Path,
    progress_jsonl: Path,
) -> dict[str, int | str]:
    n_total = len(collect_unique_songs(train_jsonl))
    n_done = len(load_progress_map(progress_jsonl))
    if n_done < n_total:
        raise RuntimeError(
            f"Tag LLM refine incomplete: {n_done}/{n_total} songs in {progress_jsonl}"
        )
    print(f"Tag LLM refine complete: {n_done}/{n_total} songs", flush=True)
    return {
        "train_jsonl": str(train_jsonl.resolve()),
        "progress_jsonl": str(progress_jsonl.resolve()),
        "n_songs_total": n_total,
        "n_songs_done": n_done,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Full-corpus LLM expand tag-derived train text (one call per song)."
    )
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        default=settings.MAPPING_DIR / "clap_train_tag.jsonl",
        help="Tag train manifest (clip-level).",
    )
    parser.add_argument(
        "--progress-jsonl",
        type=Path,
        default=settings.MAPPING_DIR / "clap_train_tag_llm_songs.jsonl",
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=settings.MAPPING_DIR / "clap_train_tag_llm.jsonl",
    )
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--check-complete-only", action="store_true")
    parser.add_argument("--max-songs", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    train_jsonl = args.train_jsonl.resolve()
    progress_jsonl = args.progress_jsonl.resolve()
    out_jsonl = args.out_jsonl.resolve()

    if args.check_complete_only:
        try:
            check_tag_llm_refine_complete(
                train_jsonl=train_jsonl,
                progress_jsonl=progress_jsonl,
            )
        except RuntimeError as exc:
            print(str(exc), flush=True)
            return 1
        return 0

    if not args.merge_only:
        refine_stats = refine_tag_songs(
            train_jsonl=train_jsonl,
            progress_jsonl=progress_jsonl,
            max_songs=args.max_songs,
            llm_max_new_tokens=args.max_new_tokens,
            llm_temperature=args.temperature,
        )
        print(json.dumps(refine_stats, ensure_ascii=False, indent=2))
        if refine_stats["n_songs_remaining"] > 0:
            print(
                f"Remaining songs: {refine_stats['n_songs_remaining']} (re-run to continue)",
                flush=True,
            )

    merge_stats = merge_tag_llm_train_jsonl(
        train_jsonl=train_jsonl,
        progress_jsonl=progress_jsonl,
        out_jsonl=out_jsonl,
    )
    print(json.dumps(merge_stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
