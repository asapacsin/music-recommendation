"""
Full-corpus tag-aware LLM caption refinement at song level (``source_path``).

One LLM call per unique song; captions propagate to all 15s clip rows in the train
manifest. Resumable via append-only ``--progress-jsonl``.

Usage:
  python -m app.data_handling.music_refine_full_corpus_captions \\
    --train-jsonl data/mapping/clap_train_15s.jsonl \\
    --progress-jsonl data/mapping/clap_train_llm_full_songs.jsonl \\
    --out-jsonl data/mapping/clap_train_llm_full.jsonl
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

from app.llm_local import chat_generate, load_llm, model_is_ready
from app.self_train.jsonl_io import resolve_project_path, write_jsonl_rows

_REFINE_SYSTEM = (
    "You write short English captions for music retrieval. "
    "Do not invent artist names, song titles, or lyrics. "
    "One or two sentences."
)


def _norm_source_key(source_path: str) -> str:
    return str(resolve_project_path(source_path))


def build_refine_user_prompt(row: dict[str, Any]) -> str:
    text_orig = str(row.get("text") or "").strip()
    mood = row.get("mood")
    confidence = row.get("confidence")
    parts = [
        "Rewrite this music caption for retrieval. Keep the same facts. "
        "Explicitly mention instrumentation and mood when present "
        "(e.g. piano, vocal, relaxing).",
        f"Original caption: {text_orig}",
    ]
    if isinstance(mood, str) and mood.strip():
        parts.append(f"Mood hint: {mood.strip()}")
    if confidence is not None:
        parts.append(f"Metadata confidence: {confidence}")
    parts.append("Return only the revised caption.")
    return "\n".join(parts)


def collect_unique_songs(train_jsonl: Path) -> dict[str, dict[str, Any]]:
    """Map normalized source_path -> representative row (first clip seen)."""
    if not train_jsonl.is_file():
        raise FileNotFoundError(f"Train JSONL not found: {train_jsonl}")

    songs: dict[str, dict[str, Any]] = {}
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
            key = _norm_source_key(source_path)
            if key not in songs:
                songs[key] = dict(row)
    if not songs:
        raise ValueError(f"No songs with source_path in {train_jsonl}")
    return songs


def load_progress_map(progress_jsonl: Path) -> dict[str, str]:
    """Map normalized source_path -> LLM text from prior runs."""
    done: dict[str, str] = {}
    if not progress_jsonl.is_file():
        return done
    with progress_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            sp = row.get("source_path")
            text = row.get("text")
            if isinstance(sp, str) and isinstance(text, str) and text.strip():
                done[_norm_source_key(sp)] = text.strip()
    return done


def refine_songs(
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
                    user_prompt = build_refine_user_prompt(row)
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
                        "text_source": "llm_full_corpus",
                    }
                    out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                    out_f.flush()
                    done[key] = llm_text.strip()
                    n_written += 1
                    if n_written % 50 == 0:
                        print(
                            f"refined {n_written} new songs "
                            f"({len(done)}/{len(songs)} total)",
                            flush=True,
                        )
                except Exception as exc:
                    n_errors += 1
                    print(f"warning: refine failed for {source_path}: {exc}", flush=True)
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


def merge_full_llm_train_jsonl(
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
            orig_text = out.get("text")
            if not isinstance(orig_text, str):
                orig_text = ""

            key = _norm_source_key(source_path)
            llm_text = song_text.get(key)
            if llm_text is None:
                n_missing_song += 1
                out.setdefault("text_source", "grok")
            else:
                out["text_orig"] = orig_text
                out["text"] = llm_text
                out["text_source"] = "llm_full_corpus"
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


def check_llm_refine_complete(
    *,
    train_jsonl: Path,
    progress_jsonl: Path,
) -> dict[str, int | str]:
    """Verify song-level LLM progress covers every unique train song."""
    n_total = len(collect_unique_songs(train_jsonl))
    n_done = len(load_progress_map(progress_jsonl))
    if n_done < n_total:
        raise RuntimeError(
            f"LLM refine incomplete: {n_done}/{n_total} songs in {progress_jsonl}"
        )
    print(f"LLM refine complete: {n_done}/{n_total} songs", flush=True)
    return {
        "train_jsonl": str(train_jsonl.resolve()),
        "progress_jsonl": str(progress_jsonl.resolve()),
        "n_songs_total": n_total,
        "n_songs_done": n_done,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Full-corpus tag-aware LLM caption refine (one call per song)."
    )
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        default=settings.CLAP_TRAIN_JSONL,
    )
    parser.add_argument(
        "--progress-jsonl",
        type=Path,
        default=settings.MAPPING_DIR / "clap_train_llm_full_songs.jsonl",
        help="Append-only song-level LLM outputs (resume checkpoint).",
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=settings.MAPPING_DIR / "clap_train_llm_full.jsonl",
        help="Full clip-level train manifest with LLM captions.",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip LLM; merge progress JSONL into clip-level out-jsonl.",
    )
    parser.add_argument(
        "--check-complete-only",
        action="store_true",
        help="Exit 0 when progress covers all songs; exit 1 if incomplete.",
    )
    parser.add_argument(
        "--max-songs",
        type=int,
        default=None,
        help="Cap new LLM calls this run (for smoke tests).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    train_jsonl = args.train_jsonl.resolve()
    progress_jsonl = args.progress_jsonl.resolve()
    out_jsonl = args.out_jsonl.resolve()

    if args.check_complete_only:
        try:
            check_llm_refine_complete(
                train_jsonl=train_jsonl,
                progress_jsonl=progress_jsonl,
            )
        except RuntimeError as exc:
            print(str(exc), flush=True)
            return 1
        return 0

    if not args.merge_only:
        refine_stats = refine_songs(
            train_jsonl=train_jsonl,
            progress_jsonl=progress_jsonl,
            max_songs=args.max_songs,
            llm_max_new_tokens=args.max_new_tokens,
            llm_temperature=args.temperature,
        )
        print(json.dumps(refine_stats, ensure_ascii=False, indent=2))
        if refine_stats["n_songs_remaining"] > 0:
            print(
                f"Remaining songs: {refine_stats['n_songs_remaining']} "
                f"(re-run to continue)",
                flush=True,
            )

    merge_stats = merge_full_llm_train_jsonl(
        train_jsonl=train_jsonl,
        progress_jsonl=progress_jsonl,
        out_jsonl=out_jsonl,
    )
    print(json.dumps(merge_stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
