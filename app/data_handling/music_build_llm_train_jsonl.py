"""
Build a train JSONL with LLM-refined captions swapped in-place (one row per clip).

Used for LLM vs original caption ablation: same audio paths as ``clap_train_15s.jsonl``,
with ``text`` replaced where a gate-passed refined caption exists.
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

from app.self_train.jsonl_io import resolve_project_path, write_jsonl_rows


def _path_keys(audio_path: str) -> tuple[str, str]:
    """Resolved absolute path and basename for join lookups."""
    resolved = str(resolve_project_path(audio_path))
    return resolved, Path(audio_path).name


def load_refined_text_map(refined_jsonl: Path) -> dict[str, str]:
    """
    Map audio path (resolved or basename) -> LLM ``text``.

    Later keys overwrite earlier; basename is a fallback when train uses relative paths.
    """
    if not refined_jsonl.is_file():
        raise FileNotFoundError(f"Refined JSONL not found: {refined_jsonl}")

    by_resolved: dict[str, str] = {}
    by_basename: dict[str, str] = {}

    with refined_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            audio_value = row.get("audio_path")
            text = row.get("text")
            if not isinstance(audio_value, str) or not isinstance(text, str) or not text.strip():
                continue
            resolved, basename = _path_keys(audio_value)
            by_resolved[resolved] = text.strip()
            by_basename[basename] = text.strip()

    # Prefer resolved keys; basename only for rows not hit via resolved lookup.
    merged: dict[str, str] = dict(by_basename)
    merged.update(by_resolved)
    return merged


def lookup_refined_text(
    audio_path: str,
    refined_map: dict[str, str],
) -> str | None:
    resolved, basename = _path_keys(audio_path)
    if resolved in refined_map:
        return refined_map[resolved]
    if basename in refined_map:
        return refined_map[basename]
    return None


def build_llm_train_jsonl(
    *,
    train_jsonl: Path,
    refined_jsonl: Path,
    out_jsonl: Path,
) -> dict[str, Any]:
    if not train_jsonl.is_file():
        raise FileNotFoundError(f"Train JSONL not found: {train_jsonl}")

    refined_map = load_refined_text_map(refined_jsonl)
    out_rows: list[dict[str, Any]] = []
    n_replaced = 0

    with train_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            audio_value = row.get("audio_path")
            if not isinstance(audio_value, str) or not audio_value.strip():
                continue

            out = dict(row)
            orig_text = out.get("text")
            if not isinstance(orig_text, str):
                orig_text = ""

            llm_text = lookup_refined_text(audio_value, refined_map)
            if llm_text is not None:
                out["text_orig"] = orig_text
                out["text"] = llm_text
                out["text_source"] = "llm_refined"
                n_replaced += 1
            else:
                out.setdefault("text_source", "grok")

            out_rows.append(out)

    if not out_rows:
        raise ValueError(f"No train rows read from {train_jsonl}")

    write_jsonl_rows(out_jsonl, out_rows)
    return {
        "train_jsonl": str(train_jsonl.resolve()),
        "refined_jsonl": str(refined_jsonl.resolve()),
        "out_jsonl": str(out_jsonl.resolve()),
        "n_total": len(out_rows),
        "n_replaced": n_replaced,
        "n_refined_in_map": len(refined_map),
    }


def main() -> int:
    default_refined = (
        settings.SELF_TRAIN_DATA_DIR / "thesis_self_v2" / "iter_0" / "refined.jsonl"
    )
    parser = argparse.ArgumentParser(
        description="Merge gate-passed LLM captions into clap_train JSONL (in-place text swap)."
    )
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        default=settings.CLAP_TRAIN_JSONL,
        help=f"Base train manifest (default: {settings.CLAP_TRAIN_JSONL}).",
    )
    parser.add_argument(
        "--refined-jsonl",
        type=Path,
        default=default_refined,
        help=f"Gate-passed refined rows (default: {default_refined}).",
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=settings.MAPPING_DIR / "clap_train_llm_gated_iter0.jsonl",
        help="Output manifest with LLM text swapped where refined rows exist.",
    )
    args = parser.parse_args()

    summary = build_llm_train_jsonl(
        train_jsonl=args.train_jsonl.resolve(),
        refined_jsonl=args.refined_jsonl.resolve(),
        out_jsonl=args.out_jsonl.resolve(),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
