"""Tests for LLM train JSONL merge (path-normalized swap)."""
from __future__ import annotations

import json
from pathlib import Path

from app.data_handling.music_build_llm_train_jsonl import (
    build_llm_train_jsonl,
    load_refined_text_map,
    lookup_refined_text,
)


def test_lookup_refined_text_resolved_and_basename(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.data_handling.music_build_llm_train_jsonl.settings.BASE_DIR",
        tmp_path,
    )
    rel = "data/music_db_15s/track__seg000.mp3"
    refined = {
        str((tmp_path / rel).resolve()): "LLM caption",
        "other__seg001.mp3": "By basename only",
    }
    assert lookup_refined_text(rel, refined) == "LLM caption"
    assert lookup_refined_text("other__seg001.mp3", refined) == "By basename only"
    assert lookup_refined_text("missing.mp3", refined) is None


def test_build_llm_train_jsonl_swaps_text_and_keeps_count(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.data_handling.music_build_llm_train_jsonl.settings.BASE_DIR",
        tmp_path,
    )

    train_path = tmp_path / "data/mapping/clap_train_15s.jsonl"
    refined_path = tmp_path / "data/self_train/refined.jsonl"
    out_path = tmp_path / "data/mapping/clap_train_llm.jsonl"
    train_path.parent.mkdir(parents=True, exist_ok=True)
    refined_path.parent.mkdir(parents=True, exist_ok=True)

    rel_audio = "data/music_db_15s/foo__seg000.mp3"
    abs_audio = str((tmp_path / rel_audio).resolve())

    train_path.write_text(
        json.dumps({"audio_path": rel_audio, "text": "Original A"})
        + "\n"
        + json.dumps({"audio_path": "data/music_db_15s/bar__seg000.mp3", "text": "Original B"})
        + "\n",
        encoding="utf-8",
    )
    refined_path.write_text(
        json.dumps({"audio_path": abs_audio, "text": "LLM refined A"})
        + "\n",
        encoding="utf-8",
    )

    summary = build_llm_train_jsonl(
        train_jsonl=train_path,
        refined_jsonl=refined_path,
        out_jsonl=out_path,
    )

    assert summary["n_total"] == 2
    assert summary["n_replaced"] == 1

    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["text"] == "LLM refined A"
    assert rows[0]["text_orig"] == "Original A"
    assert rows[0]["text_source"] == "llm_refined"
    assert rows[1]["text"] == "Original B"


def test_load_refined_text_map_from_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.data_handling.music_build_llm_train_jsonl.settings.BASE_DIR",
        tmp_path,
    )
    refined_path = tmp_path / "refined.jsonl"
    abs_p = str((tmp_path / "data/music_db_15s/x__seg000.mp3").resolve())
    refined_path.write_text(
        json.dumps({"audio_path": abs_p, "text": "Refined text"}) + "\n",
        encoding="utf-8",
    )
    m = load_refined_text_map(refined_path)
    assert lookup_refined_text("data/music_db_15s/x__seg000.mp3", m) == "Refined text"
