"""Tests for full-corpus LLM merge and caption FAISS row collection."""
from __future__ import annotations

import json
from pathlib import Path

from app.data_handling.music_build_retrieval_faiss import collect_caption_rows_from_jsonl
from app.data_handling.music_refine_full_corpus_captions import (
    check_llm_refine_complete,
    collect_unique_songs,
    merge_full_llm_train_jsonl,
)


def test_collect_unique_songs_dedupes_by_source_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.data_handling.music_refine_full_corpus_captions.settings.BASE_DIR",
        tmp_path,
    )
    train = tmp_path / "train.jsonl"
    train.parent.mkdir(parents=True, exist_ok=True)
    sp = "data/music_db/song_a.mp3"
    train.write_text(
        json.dumps(
            {
                "audio_path": "data/music_db_15s/song_a__seg000.mp3",
                "source_path": sp,
                "text": "Original",
                "confidence": 0.5,
            }
        )
        + "\n"
        + json.dumps(
            {
                "audio_path": "data/music_db_15s/song_a__seg001.mp3",
                "source_path": sp,
                "text": "Original",
                "confidence": 0.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    songs = collect_unique_songs(train)
    assert len(songs) == 1


def test_merge_full_llm_train_jsonl_propagates_song_caption(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.data_handling.music_refine_full_corpus_captions.settings.BASE_DIR",
        tmp_path,
    )
    train = tmp_path / "data/mapping/clap_train_15s.jsonl"
    progress = tmp_path / "data/mapping/songs.jsonl"
    out = tmp_path / "data/mapping/clap_train_llm_full.jsonl"
    train.parent.mkdir(parents=True, exist_ok=True)

    sp = "data/music_db/foo.mp3"
    train.write_text(
        json.dumps(
            {
                "audio_path": "data/music_db_15s/foo__seg000.mp3",
                "source_path": sp,
                "text": "Grok A",
            }
        )
        + "\n"
        + json.dumps(
            {
                "audio_path": "data/music_db_15s/foo__seg001.mp3",
                "source_path": sp,
                "text": "Grok A",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    progress.write_text(
        json.dumps({"source_path": sp, "text": "LLM rewritten"}) + "\n",
        encoding="utf-8",
    )

    summary = merge_full_llm_train_jsonl(
        train_jsonl=train,
        progress_jsonl=progress,
        out_jsonl=out,
    )
    assert summary["n_total"] == 2
    assert summary["n_replaced"] == 2
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert all(r["text"] == "LLM rewritten" for r in rows)
    assert rows[0]["text_source"] == "llm_full_corpus"


def test_check_llm_refine_complete_ok(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.data_handling.music_refine_full_corpus_captions.settings.BASE_DIR",
        tmp_path,
    )
    train = tmp_path / "train.jsonl"
    progress = tmp_path / "songs.jsonl"
    train.parent.mkdir(parents=True, exist_ok=True)
    sp = "data/music_db/foo.mp3"
    train.write_text(
        json.dumps({"source_path": sp, "text": "Grok", "audio_path": "a.mp3"}) + "\n",
        encoding="utf-8",
    )
    progress.write_text(
        json.dumps({"source_path": sp, "text": "LLM"}) + "\n",
        encoding="utf-8",
    )
    stats = check_llm_refine_complete(train_jsonl=train, progress_jsonl=progress)
    assert stats["n_songs_total"] == 1
    assert stats["n_songs_done"] == 1


def test_check_llm_refine_complete_incomplete(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.data_handling.music_refine_full_corpus_captions.settings.BASE_DIR",
        tmp_path,
    )
    train = tmp_path / "train.jsonl"
    progress = tmp_path / "songs.jsonl"
    train.parent.mkdir(parents=True, exist_ok=True)
    train.write_text(
        json.dumps({"source_path": "data/music_db/a.mp3", "text": "A"}) + "\n"
        + json.dumps({"source_path": "data/music_db/b.mp3", "text": "B"}) + "\n",
        encoding="utf-8",
    )
    progress.write_text(
        json.dumps({"source_path": "data/music_db/a.mp3", "text": "LLM A"}) + "\n",
        encoding="utf-8",
    )
    import pytest

    with pytest.raises(RuntimeError, match="LLM refine incomplete: 1/2"):
        check_llm_refine_complete(train_jsonl=train, progress_jsonl=progress)


def test_collect_caption_rows_one_per_song(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.data_handling.music_build_retrieval_faiss.settings.BASE_DIR",
        tmp_path,
    )
    jsonl = tmp_path / "train.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "source_path": "data/music_db/track.mp3",
                "text": "Caption",
                "confidence": 0.8,
            }
        )
        + "\n"
        + json.dumps(
            {
                "source_path": "data/music_db/track.mp3",
                "text": "Other seg",
                "confidence": 0.8,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = collect_caption_rows_from_jsonl(jsonl)
    assert len(rows) == 1
    assert rows[0]["audio"] == "track.mp3"
    assert rows[0]["text"] == "Caption"
