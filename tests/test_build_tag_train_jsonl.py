"""Tests for tag-derived train JSONL builder."""
from __future__ import annotations

import json
from pathlib import Path

from app.data_handling.music_build_tag_train_jsonl import (
    build_tag_train_jsonl,
    format_tag_text,
    load_gold_multihot_by_source,
)


def test_format_tag_text_active_and_fallback() -> None:
    tags = [("inst_piano", "piano"), ("inst_vocal", "vocal"), ("mood_relaxing", "relaxing")]
    text, src = format_tag_text(
        {"inst_piano": 1, "inst_vocal": 0, "mood_relaxing": 1},
        primary_tags=tags,
        fallback_text="music",
    )
    assert text == "piano, relaxing"
    assert src == "tag_primary"

    text2, src2 = format_tag_text(
        {"inst_piano": 0, "inst_vocal": 0, "mood_relaxing": 0},
        primary_tags=tags,
        fallback_text="music",
    )
    assert text2 == "music"
    assert src2 == "tag_fallback"


def test_build_tag_train_jsonl_gold_join_and_counts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.data_handling.music_build_tag_train_jsonl.settings.BASE_DIR",
        tmp_path,
    )

    song = "data/music_db/song_a.mp3"
    train_path = tmp_path / "data/mapping/clap_train_15s.jsonl"
    gold_path = tmp_path / "data/eval/gold_merged.jsonl"
    out_path = tmp_path / "data/mapping/clap_train_tag.jsonl"
    train_path.parent.mkdir(parents=True, exist_ok=True)
    gold_path.parent.mkdir(parents=True, exist_ok=True)

    train_path.write_text(
        json.dumps(
            {
                "audio_path": "data/music_db_15s/a__seg000.mp3",
                "text": "Grok caption",
                "source_path": song,
            }
        )
        + "\n"
        + json.dumps(
            {
                "audio_path": "data/music_db_15s/b__seg000.mp3",
                "text": "Other",
                "source_path": "data/music_db/song_b.mp3",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    gold_path.write_text(
        json.dumps(
            {
                "source_path": song,
                "human_multihot": {
                    "inst_piano": 1,
                    "inst_vocal": 0,
                    "mood_relaxing": 0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = build_tag_train_jsonl(
        train_jsonl=train_path,
        gold_jsonl=gold_path,
        out_jsonl=out_path,
        fallback_text="music",
    )
    assert summary["n_total"] == 2
    assert summary["n_tag_primary"] == 1
    assert summary["n_tag_no_gold"] == 1

    rows = [json.loads(ln) for ln in out_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert rows[0]["text"] == "piano"
    assert rows[0]["text_source"] == "tag_primary"
    assert rows[0]["text_orig"] == "Grok caption"
    assert rows[1]["text"] == "music"
    assert rows[1]["text_source"] == "tag_no_gold"


def test_load_gold_multihot_by_source(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.data_handling.music_build_tag_train_jsonl.settings.BASE_DIR",
        tmp_path,
    )
    gold_path = tmp_path / "gold.jsonl"
    sp = "data/music_db/x.mp3"
    gold_path.write_text(
        json.dumps({"source_path": sp, "human_multihot": {"inst_vocal": 1}}) + "\n",
        encoding="utf-8",
    )
    m = load_gold_multihot_by_source(gold_path)
    assert len(m) == 1
    key = str((tmp_path / sp).resolve())
    assert m[key]["inst_vocal"] == 1
