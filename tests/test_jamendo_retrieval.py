"""Tests for Jamendo public retrieval helpers (no GPU / CLAP load)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.data_handling.music_eval_jamendo_retrieval import PRIMARY_PUB_EVAL
from app.data_handling.music_eval_public_retrieval import (
    _relevance_vector,
    load_public_manifest as load_jamendo_manifest,
)
from app.data_handling.music_eval_retrieval_vs_random import _ndcg_at_k


def test_primary_pub_eval_columns():
    assert len(PRIMARY_PUB_EVAL) == 3
    for _tag_id, gold_col, qt in PRIMARY_PUB_EVAL:
        assert gold_col.startswith("gold_pub_")
        assert qt in ("piano", "vocal", "relaxing")


def test_relevance_vector():
    rows = [
        {"gold_pub_piano": 1, "gold_pub_vocal": 0, "gold_pub_relaxing": 0},
        {"gold_pub_piano": 0, "gold_pub_vocal": 1, "gold_pub_relaxing": 1},
    ]
    rel = _relevance_vector(rows, "gold_pub_piano")
    assert rel.tolist() == [1.0, 0.0]


def test_load_jamendo_manifest_filters_missing_audio(tmp_path: Path):
    audio_ok = tmp_path / "a.mp3"
    audio_ok.write_bytes(b"\x00\x01")
    missing = tmp_path / "b.mp3"
    manifest = tmp_path / "m.jsonl"
    rows = [
        {"track_id": "t1", "gold_pub_piano": 1, "audio_path": str(audio_ok)},
        {"track_id": "t2", "gold_pub_piano": 0, "audio_path": str(missing)},
        {"track_id": "t3", "gold_pub_piano": 0},
    ]
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    loaded = load_jamendo_manifest(manifest)
    assert len(loaded) == 1
    assert loaded[0]["track_id"] == "t1"


def test_ndcg_perfect_ranking():
    rel = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    order = np.array([0, 2, 1, 3])  # both positives first
    assert _ndcg_at_k(rel, order, 2) == pytest.approx(1.0, abs=1e-5)
