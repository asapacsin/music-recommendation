"""Tests for public OOD retrieval and manifest builders."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from app.data_handling.music_build_mtat_manifest import (
    MTAT_RELAX_COLS,
    MTAT_VOCAL_COLS,
    build_mtat_manifest,
)
from app.data_handling.music_build_openmic_manifest import build_openmic_manifest
from app.data_handling.music_eval_public_retrieval import (
    PRIMARY_BY_DATASET,
    _relevance_vector,
    load_public_manifest,
)


def test_primary_by_dataset_openmic_skips_relaxing():
    tags = [t[0] for t in PRIMARY_BY_DATASET["openmic"]]
    assert "inst_piano" in tags
    assert "inst_vocal" in tags
    assert "mood_relaxing" not in tags


def test_load_public_manifest(tmp_path: Path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"\x00\x01")
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        json.dumps({"audio_path": str(audio), "gold_pub_piano": 1}) + "\n",
        encoding="utf-8",
    )
    assert len(load_public_manifest(manifest)) == 1


def test_build_mtat_manifest_from_tiny_tsv(tmp_path: Path):
    mtat = tmp_path / "mtat"
    mtat.mkdir()
    ann = mtat / "annotations_final.csv"
    fields = ["clip_id", "piano", "vocals", "calm", "mp3_path"]
    with ann.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerow(
            {
                "clip_id": "c1",
                "piano": "1",
                "vocals": "0",
                "calm": "0",
                "mp3_path": "a/x.mp3",
            }
        )
        w.writerow(
            {
                "clip_id": "c2",
                "piano": "0",
                "vocals": "1",
                "calm": "1",
                "mp3_path": "b/y.mp3",
            }
        )

    rows = build_mtat_manifest(mtat_dir=mtat, max_per_tag=10, seed=42)
    assert len(rows) == 2
    assert rows[0]["gold_pub_piano"] == 1 or rows[1]["gold_pub_piano"] == 1
    assert any(r["gold_pub_vocal"] == 1 for r in rows)
    assert any(r["gold_pub_relaxing"] == 1 for r in rows)


def test_build_openmic_manifest_long_format(tmp_path: Path):
    root = tmp_path / "openmic" / "openmic-2018"
    root.mkdir(parents=True)
    labels = root / "openmic-2018-aggregated-labels.csv"
    with labels.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_key", "instrument", "relevance"])
        w.writeheader()
        w.writerow({"sample_key": "000001_1000", "instrument": "piano", "relevance": "0.8"})
        w.writerow({"sample_key": "000002_2000", "instrument": "voice", "relevance": "0.9"})

    rows = build_openmic_manifest(
        openmic_dir=tmp_path / "openmic",
        max_per_tag=10,
        seed=42,
        label_threshold=0.5,
    )
    assert len(rows) == 2
    assert rows[0]["gold_pub_relaxing"] == 0


def test_relevance_vector():
    rows = [{"gold_pub_piano": 1}, {"gold_pub_piano": 0}]
    rel = _relevance_vector(rows, "gold_pub_piano")
    assert rel.tolist() == [1.0, 0.0]
