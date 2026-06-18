"""Resume helpers for train_clap_multiseed."""
from __future__ import annotations

import json
from pathlib import Path

from app.train_clap_multiseed import _seed_training_complete


def test_seed_training_complete_flag_file(tmp_path: Path) -> None:
    log_dir = tmp_path / "seed_42"
    log_dir.mkdir()
    assert not _seed_training_complete(log_dir)
    (log_dir / "training_complete.json").write_text("{}", encoding="utf-8")
    assert _seed_training_complete(log_dir)


def test_seed_training_complete_legacy_metrics(tmp_path: Path) -> None:
    log_dir = tmp_path / "seed_42"
    log_dir.mkdir()
    metrics = log_dir / "metrics.jsonl"
    metrics.write_text(
        json.dumps({"epoch": 6, "val_similarity": 0.58, "checkpoint_metric": 0.58}) + "\n",
        encoding="utf-8",
    )
    assert _seed_training_complete(log_dir)


def test_seed_training_complete_partial_epoch_not_done(tmp_path: Path) -> None:
    log_dir = tmp_path / "seed_43"
    log_dir.mkdir()
    metrics = log_dir / "metrics.jsonl"
    metrics.write_text(
        json.dumps({"epoch": 1, "val_similarity": 0.31, "checkpoint_metric": 0.31}) + "\n",
        encoding="utf-8",
    )
    assert not _seed_training_complete(log_dir)
