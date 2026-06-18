"""Unit tests for CLAP refinement gate (mocked embeddings)."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np

from app.self_train.gate import passes_gate


class _FakeModel:
    pass


@patch("app.self_train.gate.diagonal_sim")
@patch("app.self_train.gate.text_cosine")
def test_passes_gate_accepts_when_sim_improves_and_text_stable(mock_text_cos, mock_diag_sim):
    mock_diag_sim.side_effect = [0.2, 0.5]
    mock_text_cos.return_value = 0.95

    ok, diag = passes_gate(
        "/fake/audio.mp3",
        "old caption",
        "new caption",
        _FakeModel(),
        min_sim_gain=0.0,
        min_text_cos=0.85,
    )

    assert ok is True
    assert diag["reject_reason"] is None
    assert diag["sim_new"] > diag["sim_old"]


@patch("app.self_train.gate.diagonal_sim")
@patch("app.self_train.gate.text_cosine")
def test_passes_gate_rejects_drift(mock_text_cos, mock_diag_sim):
    mock_diag_sim.side_effect = [0.2, 0.5]
    mock_text_cos.return_value = 0.5

    ok, diag = passes_gate(
        "/fake/audio.mp3",
        "old",
        "new",
        _FakeModel(),
        min_text_cos=0.85,
    )

    assert ok is False
    assert diag["reject_reason"] == "drift"


@patch("app.self_train.gate.diagonal_sim")
def test_passes_gate_rejects_sim(mock_diag_sim):
    mock_diag_sim.side_effect = [0.5, 0.4]

    with patch("app.self_train.gate.text_cosine", return_value=0.99):
        ok, diag = passes_gate(
            "/fake/audio.mp3",
            "old",
            "new",
            _FakeModel(),
            min_sim_gain=0.0,
        )

    assert ok is False
    assert diag["reject_reason"] == "sim"
