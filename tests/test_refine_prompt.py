"""Tests for LLM refine prompt construction (no model load)."""
from __future__ import annotations

from app.self_train.refine import build_refine_user_prompt


def test_build_refine_user_prompt_includes_original_text():
    row = {
        "text": "Dreamy atmosphere.",
        "text_orig": "Dreamy atmosphere.",
        "mood": "relaxing",
        "confidence": 0.65,
        "source_path": "data/music_db/track.mp3",
        "sim": 0.12,
    }
    prompt = build_refine_user_prompt(row)
    assert "Dreamy atmosphere." in prompt
    assert "relaxing" in prompt
    assert "0.65" in prompt
    assert "track.mp3" in prompt
    assert "0.12" in prompt
