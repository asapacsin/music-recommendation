"""CLAP backbone audio feature cache."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from app.clap_audio_cache import (
    ClapAudioBackboneCache,
    normalize_audio_manifest_key,
)


def test_normalize_audio_manifest_key_strips_prefix() -> None:
    key = normalize_audio_manifest_key("data/music_db_15s/foo__seg001.mp3")
    assert key == "foo__seg001.mp3"


def test_cache_roundtrip_and_project(tmp_path: Path) -> None:
    cache = ClapAudioBackboneCache(tmp_path)
    vec = np.arange(8, dtype=np.float32)
    cache.write_entry("clip_a.mp3", vec)
    cache.save_index()

    cache2 = ClapAudioBackboneCache(tmp_path)
    assert cache2.has_key("clip_a.mp3")
    loaded = cache2.load_backbone_batch(["clip_a.mp3"], device=torch.device("cpu"))
    assert loaded.shape == (1, 8)

    class _Proj(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    class _Core(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.audio_projection = _Proj()

    class _Model:
        def __init__(self) -> None:
            self.model = _Core()

    out = cache2.project_audio_batch(_Model(), ["clip_a.mp3"], tensor_mode=True)
    assert out is not None
    assert out.shape == (1, 8)


def test_ensure_clap_audio_cache_uses_complete_cache(tmp_path: Path, monkeypatch) -> None:
    import os

    jsonl = tmp_path / "train.jsonl"
    jsonl.write_text(
        json.dumps({"audio_path": "data/music_db_15s/clip_a.mp3", "text": "test"}) + "\n",
        encoding="utf-8",
    )

    cache_dir = tmp_path / "cache"
    cache = ClapAudioBackboneCache(cache_dir)
    cache.write_entry("clip_a.mp3", np.zeros(4, dtype=np.float32))
    cache.save_index()

    monkeypatch.setattr(
        "app.data_handling.music_precompute_clap_audio_cache._load_clap_train_jsonl",
        lambda path, with_cache_keys=False: (
            [str(tmp_path / "clip_a.mp3")],
            ["test"],
            ["clip_a.mp3"],
        ),
    )

    from app.data_handling.music_precompute_clap_audio_cache import ensure_clap_audio_cache

    out = ensure_clap_audio_cache(jsonl_paths=[jsonl], cache_dir=cache_dir)
    assert out == cache_dir.resolve()
    assert os.environ.get("RAGWEB_CLAP_AUDIO_CACHE") == str(cache_dir.resolve())
