"""Audio path staging via RAGWEB_AUDIO_15S_ROOT."""
from __future__ import annotations

import os
from pathlib import Path

from app.init_model import _resolve_project_path


def test_resolve_project_path_uses_staged_root(tmp_path: Path, monkeypatch) -> None:
    staged = tmp_path / "staged"
    staged.mkdir()
    clip = staged / "song__seg000.mp3"
    clip.write_bytes(b"fake")

    repo = tmp_path / "repo"
    (repo / "data" / "music_db_15s").mkdir(parents=True)

    monkeypatch.setenv("RAGWEB_AUDIO_15S_ROOT", str(staged))
    monkeypatch.chdir(repo)

    from config import settings

    monkeypatch.setattr(settings, "BASE_DIR", repo, raising=False)

    resolved = _resolve_project_path("data/music_db_15s/song__seg000.mp3")
    assert resolved == clip.resolve()
