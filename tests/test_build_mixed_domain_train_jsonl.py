"""Tests for mixed-domain train JSONL builder."""
from __future__ import annotations

import json
from pathlib import Path

from app.data_handling.music_build_mixed_domain_train_jsonl import (
    build_mixed_domain_train_jsonl,
    gold_pub_to_multihot,
    load_holdout_paths,
    sample_public_rows,
)


def test_gold_pub_to_multihot() -> None:
    mh = gold_pub_to_multihot(
        {"gold_pub_piano": 1, "gold_pub_vocal": 0, "gold_pub_relaxing": 1}
    )
    assert mh == {"inst_piano": 1, "inst_vocal": 0, "mood_relaxing": 1}


def test_load_holdout_paths(tmp_path: Path) -> None:
    ap = tmp_path / "hold.mp3"
    ap.write_bytes(b"x")
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        json.dumps({"audio_path": str(ap.resolve())}) + "\n",
        encoding="utf-8",
    )
    hold = load_holdout_paths([manifest])
    assert str(ap.resolve()) in hold


def test_sample_public_rows_respects_target() -> None:
    pool = [
        {
            "audio_path": f"/a/{i}.mp3",
            "text": "piano",
            "gold_pub_piano": 1,
            "gold_pub_vocal": 0,
            "gold_pub_relaxing": 0,
        }
        for i in range(20)
    ]
    out = sample_public_rows(pool, target=5, seed=42)
    assert len(out) == 5
    paths = {r["audio_path"] for r in out}
    assert len(paths) == 5


def test_build_mixed_excludes_holdout(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.data_handling.music_build_mixed_domain_train_jsonl.settings.BASE_DIR",
        tmp_path,
    )
    anime = tmp_path / "anime.jsonl"
    anime.write_text(
        json.dumps(
            {
                "audio_path": str(tmp_path / "anime_seg.mp3"),
                "text": "piano",
                "domain": "anime",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    mtat_dir = tmp_path / "mtat"
    mtat_dir.mkdir()
    (mtat_dir / "annotations_final.csv").write_text(
        "clip_id\tmp3_path\tpiano\tcalm\n"
        "1\thold.mp3\t1\t0\n"
        "2\tok.mp3\t1\t0\n",
        encoding="utf-8",
    )
    (mtat_dir / "mp3").mkdir()
    hold_audio = mtat_dir / "mp3" / "hold.mp3"
    hold_audio.write_bytes(b"1")
    manifest = tmp_path / "hold.jsonl"
    manifest.write_text(
        json.dumps({"audio_path": str(hold_audio.resolve())}) + "\n",
        encoding="utf-8",
    )
    (mtat_dir / "mp3" / "ok.mp3").write_bytes(b"2")

    openmic_dir = tmp_path / "openmic"
    om_root = openmic_dir / "openmic-2018"
    om_root.mkdir(parents=True)
    (om_root / "openmic-2018-aggregated-labels.csv").write_text(
        "sample_key,instrument,relevance\n",
        encoding="utf-8",
    )

    out = tmp_path / "mixed.jsonl"
    hold_txt = tmp_path / "holdout.txt"
    summary = build_mixed_domain_train_jsonl(
        anime_jsonl=anime,
        holdout_manifests=[manifest],
        out_jsonl=out,
        holdout_txt=hold_txt,
        mtat_dir=mtat_dir,
        openmic_dir=openmic_dir,
        public_clip_target=10,
        mix_ratio=None,
        seed=42,
    )
    assert summary["n_anime"] == 1
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    public_paths = {r["audio_path"] for r in rows if r.get("domain") == "mtat"}
    assert str((mtat_dir / "mp3" / "hold.mp3").resolve()) not in public_paths
    assert any("ok.mp3" in p for p in public_paths)


def test_build_mixed_preserves_grok_anime_text(tmp_path: Path, monkeypatch) -> None:
    """Grok-style anime captions pass through unchanged into mixed JSONL."""
    monkeypatch.setattr(
        "app.data_handling.music_build_mixed_domain_train_jsonl.settings.BASE_DIR",
        tmp_path,
    )
    grok_text = (
        "Off-vocal mix featuring an energetic and atmospheric electronic track."
    )
    anime = tmp_path / "anime_grok.jsonl"
    anime.write_text(
        json.dumps(
            {
                "audio_path": str(tmp_path / "anime_seg.mp3"),
                "text": grok_text,
                "source_path": "data/music_db/example.mp3",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    mtat_dir = tmp_path / "mtat"
    mtat_dir.mkdir()
    (mtat_dir / "annotations_final.csv").write_text(
        "clip_id\tmp3_path\tpiano\tcalm\n",
        encoding="utf-8",
    )
    openmic_dir = tmp_path / "openmic"
    om_root = openmic_dir / "openmic-2018"
    om_root.mkdir(parents=True)
    (om_root / "openmic-2018-aggregated-labels.csv").write_text(
        "sample_key,instrument,relevance\n",
        encoding="utf-8",
    )

    out = tmp_path / "grok_mixed.jsonl"
    hold_txt = tmp_path / "holdout.txt"
    build_mixed_domain_train_jsonl(
        anime_jsonl=anime,
        holdout_manifests=[],
        out_jsonl=out,
        holdout_txt=hold_txt,
        mtat_dir=mtat_dir,
        openmic_dir=openmic_dir,
        public_clip_target=0,
        mix_ratio=0.5,
        seed=42,
    )
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    anime_rows = [r for r in rows if r.get("domain") == "anime"]
    assert len(anime_rows) == 1
    assert anime_rows[0]["text"] == grok_text
