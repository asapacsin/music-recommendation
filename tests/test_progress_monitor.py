"""Tests for read-only progress monitor."""

from __future__ import annotations

import json
from pathlib import Path

from app.progress_monitor import (
    build_snapshot,
    public_ood_pipeline_actions,
    render_markdown,
    write_outputs,
)


def _touch(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_snapshot_question_d_units(tmp_path: Path) -> None:
    repo = tmp_path
    _touch(repo / "data/mapping/clap_train_tag.jsonl", "x\n" * 65041)
    _touch(repo / "data/mapping/clap_train_tag_llm.jsonl", "x\n" * 65041)
    _touch(repo / "data/mapping/clap_train_tag_llm_songs.jsonl", "x\n" * 3440)
    cache = repo / "data/embeddings_cache/clap_backbone/music_audioset_epoch_15_esc_90.14"
    _touch(cache / "index.json", "{}")

    snap = build_snapshot(repo)
    units = {u["unit"]: u for u in snap["question_d_units"]}
    assert units["0"]["state"] == "done"
    assert units["1"]["state"] == "done"
    assert units["2"]["state"] == "done"
    assert units["3"]["state"] in {"next", "pending", "running"}


def test_seed_complete_detection(tmp_path: Path) -> None:
    repo = tmp_path
    seed_dir = repo / "model/clap/finetune/thesis_tag_only/seed_42"
    log_dir = repo / "data/log/finetune_runs/thesis_tag_only/seed_42"
    _touch(seed_dir / "best_model.pt")
    _touch(
        log_dir / "training_complete.json",
        json.dumps({"seed": 42, "best_epoch": 5, "best_similarity": 0.9079}) + "\n",
    )
    _touch(
        log_dir / "metrics.jsonl",
        json.dumps({"epoch": 7, "val_similarity": 0.8177}) + "\n",
    )

    snap = build_snapshot(repo)
    d = next(q for q in snap["thesis_questions"] if q["id"] == "D")
    tag_only = next(r for r in d["runs"] if r["run_id"] == "thesis_tag_only")
    s42 = next(s for s in tag_only["seeds"] if s["seed"] == 42)
    assert s42["checkpoint"] is True
    assert s42["training_complete"] is True
    assert s42["best_epoch"] == 5
    assert s42["best_val_similarity"] == 0.9079


def test_render_markdown_includes_training_recipe(tmp_path: Path) -> None:
    repo = tmp_path
    _touch(
        repo / "data/eval/llm_ablation/train_params.json",
        json.dumps(
            {
                "val_jsonl": "data/mapping/clap_val_15s.jsonl",
                "num_epochs": 20,
                "batch_size": 32,
                "early_stopping": {"patience": 2, "min_epochs": 5},
            }
        ),
    )
    _touch(repo / "data/mapping/clap_train_tag.jsonl", "x\n" * 100)
    snap = build_snapshot(repo)
    md = render_markdown(snap)
    assert "## Question D — training recipe" in md
    assert "thesis_tag_only" in md
    assert "best** val_similarity" in md or "ep5 val=" in md


def test_public_ood_manifest_readiness(tmp_path: Path) -> None:
    repo = tmp_path
    manifest = repo / "data/eval/jamendo_five_tag_manifest.jsonl"
    audio = repo / "data/public_eval/jamendo/audio_five_tag/a/1.mp3"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"x")
    _touch(
        manifest,
        json.dumps({"audio_path": str(audio.resolve())}) + "\n"
        + json.dumps({"audio_path": "/missing/file.mp3"}) + "\n",
    )
    snap = build_snapshot(repo)
    pub = snap["public_ood"]
    jamendo = next(d for d in pub["datasets"] if d["dataset"] == "jamendo")
    assert jamendo["manifest"]["n_audio_ready"] == 1
    assert jamendo["manifest"]["n_rows"] == 2
    assert jamendo["prep_state"] == "partial"


def test_render_markdown_includes_public_ood(tmp_path: Path) -> None:
    repo = tmp_path
    _touch(repo / "data/eval/download_status_snapshot.json", '{"updated_utc":"2026-01-01T00:00:00Z"}')
    snap = build_snapshot(repo)
    md = render_markdown(snap)
    assert "## Public OOD pipeline" in md
    assert "PUBLIC_OOD_EVAL.md" in md
    assert "Eval matrix" in md


def test_public_ood_pipeline_units(tmp_path: Path) -> None:
    repo = tmp_path
    manifest = repo / "data/eval/jamendo_five_tag_manifest.jsonl"
    audio = repo / "data/public_eval/jamendo/audio_five_tag/a/1.mp3"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"x")
    _touch(manifest, json.dumps({"audio_path": str(audio.resolve())}) + "\n")
    snap = build_snapshot(repo)
    pub = snap["public_ood"]
    units = {u["unit"]: u for u in pub["pipeline_units"]}
    assert units["0"]["state"] == "done"
    assert units["3"]["state"] in {"next", "pending"}
    assert "next_commands" in pub
    assert pub.get("eval_matrix")


def test_public_ood_pipeline_actions(tmp_path: Path) -> None:
    repo = tmp_path
    manifest = repo / "data/eval/jamendo_five_tag_manifest.jsonl"
    audio = repo / "data/public_eval/jamendo/audio_five_tag/a/1.mp3"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"x")
    _touch(manifest, json.dumps({"audio_path": str(audio.resolve())}) + "\n")
    plan = public_ood_pipeline_actions(repo)
    assert "jamendo" in plan["datasets_ready"]
    assert any(a["type"] == "eval" for a in plan["actions"])


def test_resolve_openmic_audio_path(tmp_path: Path) -> None:
    from app.data_handling.music_build_openmic_manifest import resolve_openmic_audio

    openmic = tmp_path / "openmic"
    audio = openmic / "openmic-2018" / "audio" / "000" / "000046_3840.ogg"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"x")
    got = resolve_openmic_audio(openmic, "000046_3840")
    assert got == audio.resolve()


def test_write_outputs(tmp_path: Path) -> None:
    repo = tmp_path
    _touch(repo / "data/mapping/clap_train_tag.jsonl", "line\n")
    snapshot = write_outputs(repo)
    assert (repo / "data/eval/progress_snapshot.json").is_file()
    md = (repo / "docs/PROGRESS.md").read_text(encoding="utf-8")
    assert "Progress monitor" in md
    assert snapshot["updated_utc"] in md


def test_render_markdown_includes_slurm_tail(tmp_path: Path) -> None:
    repo = tmp_path
    _touch(
        repo / "slurm-999.out",
        "=== Fine-tune TAG-ONLY: thesis_tag_only ===\nEpoch 1\nDone.\n",
    )
    snap = build_snapshot(repo)
    md = render_markdown(snap)
    assert "Job `999`" in md
    assert "ft_tag_only" in md
