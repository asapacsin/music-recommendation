"""Read-only thesis / cluster progress monitor.

Writes ``data/eval/progress_snapshot.json`` and ``docs/PROGRESS.md``.
Refresh: ``bash scripts/refresh_progress.sh`` or ``python -m app.progress_monitor``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_SEEDS = (42, 43, 44)

PUBLIC_DATASETS: tuple[tuple[str, str, str], ...] = (
    ("jamendo", "data/eval/jamendo_five_tag_manifest.jsonl", "data/eval/jamendo_public"),
    ("mtat", "data/eval/mtat_manifest.jsonl", "data/eval/mtat_public"),
    ("openmic", "data/eval/openmic_manifest.jsonl", "data/eval/openmic_public"),
)

DEFAULT_PUBLIC_ARMS = ("pretrained", "thesis_ft_v1", "thesis_tag_only", "thesis_tag_llm")

REPO = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = REPO / "data/eval/progress_snapshot.json"
MARKDOWN_PATH = REPO / "docs/PROGRESS.md"


@dataclass
class SeedStatus:
    seed: int
    checkpoint: bool
    training_complete: bool
    best_epoch: int | None = None
    best_val_similarity: float | None = None


@dataclass
class RunStatus:
    run_id: str
    seeds: list[SeedStatus]
    n_complete: int
    n_total: int

    @property
    def status(self) -> str:
        if self.n_complete == self.n_total and self.n_total > 0:
            return "done"
        if self.n_complete > 0:
            return "partial"
        return "pending"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _count_lines(path: Path) -> int | None:
    if not path.is_file():
        return None
    n = 0
    with path.open(encoding="utf-8", errors="replace") as fh:
        for _ in fh:
            n += 1
    return n


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _best_training_metric(log_seed_dir: Path) -> tuple[int | None, float | None]:
    """Best (epoch, val_similarity) from training_complete.json or max in metrics.jsonl."""
    complete = _read_json(log_seed_dir / "training_complete.json")
    if complete:
        sim = complete.get("best_similarity")
        if sim is not None:
            ep = complete.get("best_epoch")
            return (int(ep) if ep is not None else None, float(sim))

    metrics = log_seed_dir / "metrics.jsonl"
    if not metrics.is_file():
        return None, None
    best_ep: int | None = None
    best_val: float | None = None
    for line in metrics.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        val = rec.get("val_similarity")
        if val is None:
            continue
        val_f = float(val)
        if best_val is None or val_f > best_val:
            best_val = val_f
            ep = rec.get("epoch")
            best_ep = int(ep) if ep is not None else best_ep
    return best_ep, best_val


def _seed_status(repo: Path, run_id: str, seed: int) -> SeedStatus:
    model_dir = repo / "model/clap/finetune" / run_id / f"seed_{seed}"
    log_dir = repo / "data/log/finetune_runs" / run_id / f"seed_{seed}"
    ckpt = (model_dir / "best_model.pt").is_file() or (log_dir / "best_model.pt").is_file()
    complete = (log_dir / "training_complete.json").is_file()
    best_ep, best_val = _best_training_metric(log_dir)
    if not complete and ckpt and best_val is not None:
        complete = True
    return SeedStatus(
        seed=seed,
        checkpoint=ckpt,
        training_complete=complete,
        best_epoch=best_ep,
        best_val_similarity=best_val,
    )


def _run_status(repo: Path, run_id: str, seeds: tuple[int, ...] = DEFAULT_SEEDS) -> RunStatus:
    statuses = [_seed_status(repo, run_id, s) for s in seeds]
    n_complete = sum(1 for s in statuses if s.checkpoint and s.training_complete)
    return RunStatus(run_id=run_id, seeds=statuses, n_complete=n_complete, n_total=len(seeds))


def _rel(repo: Path, path: Path) -> str:
    try:
        return str(path.relative_to(repo))
    except ValueError:
        return str(path)


def _artifact(repo: Path, path: Path) -> dict[str, Any]:
    return {
        "path": _rel(repo, path),
        "exists": path.is_file(),
        "lines": _count_lines(path) if path.is_file() else None,
        "bytes": path.stat().st_size if path.is_file() else None,
    }


def _parse_slurm_log(repo: Path, path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    job_id = path.stem.replace("slurm-", "")
    markers = {
        "tag_jsonl_build": "=== Build tag train JSONL ===" in text,
        "llm_gen": "n_songs_done" in text or "Tag → LLM" in text or "tag_llm" in text.lower(),
        "fine_tune_tag_only": "=== Fine-tune TAG-ONLY: thesis_tag_only ===" in text,
        "fine_tune_tag_llm": "thesis_tag_llm ===" in text and "Fine-tune TAG" in text,
        "public_download": "public-dl Slurm job" in text or "public_eval backend driver" in text,
        "public_eval": "public OOD" in text.lower() or "music_eval_public_retrieval" in text,
        "skip_train": "SKIP_TRAIN=1" in text,
        "skip_eval": "SKIP_EVAL=1" in text,
        "done": "\nDone.\n" in text or text.rstrip().endswith("Done."),
        "error": "ERROR:" in text or "Traceback (most recent call last)" in text,
    }
    phase = "unknown"
    if markers["fine_tune_tag_llm"]:
        phase = "ft_tag_llm"
    elif markers["fine_tune_tag_only"]:
        phase = "ft_tag_only"
    elif markers["public_eval"]:
        phase = "public_ood_eval"
    elif markers["public_download"]:
        phase = "public_ood_download"
    elif markers["llm_gen"] and not markers["fine_tune_tag_only"]:
        phase = "llm_corpus_gen"
    elif markers["skip_train"] and markers["done"]:
        phase = "skipped_train"

    tail = [ln for ln in text.splitlines() if ln.strip()][-8:]

    state = "completed" if markers["done"] and not markers["error"] else "failed" if markers["error"] else "running"
    if markers["skip_train"] and markers["done"] and not markers["fine_tune_tag_only"] and not markers["fine_tune_tag_llm"]:
        state = "skipped"

    return {
        "job_id": job_id,
        "log": _rel(repo, path),
        "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "phase": phase,
        "state": state,
        "markers": markers,
        "tail": tail,
    }


def _thesis_question_status(repo: Path) -> list[dict[str, Any]]:
    checks = [
        {
            "id": "A",
            "label": "Fine-tune vs pretrained",
            "result_glob": "data/eval/retrieval_vs_random_matrix*.csv",
            "run_ids": [],
            "report": None,
        },
        {
            "id": "B",
            "label": "Grok vs LLM captions",
            "result_glob": None,
            "run_ids": ["thesis_llm_full_llm"],
            "report": "data/eval/llm_full_ablation/REPORT.md",
        },
        {
            "id": "C",
            "label": "Self-train loop",
            "result_glob": None,
            "run_ids": [],
            "report": None,
            "extra": "docs/agent_runs/20260526_self_train_v2/REVIEW.md",
        },
        {
            "id": "D",
            "label": "Tag-only vs tag→LLM",
            "result_glob": None,
            "run_ids": ["thesis_tag_only", "thesis_tag_llm"],
            "report": "data/eval/tag_llm_ablation/REPORT.md",
        },
    ]
    rows: list[dict[str, Any]] = []
    for item in checks:
        report_path = (repo / item["report"]) if item.get("report") else None
        extra_raw = item.get("extra")
        extra = (repo / extra_raw) if extra_raw else None
        glob_hits: list[str] = []
        if item.get("result_glob"):
            glob_hits = sorted(
                str(p.relative_to(repo)) for p in repo.glob(item["result_glob"])
            )
        runs = [_run_status(repo, rid) for rid in item.get("run_ids", [])]
        report_ok = report_path.is_file() if report_path else False
        if item["id"] == "A":
            ft = _run_status(repo, "thesis_ft_v1")
            status = "done" if glob_hits and ft.n_complete == ft.n_total else "partial" if glob_hits or ft.n_complete else "pending"
        elif item["id"] == "C":
            status = "done" if extra and extra.is_file() else "pending"
        elif item["id"] == "D":
            tag_only = _run_status(repo, "thesis_tag_only")
            tag_llm = _run_status(repo, "thesis_tag_llm")
            if report_ok:
                status = "done"
            elif tag_only.n_complete == tag_only.n_total and tag_llm.n_complete == tag_llm.n_total:
                status = "eval_pending"
            elif tag_only.n_complete > 0 or tag_llm.n_complete > 0:
                status = "running"
            else:
                status = "pending"
        else:
            status = "done" if report_ok else "partial" if any(r.n_complete for r in runs) else "pending"

        rows.append(
            {
                "id": item["id"],
                "label": item["label"],
                "status": status,
                "report": str(report_path.relative_to(repo)) if report_path else None,
                "report_exists": report_ok,
                "artifacts": glob_hits,
                "runs": [
                    {
                        "run_id": r.run_id,
                        "status": r.status,
                        "n_complete": r.n_complete,
                        "n_total": r.n_total,
                        "seeds": [asdict(s) for s in r.seeds],
                    }
                    for r in runs
                ],
            }
        )
    return rows


def _question_d_units(repo: Path) -> list[dict[str, Any]]:
    tag_jsonl = repo / "data/mapping/clap_train_tag.jsonl"
    tag_llm_jsonl = repo / "data/mapping/clap_train_tag_llm.jsonl"
    tag_songs = repo / "data/mapping/clap_train_tag_llm_songs.jsonl"
    cache_index = (
        repo / "data/embeddings_cache/clap_backbone/music_audioset_epoch_15_esc_90.14/index.json"
    )
    tag_only = _run_status(repo, "thesis_tag_only")
    tag_llm = _run_status(repo, "thesis_tag_llm")
    report = repo / "data/eval/tag_llm_ablation/REPORT.md"

    n_songs = _count_lines(tag_songs) or 0
    units = [
        ("0", "Tag JSONL", tag_jsonl.is_file() and (_count_lines(tag_jsonl) or 0) > 60000),
        ("1", "Llama tag→text", tag_llm_jsonl.is_file() and n_songs >= 3440),
        ("2", "Audio cache", cache_index.is_file()),
        (
            "3",
            "FT thesis_tag_only (seeds 42–44)",
            tag_only.n_complete == tag_only.n_total,
        ),
        (
            "4",
            "FT thesis_tag_llm (seeds 42–44)",
            tag_llm.n_complete == tag_llm.n_total,
        ),
        ("5", "Gold eval + REPORT.md", report.is_file()),
    ]
    out: list[dict[str, Any]] = []
    first_pending = True
    run_by_unit = {"3": tag_only, "4": tag_llm}
    for uid, label, done in units:
        if done:
            state = "done"
        elif first_pending:
            run = run_by_unit.get(uid)
            in_progress = (
                run is not None
                and run.n_complete > 0
                and run.n_complete < run.n_total
            )
            state = "running" if in_progress else "next"
            first_pending = False
        else:
            state = "pending"
        detail: dict[str, Any] = {}
        if uid == "0":
            detail["lines"] = _count_lines(tag_jsonl)
        if uid == "1":
            detail["song_lines"] = n_songs
            detail["clip_lines"] = _count_lines(tag_llm_jsonl)
        if uid == "2":
            detail["cache_index_bytes"] = cache_index.stat().st_size if cache_index.is_file() else None
        if uid == "3":
            detail.update({"n_complete": tag_only.n_complete, "n_total": tag_only.n_total})
        if uid == "4":
            detail.update({"n_complete": tag_llm.n_complete, "n_total": tag_llm.n_total})
        out.append({"unit": uid, "label": label, "state": state, "detail": detail})
    return out


def _question_d_training_recipe(repo: Path) -> dict[str, Any]:
    """Static + on-disk params for Question D CLAP fine-tune (both arms)."""
    params_path = repo / "data/eval/llm_ablation/train_params.json"
    params = _read_json(params_path) or {}
    early = params.get("early_stopping") or {}
    tag_lines = _count_lines(repo / "data/mapping/clap_train_tag.jsonl")
    val_jsonl = params.get("val_jsonl", "data/mapping/clap_val_15s.jsonl")
    return {
        "question": "Does tag→LLM training text beat short tag strings for gold tag retrieval?",
        "arms": {
            "thesis_tag_only": {
                "run_id": "thesis_tag_only",
                "train_jsonl": "data/mapping/clap_train_tag.jsonl",
                "train_text": "Short tags from gold multihot (piano, vocal, relaxing) or fallback \"music\"",
            },
            "thesis_tag_llm": {
                "run_id": "thesis_tag_llm",
                "train_jsonl": "data/mapping/clap_train_tag_llm.jsonl",
                "train_text": "Llama-expanded sentence per song (same tags), copied to all 15s clips",
            },
        },
        "shared": {
            "train_clips": tag_lines,
            "val_jsonl": val_jsonl,
            "val_text": "Grok-style captions on held-out clips (same val for both arms)",
            "audio": "15s segments under data/music_db_15s/; backbone audio cache used at train time",
            "backbone": "Frozen CLAP AudioSet; train audio/text projection + transform heads",
            "loss": "Contrastive (scaled audio–text similarity, batch cross-entropy)",
            "params_file": _rel(repo, params_path) if params_path.is_file() else None,
        },
        "seeds": list(DEFAULT_SEEDS),
        "hyperparams": {
            "num_epochs": params.get("num_epochs", 20),
            "batch_size": params.get("batch_size", 32),
            "learning_rate": params.get("learning_rate", 1e-4),
        },
        "early_stopping": {
            "monitor": early.get("monitor", "val_similarity"),
            "mode": early.get("mode", "max"),
            "patience": early.get("patience", 2),
            "min_epochs": early.get("min_epochs", 5),
            "note": (
                "val_similarity = mean diagonal audio–text match on val JSONL; "
                "not the thesis retrieval metric (P@K / nDCG on gold — Unit 5)"
            ),
        },
        "checkpoints": "model/clap/finetune/<run_id>/seed_<n>/best_model.pt",
        "thesis_eval": "Gold retrieval vs random → data/eval/tag_llm_ablation/REPORT.md (Unit 5)",
    }


def _manifest_audio_readiness(repo: Path, manifest_path: Path) -> dict[str, Any]:
    """Count manifest rows and rows whose audio_path exists on disk."""
    if not manifest_path.is_file():
        return {
            "manifest": _rel(repo, manifest_path),
            "manifest_exists": False,
            "n_rows": 0,
            "n_audio_ready": 0,
        }
    n_rows = 0
    n_ready = 0
    with manifest_path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_rows += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ap = rec.get("audio_path")
            if ap and Path(ap).is_file():
                n_ready += 1
    return {
        "manifest": _rel(repo, manifest_path),
        "manifest_exists": True,
        "n_rows": n_rows,
        "n_audio_ready": n_ready,
    }


def _text_progress(done: int, total: int, width: int = 16) -> str:
    if total <= 0:
        return "░" * width + " —"
    ratio = min(1.0, done / total)
    filled = int(round(ratio * width))
    pct = int(round(ratio * 100))
    return "█" * filled + "░" * (width - filled) + f" {done}/{total} ({pct}%)"


def _read_public_download_status(repo: Path, dataset: str) -> dict[str, Any] | None:
    path = repo / "data/log/public_eval_downloads" / f"{dataset}.status.json"
    return _read_json(path)


def _public_eval_matrix(
    repo: Path, seeds: tuple[int, ...], arms: tuple[str, ...]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ds_id, _, eval_rel in PUBLIC_DATASETS:
        eval_dir = repo / eval_rel
        arm_cells: dict[str, Any] = {}
        for arm in arms:
            n_done = _public_eval_csv_count(eval_dir, arm, seeds)
            arm_cells[arm] = {"csv_done": n_done, "csv_expected": len(seeds)}
        rows.append({"dataset": ds_id, "arms": arm_cells})
    return rows


def _public_ood_pipeline_units(repo: Path, pub: dict[str, Any]) -> list[dict[str, Any]]:
    """Numbered pipeline units (download → eval → report), mirroring Question D."""
    ds_by_id = {d["dataset"]: d for d in pub.get("datasets") or []}
    report_ok = bool(pub.get("report_exists"))
    any_eval = any(d.get("eval_csvs", 0) > 0 for d in pub.get("datasets") or [])
    all_eval_done = all(d.get("eval_state") == "done" for d in pub.get("datasets") or [])

    prep_units = [
        ("0", "Jamendo five-tag download + manifest", "jamendo"),
        ("1", "MTAT download + manifest", "mtat"),
        ("2", "OpenMIC download + manifest", "openmic"),
    ]
    out: list[dict[str, Any]] = []
    first_pending = True
    for uid, label, ds_id in prep_units:
        ds = ds_by_id.get(ds_id) or {}
        prep = ds.get("prep_state", "pending")
        dl = _read_public_download_status(repo, ds_id) or {}
        dl_state = (dl.get("state") or "").upper()
        if prep == "done":
            state = "done"
        elif dl_state == "RUNNING":
            state = "running"
            first_pending = False
        elif dl_state == "FAILED":
            state = "failed"
            first_pending = False
        elif prep == "partial":
            state = "running" if first_pending else "pending"
            if first_pending:
                first_pending = False
        elif first_pending:
            state = "next"
            first_pending = False
        else:
            state = "pending"
        m = ds.get("manifest") or {}
        detail = {
            "audio_ready": f"{m.get('n_audio_ready', 0)}/{m.get('n_rows', 0)}",
            "download_status": dl_state or "—",
        }
        out.append({"unit": uid, "label": label, "state": state, "detail": detail, "dataset": ds_id})

    # Unit 3 — retrieval eval (all datasets that finished prep contribute to progress)
    ready_ds = [d for d in pub.get("datasets") or [] if d.get("prep_state") == "done"]
    total_csvs = sum(d.get("eval_csvs", 0) for d in pub.get("datasets") or [])
    expected_csvs = sum(d.get("eval_expected", 0) for d in ready_ds)
    if all_eval_done and expected_csvs > 0:
        eval_unit_state = "done"
    elif any_eval:
        eval_unit_state = "running"
    elif ready_ds:
        eval_unit_state = "next"
    else:
        eval_unit_state = "pending"
    out.append(
        {
            "unit": "3",
            "label": "Public retrieval eval (per-arm CSVs)",
            "state": eval_unit_state,
            "detail": {
                "csvs": f"{total_csvs}/{expected_csvs or pub.get('eval_expected_total', 0)}",
                "datasets_ready": ",".join(d["dataset"] for d in ready_ds) or "none",
                "arms": ",".join(pub.get("default_arms") or []),
            },
        }
    )

    # Unit 4 — combined report
    if report_ok and all_eval_done:
        report_state = "done"
    elif report_ok and any_eval:
        report_state = "partial"
    elif eval_unit_state == "done":
        report_state = "next"
    elif any_eval:
        report_state = "running"
    else:
        report_state = "pending"
    out.append(
        {
            "unit": "4",
            "label": "Combined data/eval/REPORT.md",
            "state": report_state,
            "detail": {"report_exists": report_ok},
        }
    )
    return out


def _public_ood_next_commands(repo: Path, pub: dict[str, Any], units: list[dict[str, Any]]) -> list[str]:
    """Actionable shell commands for the next pipeline step(s)."""
    cmds: list[str] = []
    by_unit = {u["unit"]: u for u in units}

    for uid in ("0", "1", "2"):
        u = by_unit.get(uid) or {}
        if u.get("state") in {"next", "failed", "running", "pending"}:
            ds = u.get("dataset")
            if u.get("state") == "failed":
                cmds.append(f"bash scripts/run_public_eval_download.sh {ds}  # retry failed download")
            elif u.get("state") in {"next", "pending"} and not any(
                (by_unit.get(x) or {}).get("state") == "running" for x in ("0", "1", "2")
            ):
                if ds and u.get("state") == "next":
                    cmds.append(f"bash scripts/run_public_eval_download.sh {ds}")
            break

    if not cmds:
        cmds.append("bash scripts/status_public_eval_download.sh")

    ready = [
        d["dataset"]
        for d in pub.get("datasets") or []
        if d.get("prep_state") == "done"
    ]
    eval_u = by_unit.get("3") or {}
    if ready and eval_u.get("state") in {"next", "running", "partial"}:
        ds_arg = " ".join(ready)
        cmds.append(
            f'DATASETS="{ds_arg}" ARMS="pretrained thesis_tag_only thesis_tag_llm" '
            "SKIP_EXISTING=1 sbatch scripts/sbatch_public_eval.sh"
        )
    elif eval_u.get("state") == "done" and not pub.get("report_exists"):
        cmds.append(
            'DATASETS="jamendo mtat openmic" ARMS="pretrained thesis_tag_only thesis_tag_llm" '
            "RUN_REPORT=1 SKIP_EXISTING=1 bash scripts/run_public_eval.sh  # report only"
        )

    if pub.get("status") == "done":
        cmds = ["# Public OOD pipeline complete — open data/eval/REPORT.md"]

    return cmds


def _public_ood_mermaid(units: list[dict[str, Any]]) -> str:
    def node(label: str, unit_id: str) -> str:
        u = next((x for x in units if x["unit"] == unit_id), {})
        st = u.get("state", "pending")
        return f'{unit_id}["{label}<br/>({st})"]'

    return "\n".join(
        [
            "```mermaid",
            "flowchart LR",
            f"  {node('Jamendo', '0')}",
            f"  {node('MTAT', '1')}",
            f"  {node('OpenMIC', '2')}",
            '  3["Retrieval eval"]',
            '  4["REPORT.md"]',
            "  0 --> 3",
            "  1 --> 3",
            "  2 --> 3",
            "  3 --> 4",
            "```",
        ]
    )


def _load_download_snapshot(repo: Path) -> dict[str, Any] | None:
    return _read_json(repo / "data/eval/download_status_snapshot.json")


def _public_eval_csv_count(eval_dir: Path, arm: str, seeds: tuple[int, ...]) -> int:
    if not eval_dir.is_dir():
        return 0
    n = 0
    for seed in seeds:
        if (eval_dir / f"{arm}_seed{seed}.csv").is_file():
            n += 1
    return n


def _public_ood_status(repo: Path) -> dict[str, Any]:
    seeds = DEFAULT_SEEDS
    download = _load_download_snapshot(repo) or {}
    datasets_out: list[dict[str, Any]] = []
    units_out: list[dict[str, Any]] = []

    for ds_id, manifest_rel, eval_rel in PUBLIC_DATASETS:
        manifest_path = repo / manifest_rel
        eval_dir = repo / eval_rel
        readiness = _manifest_audio_readiness(repo, manifest_path)
        n_rows = readiness["n_rows"]
        n_ready = readiness["n_audio_ready"]

        if not readiness["manifest_exists"]:
            prep_state = "pending"
        elif n_ready >= n_rows and n_rows > 0:
            prep_state = "done"
        elif n_ready > 0:
            prep_state = "partial"
        else:
            prep_state = "pending"

        arms_expected = len(DEFAULT_PUBLIC_ARMS) * len(seeds)
        if ds_id == "jamendo":
            arms_expected += 0  # pretrained uses same pattern
        eval_csvs = 0
        for arm in DEFAULT_PUBLIC_ARMS:
            eval_csvs += _public_eval_csv_count(eval_dir, arm, seeds)

        if eval_csvs >= arms_expected and arms_expected > 0:
            eval_state = "done"
        elif eval_csvs > 0:
            eval_state = "partial"
        elif prep_state == "done":
            eval_state = "next"
        else:
            eval_state = "pending"

        datasets_out.append(
            {
                "dataset": ds_id,
                "manifest": readiness,
                "prep_state": prep_state,
                "eval_dir": _rel(repo, eval_dir),
                "eval_csvs": eval_csvs,
                "eval_expected": arms_expected,
                "eval_state": eval_state,
            }
        )
        units_out.append(
            {
                "dataset": ds_id,
                "step": "manifest + audio",
                "state": prep_state,
                "detail": f"{n_ready}/{n_rows} audio_ready",
            }
        )
        units_out.append(
            {
                "dataset": ds_id,
                "step": "retrieval eval",
                "state": eval_state,
                "detail": f"{eval_csvs}/{arms_expected} csvs",
            }
        )

    report_path = repo / "data/eval/REPORT.md"
    ready_datasets = [d for d in datasets_out if d["prep_state"] == "done"]
    all_prep_done = len(ready_datasets) == len(datasets_out) and len(datasets_out) > 0
    any_eval = any(d["eval_csvs"] > 0 for d in datasets_out)
    all_eval_done = bool(ready_datasets) and all(
        d["eval_state"] == "done" for d in ready_datasets
    )
    eval_expected_total = sum(d["eval_expected"] for d in ready_datasets)
    total_csvs = sum(d["eval_csvs"] for d in datasets_out)

    if report_path.is_file() and all_eval_done:
        overall = "done"
    elif any_eval or ready_datasets:
        overall = "partial" if not all_eval_done else "eval pending"
    elif any(d["prep_state"] == "partial" for d in datasets_out):
        overall = "partial"
    else:
        overall = "pending"

    pub_core = {
        "status": overall,
        "report": _rel(repo, report_path),
        "report_exists": report_path.is_file(),
        "default_arms": list(DEFAULT_PUBLIC_ARMS),
        "seeds": list(seeds),
        "download_snapshot_utc": download.get("updated_utc"),
        "download_counts": {
            "jamendo_five_tag_mp3": download.get("jamendo_five_tag_mp3"),
            "jamendo_five_tag_target": download.get("jamendo_five_tag_target", 297),
            "mtat_mp3": download.get("mtat_mp3"),
            "openmic_ogg": download.get("openmic_ogg"),
        },
        "datasets": datasets_out,
        "units": units_out,
        "eval_expected_total": eval_expected_total,
        "eval_csvs_total": total_csvs,
        "guide": "docs/PUBLIC_OOD_EVAL.md",
        "orchestrator": "bash scripts/run_public_ood_pipeline.sh",
    }
    pipeline_units = _public_ood_pipeline_units(repo, pub_core)
    pub_core["pipeline_units"] = pipeline_units
    pub_core["eval_matrix"] = _public_eval_matrix(repo, seeds, DEFAULT_PUBLIC_ARMS)
    pub_core["next_commands"] = _public_ood_next_commands(repo, pub_core, pipeline_units)
    pub_core["mermaid"] = _public_ood_mermaid(pipeline_units)
    return pub_core


def public_ood_pipeline_actions(repo: Path | None = None) -> dict[str, Any]:
    """Structured plan for ``run_public_ood_pipeline.sh`` (download → eval)."""
    repo = repo or REPO
    pub = _public_ood_status(repo)
    units = pub.get("pipeline_units") or []
    by_unit = {u["unit"]: u for u in units}
    actions: list[dict[str, Any]] = []

    for uid in ("0", "1", "2"):
        u = by_unit.get(uid) or {}
        st = u.get("state")
        ds = u.get("dataset")
        if st in {"failed", "next"} and ds:
            actions.append({"type": "download", "dataset": ds, "reason": st})
            break
        if st == "running":
            break

    ready = [d["dataset"] for d in pub.get("datasets") or [] if d.get("prep_state") == "done"]
    eval_u = by_unit.get("3") or {}
    if ready and eval_u.get("state") in {"next", "partial", "running"}:
        actions.append(
            {
                "type": "eval",
                "datasets": ready,
                "arms": ["pretrained", "thesis_tag_only", "thesis_tag_llm"],
                "skip_existing": True,
            }
        )

    return {
        "status": pub.get("status"),
        "pipeline_units": units,
        "next_commands": pub.get("next_commands"),
        "datasets_ready": ready,
        "actions": actions,
    }


def build_snapshot(repo: Path | None = None) -> dict[str, Any]:
    repo = repo or REPO
    slurm_logs = sorted(repo.glob("slurm-*.out"), key=lambda p: p.stat().st_mtime, reverse=True)[:8]
    return {
        "updated_utc": _utc_now(),
        "thesis_questions": _thesis_question_status(repo),
        "question_d_units": _question_d_units(repo),
        "question_d_training_recipe": _question_d_training_recipe(repo),
        "public_ood": _public_ood_status(repo),
        "artifacts": {
            "clap_train_tag": _artifact(repo, repo / "data/mapping/clap_train_tag.jsonl"),
            "clap_train_tag_llm": _artifact(repo, repo / "data/mapping/clap_train_tag_llm.jsonl"),
            "tag_llm_ablation_report": _artifact(repo, repo / "data/eval/tag_llm_ablation/REPORT.md"),
            "llm_full_ablation_report": _artifact(repo, repo / "data/eval/llm_full_ablation/REPORT.md"),
        },
        "slurm_recent": [_parse_slurm_log(repo, p) for p in slurm_logs],
    }


def _status_badge(status: str) -> str:
    return {
        "done": "done",
        "completed": "done",
        "running": "running",
        "partial": "partial",
        "eval_pending": "eval pending",
        "next": "next",
        "pending": "pending",
        "skipped": "skipped",
        "failed": "failed",
    }.get(status, status)


def render_markdown(snapshot: dict[str, Any]) -> str:
    lines = [
        "# Progress monitor",
        "",
        f"*Auto-generated — do not edit by hand. Last refresh: `{snapshot['updated_utc']}`*",
        "",
        "Refresh:",
        "",
        "```bash",
        "bash scripts/refresh_progress.sh",
        "```",
        "",
        "Guide: [`docs/PROGRESS_MONITOR.md`](PROGRESS_MONITOR.md). Objectives: [`docs/THESIS_QUESTIONS.md`](THESIS_QUESTIONS.md). Public OOD: [`docs/PUBLIC_OOD_EVAL.md`](PUBLIC_OOD_EVAL.md).",
        "",
        "## Thesis questions",
        "",
        "| ID | Topic | Status | Report / artifacts |",
        "|----|-------|--------|-------------------|",
    ]
    for q in snapshot["thesis_questions"]:
        report = q.get("report") or "—"
        if q.get("report_exists"):
            report = f"`{report}`"
        elif q.get("artifacts"):
            report = ", ".join(f"`{a}`" for a in q["artifacts"][:2])
        runs = q.get("runs") or []
        if runs:
            run_bits = [f"{r['run_id']}: {r['n_complete']}/{r['n_total']} seeds" for r in runs]
            report = f"{report}; " + "; ".join(run_bits) if report != "—" else "; ".join(run_bits)
        lines.append(f"| **{q['id']}** | {q['label']} | **{_status_badge(q['status'])}** | {report} |")

    lines.extend(
        [
            "",
            "## Question D pipeline",
            "",
            "| Unit | Step | State | Detail |",
            "|------|------|-------|--------|",
        ]
    )
    for u in snapshot["question_d_units"]:
        detail = u.get("detail") or {}
        detail_s = ", ".join(f"{k}={v}" for k, v in detail.items() if v is not None) or "—"
        lines.append(
            f"| {u['unit']} | {u['label']} | **{_status_badge(u['state'])}** | {detail_s} |"
        )

    recipe = snapshot.get("question_d_training_recipe") or {}
    if recipe:
        shared = recipe.get("shared") or {}
        early = recipe.get("early_stopping") or {}
        hyper = recipe.get("hyperparams") or {}
        arms = recipe.get("arms") or {}
        tag_only = arms.get("thesis_tag_only") or {}
        tag_llm = arms.get("thesis_tag_llm") or {}
        seeds_s = ", ".join(str(s) for s in recipe.get("seeds") or DEFAULT_SEEDS)
        clips = shared.get("train_clips")
        clips_s = str(clips) if clips is not None else "?"
        lines.extend(
            [
                "",
                "## Question D — training recipe",
                "",
                recipe.get("question", ""),
                "",
                "**Two arms — only training text differs:**",
                "",
                f"| Arm | Run ID | Train JSONL | Text paired with each clip |",
                f"|-----|--------|-------------|----------------------------|",
                f"| Tag-only | `{tag_only.get('run_id', 'thesis_tag_only')}` | `{tag_only.get('train_jsonl', '')}` | {tag_only.get('train_text', '')} |",
                f"| Tag→LLM | `{tag_llm.get('run_id', 'thesis_tag_llm')}` | `{tag_llm.get('train_jsonl', '')}` | {tag_llm.get('train_text', '')} |",
                "",
                "**Shared setup:**",
                "",
                f"- **Train clips:** {clips_s} (`{tag_only.get('train_jsonl', 'clap_train_tag.jsonl')}`)",
                f"- **Val:** `{shared.get('val_jsonl', 'clap_val_15s.jsonl')}` — {shared.get('val_text', '')}",
                f"- **Audio:** {shared.get('audio', '')}",
                f"- **Backbone / loss:** {shared.get('backbone', '')}; {shared.get('loss', '')}",
                f"- **Params:** `{shared.get('params_file') or 'data/eval/llm_ablation/train_params.json'}`",
                "",
                "**Seeds & stop rule:**",
                "",
                f"- **Seeds:** {seeds_s} (one checkpoint per seed)",
                f"- **Max epochs:** {hyper.get('num_epochs', 20)}; **batch size:** {hyper.get('batch_size', 32)}",
                (
                    f"- **Early stop:** maximize `{early.get('monitor', 'val_similarity')}`, "
                    f"patience {early.get('patience', 2)}, min_epochs {early.get('min_epochs', 5)}"
                ),
                f"- **Note:** {early.get('note', '')}",
                "",
                f"**Thesis result (after Unit 5):** {recipe.get('thesis_eval', '')}",
                "",
            ]
        )

    lines.extend(["", "## Fine-tune seeds", ""])
    lines.append(
        "*Per seed: **ok** = checkpoint + complete; number is **best** val_similarity at best epoch "
        "(from `training_complete.json`, not last epoch).*"
    )
    lines.append("")
    for run_id in ("thesis_tag_only", "thesis_tag_llm", "thesis_ft_v1", "thesis_llm_full_llm"):
        r = _run_status(REPO, run_id)
        if r.n_complete == 0 and not any(s.checkpoint for s in r.seeds):
            continue
        seed_cells = []
        for s in r.seeds:
            mark = "ok" if s.training_complete and s.checkpoint else "…" if s.checkpoint else "—"
            if s.best_val_similarity is not None:
                ep = f"ep{s.best_epoch} " if s.best_epoch is not None else ""
                val = f" ({ep}val={s.best_val_similarity:.4f})"
            else:
                val = ""
            seed_cells.append(f"seed_{s.seed}: {mark}{val}")
        lines.append(f"- **`{run_id}`** — {r.n_complete}/{r.n_total} complete — {', '.join(seed_cells)}")

    pub = snapshot.get("public_ood") or {}
    if pub:
        dl = pub.get("download_counts") or {}
        j_mp3 = dl.get("jamendo_five_tag_mp3")
        j_tgt = dl.get("jamendo_five_tag_target", 297)
        dl_note = pub.get("download_snapshot_utc") or "never"
        pipeline_units = pub.get("pipeline_units") or []
        arms = pub.get("default_arms") or list(DEFAULT_PUBLIC_ARMS)
        lines.extend(
            [
                "",
                "## Public OOD pipeline",
                "",
                f"*Post-train external retrieval — separate from Question A–D. "
                f"Overall: **{_status_badge(pub.get('status', 'pending'))}**. "
                f"Orchestrator: `{pub.get('orchestrator', 'bash scripts/run_public_ood_pipeline.sh')}`*",
                "",
                pub.get("mermaid") or "",
                "",
                "| Unit | Step | State | Detail |",
                "|------|------|-------|--------|",
            ]
        )
        for u in pipeline_units:
            detail = u.get("detail") or {}
            if isinstance(detail, dict):
                detail_s = ", ".join(f"{k}={v}" for k, v in detail.items() if v is not None) or "—"
            else:
                detail_s = str(detail)
            lines.append(
                f"| {u.get('unit', '?')} | {u.get('label', '?')} | "
                f"**{_status_badge(u.get('state', 'pending'))}** | {detail_s} |"
            )

        lines.extend(
            [
                "",
                "### Prep & eval progress",
                "",
                "| Dataset | Prep (audio) | Eval CSVs |",
                "|---------|--------------|-----------|",
            ]
        )
        for ds in pub.get("datasets") or []:
            m = ds.get("manifest") or {}
            n_ready = m.get("n_audio_ready", 0)
            n_rows = m.get("n_rows", 0)
            prep_bar = _text_progress(n_ready, n_rows)
            eval_bar = _text_progress(ds.get("eval_csvs", 0), ds.get("eval_expected", 0))
            lines.append(
                f"| **{ds.get('dataset', '?')}** | `{prep_bar}` | `{eval_bar}` |"
            )

        lines.extend(["", "### Eval matrix (CSV seeds per arm)", ""])
        header = "| Dataset | " + " | ".join(arms) + " |"
        sep = "|---------|" + "|".join(["--------"] * len(arms)) + "|"
        lines.extend([header, sep])
        for row in pub.get("eval_matrix") or []:
            cells = []
            for arm in arms:
                cell = (row.get("arms") or {}).get(arm) or {}
                cells.append(f"{cell.get('csv_done', 0)}/{cell.get('csv_expected', 0)}")
            lines.append(f"| **{row.get('dataset', '?')}** | " + " | ".join(cells) + " |")

        report_line = (
            f"`{pub.get('report', 'data/eval/REPORT.md')}`"
            if pub.get("report_exists")
            else "— *(after unit 4)*"
        )
        lines.extend(
            [
                "",
                f"Combined report: {report_line} "
                f"(total CSVs {pub.get('eval_csvs_total', 0)}/{pub.get('eval_expected_total', 0)} "
                f"for prep-ready datasets; arms: {', '.join(arms)}; seeds 42–44).",
                "",
                f"Download snapshot (`refresh_download_status.sh`, `{dl_note}`): "
                f"Jamendo MP3 {j_mp3}/{j_tgt}, MTAT mp3 {dl.get('mtat_mp3', '?')}, "
                f"OpenMIC ogg {dl.get('openmic_ogg', '?')}.",
                "",
                "**Next commands:**",
                "",
                "```bash",
            ]
        )
        for cmd in pub.get("next_commands") or []:
            lines.append(cmd)
        lines.extend(
            [
                "```",
                "",
                "Guide: [`docs/PUBLIC_OOD_EVAL.md`](PUBLIC_OOD_EVAL.md). "
                "Download status: `bash scripts/status_public_eval_download.sh`.",
                "",
            ]
        )

    lines.extend(["", "## Recent Slurm jobs", ""])
    for job in snapshot.get("slurm_recent", [])[:5]:
        lines.append(f"### Job `{job['job_id']}` ({_status_badge(job['state'])})")
        lines.append(f"- Log: `{job['log']}` (mtime `{job['mtime_utc']}`)")
        lines.append(f"- Phase: `{job['phase']}`")
        if job.get("tail"):
            lines.append("- Tail:")
            lines.append("```")
            lines.extend(job["tail"])
            lines.append("```")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_outputs(repo: Path | None = None) -> dict[str, Any]:
    repo = repo or REPO
    snapshot = build_snapshot(repo)
    snap_path = repo / "data/eval/progress_snapshot.json"
    md_path = repo / "docs/PROGRESS.md"
    snap_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    snap_path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(snapshot), encoding="utf-8")
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh thesis progress monitor outputs.")
    parser.add_argument("--repo", type=Path, default=REPO, help="Repository root")
    parser.add_argument("--json-only", action="store_true", help="Print snapshot JSON to stdout")
    parser.add_argument(
        "--ood-plan-json",
        action="store_true",
        help="Print public OOD pipeline actions JSON (for run_public_ood_pipeline.sh)",
    )
    args = parser.parse_args()
    if args.ood_plan_json:
        print(json.dumps(public_ood_pipeline_actions(args.repo), indent=2))
        return
    snapshot = write_outputs(args.repo)
    if args.json_only:
        print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()
