"""Per-iteration validation metrics for self-train loop."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from app.init_model import compute_avg_similarity, embed_pipeline, load_model_from_checkpoint
from app.self_train.jsonl_io import load_jsonl_rows


def eval_val_mean_similarity(
    *,
    val_jsonl: Path,
    checkpoint: Path | None = None,
    max_samples: int | None = None,
    batch_size: int = 8,
) -> float:
    rows = load_jsonl_rows(val_jsonl, max_samples=max_samples)
    model = load_model_from_checkpoint(
        str(checkpoint) if checkpoint else None
    )
    paths = [r["audio_path"] for r in rows]
    texts = [r["text"] for r in rows]
    audio_parts = []
    text_parts = []
    for i in range(0, len(paths), batch_size):
        a, t = embed_pipeline(
            paths[i : i + batch_size],
            model,
            tensor_mode=False,
            texts=texts[i : i + batch_size],
        )
        audio_parts.extend(list(a))
        text_parts.extend(list(t))
    return float(compute_avg_similarity(audio_parts, text_parts).item())


def run_gold_retrieval_eval(
    *,
    checkpoint: Path,
    repo_root: Path,
) -> dict[str, Any] | None:
    """Optional subprocess: music_eval_retrieval_vs_random with RAGWEB_CLAP_CHECKPOINT."""
    env = os.environ.copy()
    env["RAGWEB_CLAP_CHECKPOINT"] = str(checkpoint.resolve())
    env["PYTHONPATH"] = str(repo_root)
    cmd = [
        sys.executable,
        "-m",
        "app.data_handling.music_eval_retrieval_vs_random",
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,
            check=False,
        )
        return {
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-2000:] if proc.stdout else "",
            "stderr_tail": proc.stderr[-2000:] if proc.stderr else "",
        }
    except Exception as exc:
        return {"error": str(exc)}


def write_iter_metrics(
    path: Path,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
