"""
Zero-shot tempo evaluation for CLAP using BPM-derived pseudo labels.

Inputs:
  - data/mapping/clap_val_15s.jsonl (expects audio_path field by default)

Outputs:
  - data/mapping/tempo_eval_predictions.jsonl
  - data/mapping/tempo_eval_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import laion_clap
import librosa
import numpy as np
from tqdm import tqdm

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import settings

TEMPO_LABELS = ["slow", "mid-tempo", "fast"]
TEMPO_PROMPTS = [
    "a slow tempo music track",
    "a mid-tempo music track",
    "a fast tempo music track",
]


def _normalize_embeddings(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def _bpm_to_label(bpm: float) -> str:
    if bpm < 80.0:
        return "slow"
    if bpm <= 120.0:
        return "mid-tempo"
    return "fast"


def _load_val_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        raise FileNotFoundError(f"Validation JSONL not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _resolve_audio_path(value: str, project_root: Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def _load_model() -> laion_clap.CLAP_Module:
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    model.load_ckpt(str(settings.CLAP_MODEL_FILE))
    return model


def _estimate_bpm(path: Path) -> float:
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo)


def _compute_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    label_to_id = {label: idx for idx, label in enumerate(TEMPO_LABELS)}
    cm = [[0 for _ in TEMPO_LABELS] for _ in TEMPO_LABELS]  # rows=true, cols=pred
    for truth, pred in zip(y_true, y_pred):
        cm[label_to_id[truth]][label_to_id[pred]] += 1

    total = len(y_true)
    correct = sum(cm[i][i] for i in range(len(TEMPO_LABELS)))
    accuracy = (correct / total) if total else 0.0

    per_class: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for i, label in enumerate(TEMPO_LABELS):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(len(TEMPO_LABELS)) if r != i)
        fn = sum(cm[i][c] for c in range(len(TEMPO_LABELS)) if c != i)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(sum(cm[i])),
        }
        f1_values.append(f1)

    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0
    return {
        "labels": TEMPO_LABELS,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "per_class": per_class,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_existing_predictions(
    path: Path,
    *,
    audio_field: str,
) -> tuple[set[str], list[dict[str, Any]], list[str], list[str]]:
    """
    Load existing prediction ledger and return:
      - processed audio keys
      - raw records
      - valid y_true
      - valid y_pred
    """
    processed: set[str] = set()
    records: list[dict[str, Any]] = []
    y_true: list[str] = []
    y_pred: list[str] = []
    if not path.is_file():
        return processed, records, y_true, y_pred

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue

            audio_value = row.get(audio_field)
            if isinstance(audio_value, str) and audio_value.strip():
                processed.add(audio_value)

            status = row.get("status")
            gt_tempo = row.get("gt_tempo")
            pred_tempo = row.get("pred_tempo")
            if (
                status == "ok"
                and isinstance(gt_tempo, str)
                and isinstance(pred_tempo, str)
                and gt_tempo in TEMPO_LABELS
                and pred_tempo in TEMPO_LABELS
            ):
                y_true.append(gt_tempo)
                y_pred.append(pred_tempo)

            records.append(row)

    return processed, records, y_true, y_pred


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Zero-shot tempo evaluation for CLAP.")
    parser.add_argument(
        "--val-jsonl",
        type=Path,
        default=settings.MAPPING_DIR / "clap_val_15s.jsonl",
        help="Validation manifest JSONL path.",
    )
    parser.add_argument(
        "--audio-field",
        default="audio_path",
        help="Field name holding the clip path in each JSONL row.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=settings.BASE_DIR,
        help="Project root for resolving relative clip paths.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Audio embedding batch size.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="If >0, evaluate only first N rows.",
    )
    parser.add_argument(
        "--pred-output",
        type=Path,
        default=settings.MAPPING_DIR / "tempo_eval_predictions.jsonl",
        help="Prediction rows output JSONL.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=settings.MAPPING_DIR / "tempo_eval_metrics.json",
        help="Summary metrics output JSON.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing prediction ledger if present (default: true).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Start fresh by truncating prediction output and ignoring resume ledger.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Append checkpoint to prediction ledger every N newly processed clips.",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.save_every <= 0:
        raise ValueError("--save-every must be > 0")

    rows = _load_val_rows(args.val_jsonl)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError("No rows found in validation JSONL.")

    if args.overwrite and args.pred_output.exists():
        args.pred_output.unlink()

    processed_keys: set[str] = set()
    existing_predictions: list[dict[str, Any]] = []
    existing_y_true: list[str] = []
    existing_y_pred: list[str] = []
    if args.resume and not args.overwrite:
        (
            processed_keys,
            existing_predictions,
            existing_y_true,
            existing_y_pred,
        ) = _load_existing_predictions(args.pred_output, audio_field=args.audio_field)

    # Step 1: prepare valid rows + BPM pseudo labels
    prepared: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in tqdm(rows, desc="Estimating BPM", unit="clip"):
        audio_value = row.get(args.audio_field)
        if not isinstance(audio_value, str) or not audio_value.strip():
            skipped.append({"status": "missing_audio_field", "row": row})
            continue
        if audio_value in processed_keys:
            continue
        abs_path = _resolve_audio_path(audio_value, args.project_root)
        if not abs_path.is_file():
            skipped.append({"status": "missing_file", args.audio_field: audio_value})
            continue

        try:
            bpm = _estimate_bpm(abs_path)
        except Exception as exc:
            skipped.append(
                {"status": "bpm_error", args.audio_field: audio_value, "error": str(exc)}
            )
            continue

        prepared.append(
            {
                "row": row,
                "audio_value": audio_value,
                "audio_abs": str(abs_path),
                "bpm": bpm,
                "gt_tempo": _bpm_to_label(bpm),
            }
        )

    # Persist skipped records from this run, so they can also be resumed/skipped.
    _append_jsonl(args.pred_output, skipped)

    # Step 2: load CLAP and encode prompts once (only if there is pending work)
    text_emb: np.ndarray | None = None
    model: laion_clap.CLAP_Module | None = None
    if prepared:
        model = _load_model()
        text_emb = model.get_text_embedding(x=TEMPO_PROMPTS, use_tensor=False)
        text_emb = _normalize_embeddings(np.asarray(text_emb, dtype="float32"))

    # Step 3: batch audio embeddings + prediction
    new_predictions: list[dict[str, Any]] = []
    y_true: list[str] = existing_y_true.copy()
    y_pred: list[str] = existing_y_pred.copy()
    total_batches = (len(prepared) + args.batch_size - 1) // args.batch_size
    pending_since_last_save = 0

    for start in tqdm(
        range(0, len(prepared), args.batch_size),
        total=total_batches,
        desc="Zero-shot CLAP inference",
        unit="batch",
    ):
        batch = prepared[start : start + args.batch_size]
        batch_paths = [item["audio_abs"] for item in batch]
        assert model is not None and text_emb is not None
        audio_emb = model.get_audio_embedding_from_filelist(x=batch_paths, use_tensor=False)
        audio_emb = _normalize_embeddings(np.asarray(audio_emb, dtype="float32"))
        sims = np.matmul(audio_emb, text_emb.T)  # [B, 3]

        for idx, item in enumerate(batch):
            score = sims[idx]
            pred_id = int(np.argmax(score))
            pred_label = TEMPO_LABELS[pred_id]
            rank_scores = np.sort(score)[::-1]
            margin = float(rank_scores[0] - rank_scores[1]) if len(rank_scores) > 1 else 0.0

            y_true.append(item["gt_tempo"])
            y_pred.append(pred_label)
            new_predictions.append(
                {
                    args.audio_field: item["audio_value"],
                    "bpm": item["bpm"],
                    "gt_tempo": item["gt_tempo"],
                    "pred_tempo": pred_label,
                    "scores": {
                        "slow": float(score[0]),
                        "mid-tempo": float(score[1]),
                        "fast": float(score[2]),
                    },
                    "margin_top1_top2": margin,
                    "status": "ok",
                }
            )
            pending_since_last_save += 1

            if pending_since_last_save >= args.save_every:
                _append_jsonl(args.pred_output, new_predictions)
                new_predictions = []
                pending_since_last_save = 0

    _append_jsonl(args.pred_output, new_predictions)

    metrics = _compute_metrics(y_true, y_pred)
    metrics.update(
        {
            "num_total_rows": len(rows),
            "num_valid_for_eval": len(y_true),
            "num_skipped_this_run": len(skipped),
            "num_skipped_total_seen": len(
                [
                    row
                    for row in (existing_predictions + skipped)
                    if row.get("status") != "ok"
                ]
            ),
            "num_resumed_predictions": len(existing_y_true),
            "num_new_predictions": len(y_true) - len(existing_y_true),
            "num_pending_after_resume": len(prepared),
            "val_jsonl": str(args.val_jsonl),
            "audio_field": args.audio_field,
            "tempo_prompts": TEMPO_PROMPTS,
            "clap_ckpt": str(settings.CLAP_MODEL_FILE),
            "resume": bool(args.resume and not args.overwrite),
            "overwrite": bool(args.overwrite),
        }
    )

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved predictions: {args.pred_output}")
    print(f"Saved metrics: {args.metrics_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
