"""
Song-level zero-shot tempo evaluation using K representative clips per source track.

Reads manifest from music_eval_build_song_manifest.py (eval_audio_paths per song).

Per clip: BPM pseudo-label + CLAP tempo prompt similarities.
Aggregate per song:
  - gt_tempo: majority vote over clip labels; tie-break: label from mean BPM
  - pred_tempo: majority vote over clip preds; tie-break: argmax of mean similarity vector

Resume key: source_path
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import settings

from app.data_handling.music_eval_zeroshot_tempo import (
    TEMPO_LABELS,
    TEMPO_PROMPTS,
    _append_jsonl,
    _bpm_to_label,
    _compute_metrics,
    _estimate_bpm,
    _load_model,
    _normalize_embeddings,
    _resolve_audio_path,
)


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _majority_or_none(labels: list[str]) -> str | None:
    if not labels:
        return None
    c = Counter(labels)
    top = c.most_common()
    if len(top) == 1:
        return top[0][0]
    if top[0][1] > top[1][1]:
        return top[0][0]
    return None


def _load_processed_sources(path: Path) -> set[str]:
    """Any ledger row with source_path counts as done (ok or error), so reruns do not repeat work."""
    done: set[str] = set()
    if not path.is_file():
        return done
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
            sp = row.get("source_path")
            if isinstance(sp, str) and sp.strip():
                done.add(sp.strip())
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description="Song-level zero-shot tempo eval (multi-clip).")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=settings.DATA_DIR / "eval" / "song_eval_manifest.jsonl",
        help="Song manifest JSONL from music_eval_build_song_manifest.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=settings.BASE_DIR,
        help="Resolve relative clip paths.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Clip batch size for CLAP audio embedding.",
    )
    parser.add_argument(
        "--max-songs",
        type=int,
        default=0,
        help="If >0, only first N songs from manifest.",
    )
    parser.add_argument(
        "--pred-output",
        type=Path,
        default=settings.DATA_DIR / "eval" / "tempo_eval_song_predictions.jsonl",
        help="Per-song prediction ledger.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=settings.DATA_DIR / "eval" / "tempo_eval_song_metrics.json",
        help="Song-level metrics JSON.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Append predictions after this many newly completed songs.",
    )
    args = parser.parse_args()
    if args.batch_size <= 0 or args.save_every <= 0:
        raise ValueError("--batch-size and --save-every must be > 0")

    if args.overwrite and args.pred_output.exists():
        args.pred_output.unlink()

    manifest = _load_manifest(args.manifest)
    if args.max_songs > 0:
        manifest = manifest[: args.max_songs]
    if not manifest:
        raise ValueError("Empty manifest.")

    processed = _load_processed_sources(args.pred_output) if args.resume and not args.overwrite else set()

    pending_songs: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in manifest:
        sp = row.get("source_path")
        if not isinstance(sp, str) or not sp.strip():
            skipped.append({"status": "missing_source_path", "row": row})
            continue
        if sp in processed:
            continue
        clips = row.get("eval_audio_paths")
        if not isinstance(clips, list) or not clips:
            skipped.append({"status": "no_clips", "source_path": sp})
            continue
        pending_songs.append(row)

    if skipped:
        print(f"Skipped {len(skipped)} manifest rows (missing source_path or no clips).")

    y_true: list[str] = []
    y_pred: list[str] = []
    if args.resume and not args.overwrite and args.pred_output.is_file():
        with args.pred_output.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("status") == "ok" and rec.get("gt_tempo_song") and rec.get("pred_tempo_song"):
                    gt = rec["gt_tempo_song"]
                    pr = rec["pred_tempo_song"]
                    if gt in TEMPO_LABELS and pr in TEMPO_LABELS:
                        y_true.append(gt)
                        y_pred.append(pr)

    text_emb = None
    model = None
    if pending_songs:
        model = _load_model()
        te = model.get_text_embedding(x=TEMPO_PROMPTS, use_tensor=False)
        text_emb = _normalize_embeddings(np.asarray(te, dtype="float32"))

    new_rows: list[dict[str, Any]] = []
    pending_since_save = 0

    for song_row in tqdm(pending_songs, desc="Song-level tempo eval", unit="song"):
        sp = str(song_row["source_path"]).strip()
        clips_raw = song_row["eval_audio_paths"]
        assert isinstance(clips_raw, list)

        clip_bpms: list[float] = []
        clip_gt: list[str] = []
        clip_paths_abs: list[str] = []
        clip_pred: list[str] = []

        bad_reason: str | None = None
        for rel in clips_raw:
            if not isinstance(rel, str) or not rel.strip():
                bad_reason = "bad_clip_path"
                break
            abs_p = _resolve_audio_path(rel.strip(), args.project_root)
            if not abs_p.is_file():
                bad_reason = "missing_file"
                break
            try:
                bpm = _estimate_bpm(abs_p)
            except Exception as exc:
                bad_reason = f"bpm_error:{exc}"
                break
            clip_bpms.append(bpm)
            clip_gt.append(_bpm_to_label(bpm))
            clip_paths_abs.append(str(abs_p))

        if bad_reason:
            new_rows.append({"status": bad_reason, "source_path": sp})
            pending_since_save += 1
            if pending_since_save >= args.save_every:
                _append_jsonl(args.pred_output, new_rows)
                new_rows = []
                pending_since_save = 0
            continue

        assert model is not None and text_emb is not None
        score_mat: list[list[float]] = []
        for start in range(0, len(clip_paths_abs), args.batch_size):
            batch_paths = clip_paths_abs[start : start + args.batch_size]
            audio_emb = model.get_audio_embedding_from_filelist(x=batch_paths, use_tensor=False)
            audio_emb = _normalize_embeddings(np.asarray(audio_emb, dtype="float32"))
            sims = np.matmul(audio_emb, text_emb.T)
            for bi in range(sims.shape[0]):
                score = sims[bi]
                pid = int(np.argmax(score))
                clip_pred.append(TEMPO_LABELS[pid])
                score_mat.append([float(score[0]), float(score[1]), float(score[2])])

        mean_vec = np.mean(np.stack([np.array(s, dtype=np.float32) for s in score_mat], axis=0), axis=0)
        maj_gt = _majority_or_none(clip_gt)
        gt_song = maj_gt if maj_gt is not None else _bpm_to_label(float(np.mean(clip_bpms)))
        maj_pr = _majority_or_none(clip_pred)
        pred_song = maj_pr if maj_pr is not None else TEMPO_LABELS[int(np.argmax(mean_vec))]

        y_true.append(gt_song)
        y_pred.append(pred_song)

        new_rows.append(
            {
                "status": "ok",
                "source_path": sp,
                "num_segments": song_row.get("num_segments"),
                "eval_audio_paths": clips_raw,
                "clip_bpms": clip_bpms,
                "clip_gt_tempo": clip_gt,
                "clip_pred_tempo": clip_pred,
                "gt_tempo_song": gt_song,
                "pred_tempo_song": pred_song,
                "mean_scores_song": {
                    "slow": float(mean_vec[0]),
                    "mid-tempo": float(mean_vec[1]),
                    "fast": float(mean_vec[2]),
                },
                "tempo_prompts": TEMPO_PROMPTS,
            }
        )
        pending_since_save += 1
        if pending_since_save >= args.save_every:
            _append_jsonl(args.pred_output, new_rows)
            new_rows = []
            pending_since_save = 0

    _append_jsonl(args.pred_output, new_rows)

    metrics = _compute_metrics(y_true, y_pred)
    metrics.update(
        {
            "eval_unit": "song",
            "aggregation": "majority_over_K_clips_tie_mean_bpm_or_mean_scores",
            "manifest": str(args.manifest),
            "num_songs_manifest": len(manifest),
            "num_songs_scored": len(y_true),
            "clap_ckpt": str(settings.CLAP_MODEL_FILE),
            "pred_output": str(args.pred_output),
        }
    )
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved: {args.pred_output}")
    print(f"Saved: {args.metrics_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
