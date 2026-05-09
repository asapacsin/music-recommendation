"""
Merge human multi-hot gold CSV + sidecar + program tempo (song eval) + optional LLM metadata.

Output: JSONL suitable for downstream val/test evaluation (one object per labeled song).

Inputs (defaults under data/eval/):
  - gold_labels_multihot_template.csv  (filled 0/1; first column song_name)
  - gold_labels_multihot_template.csv.sidecar.jsonl  (song_name, source_path per line, same order as CSV body)
  - tempo_eval_song_predictions.jsonl  (from music_eval_zeroshot_tempo_song; status ok rows)

Optional:
  - music_metadata.json — matched by basename of source_path vs metadata \"audio\" field.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import settings

from app.data_handling.music_eval_prepare_gold_multihot_csv import MULTIHOT_COLUMNS


def _norm_sp(value: str) -> str:
    return value.strip().replace("\\", "/")


def _basename_key(path_str: str) -> str:
    return Path(_norm_sp(path_str)).name


def _load_sidecar(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        raise FileNotFoundError(f"Sidecar not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _load_tempo_index(path: Path) -> dict[str, dict[str, Any]]:
    by_sp: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return by_sp
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                continue
            sp = rec.get("source_path")
            if isinstance(sp, str) and sp.strip():
                by_sp[_norm_sp(sp.strip())] = rec
    return by_sp


def _load_metadata_index(path: Path) -> dict[str, dict[str, Any]]:
    """Index by basename of metadata audio field (matches this project's extract format)."""
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        audio = row.get("audio")
        if isinstance(audio, str) and audio.strip():
            key = _basename_key(audio)
            out[key] = row
    return out


def _parse_bin(v: str) -> int:
    v = v.strip()
    if not v:
        return 0
    return int(float(v))


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge gold CSV + tempo + metadata into JSONL.")
    parser.add_argument(
        "--gold-csv",
        type=Path,
        default=settings.DATA_DIR / "eval" / "gold_labels_multihot_template.csv",
        help="Human-labeled CSV (song_name + multihot columns).",
    )
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=None,
        help="Defaults to <gold-csv>.sidecar.jsonl",
    )
    parser.add_argument(
        "--tempo-jsonl",
        type=Path,
        default=settings.DATA_DIR / "eval" / "tempo_eval_song_predictions.jsonl",
        help="Song-level tempo eval output (optional if missing tempo keys will be null).",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=settings.MUSIC_METADATA_FILE,
        help="music_metadata.json array (optional).",
    )
    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Do not attach LLM metadata rows.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=settings.DATA_DIR / "eval" / "gold_merged.jsonl",
        help="Merged JSONL output.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=settings.DATA_DIR / "eval" / "gold_merge_summary.json",
        help="Merge statistics JSON.",
    )
    args = parser.parse_args()

    sidecar_path = args.sidecar or args.gold_csv.with_suffix(args.gold_csv.suffix + ".sidecar.jsonl")

    sidecar_rows = _load_sidecar(sidecar_path)
    tempo_idx = _load_tempo_index(args.tempo_jsonl)
    meta_idx = {} if args.skip_metadata else _load_metadata_index(args.metadata_json)

    merged: list[dict[str, Any]] = []
    warnings: list[str] = []

    if not args.gold_csv.is_file():
        raise FileNotFoundError(f"Gold CSV not found: {args.gold_csv}")

    with args.gold_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        body_rows = list(reader)

    if len(body_rows) != len(sidecar_rows):
        raise ValueError(
            f"Row count mismatch: CSV data rows={len(body_rows)} sidecar={len(sidecar_rows)}. "
            "Regenerate CSV and sidecar together."
        )

    seen_names: set[str] = set()
    for i, (csv_row, sc) in enumerate(zip(body_rows, sidecar_rows)):
        song_csv = (csv_row.get("song_name") or "").strip()
        song_sc = (sc.get("song_name") or "").strip()
        sp = sc.get("source_path")
        if not isinstance(sp, str) or not sp.strip():
            warnings.append(f"line {i}: missing source_path in sidecar")
            continue
        sp = _norm_sp(sp.strip())
        if song_csv and song_sc and song_csv != song_sc:
            warnings.append(f"line {i}: song_name mismatch csv={song_csv!r} sidecar={song_sc!r}")
        display_name = song_csv or song_sc or _basename_key(sp)
        if display_name in seen_names:
            warnings.append(f"duplicate song_name in gold CSV: {display_name!r}")
        seen_names.add(display_name)

        human: dict[str, int] = {}
        for col in MULTIHOT_COLUMNS:
            raw = csv_row.get(col, "")
            try:
                human[col] = _parse_bin(str(raw))
            except ValueError:
                human[col] = 0
                warnings.append(f"line {i}: bad value for {col}, treated as 0")

        tempo_rec = tempo_idx.get(sp)
        program_tempo: dict[str, Any] | None = None
        if tempo_rec and tempo_rec.get("status") == "ok":
            bpms = tempo_rec.get("clip_bpms")
            bpm_mean = None
            if isinstance(bpms, list) and bpms:
                bpm_mean = float(sum(float(x) for x in bpms) / len(bpms))
            program_tempo = {
                "clip_bpms": bpms,
                "bpm_mean": bpm_mean,
                "tempo_bin_bpm": tempo_rec.get("gt_tempo_song"),
                "tempo_clap_zeroshot": tempo_rec.get("pred_tempo_song"),
                "clip_gt_tempo": tempo_rec.get("clip_gt_tempo"),
                "clip_pred_tempo": tempo_rec.get("clip_pred_tempo"),
                "mean_scores_song": tempo_rec.get("mean_scores_song"),
                "eval_audio_paths": tempo_rec.get("eval_audio_paths"),
            }
        else:
            warnings.append(f"no ok tempo row for source_path={sp!r}")

        basename = _basename_key(sp)
        meta_row = meta_idx.get(basename)
        program_meta = None
        if meta_row is not None:
            program_meta = {
                "audio": meta_row.get("audio"),
                "text": meta_row.get("text"),
                "mood": meta_row.get("mood"),
                "confidence": meta_row.get("confidence"),
            }

        merged.append(
            {
                "song_name": display_name,
                "source_path": sp,
                "human_multihot": human,
                "program_tempo": program_tempo,
                "program_metadata": program_meta,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        for row in merged:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "rows_written": len(merged),
        "with_program_tempo": sum(1 for r in merged if r.get("program_tempo")),
        "with_program_metadata": sum(1 for r in merged if r.get("program_metadata")),
        "warnings": warnings[:50],
        "warning_total": len(warnings),
        "inputs": {
            "gold_csv": str(args.gold_csv),
            "sidecar": str(sidecar_path),
            "tempo_jsonl": str(args.tempo_jsonl),
            "metadata_json": None if args.skip_metadata else str(args.metadata_json),
        },
        "output": str(args.out),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    brief = {
        "rows_written": summary["rows_written"],
        "with_program_tempo": summary["with_program_tempo"],
        "with_program_metadata": summary["with_program_metadata"],
        "warning_total": summary["warning_total"],
        "output": summary["output"],
        "summary_json": str(args.summary_json),
    }
    try:
        print(json.dumps(brief, ensure_ascii=False, indent=2))
    except UnicodeEncodeError:
        print(json.dumps(brief, ensure_ascii=True, indent=2))
    print(f"Details (warnings list): {args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
