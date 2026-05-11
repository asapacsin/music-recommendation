"""
Report BPM/tempo coverage for the gold pipeline.

Modes:
  - --sidecar + --tempo-jsonl: which labeled songs lack an ok song-level tempo row (pre-merge).
  - --merged-jsonl: how many merged rows have non-null program_tempo (post-merge audit).

Exit code 1 with --strict if any expected song is missing ok tempo / program_tempo.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import settings


def _norm_sp(value: str) -> str:
    return value.strip().replace("\\", "/")


def _load_sidecar_paths(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Sidecar not found: {path}")
    out: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            sp = row.get("source_path")
            if isinstance(sp, str) and sp.strip():
                out.append(_norm_sp(sp))
    return out


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


def _tempo_ok(rec: dict[str, Any]) -> bool:
    if rec.get("status") != "ok":
        return False
    bpms = rec.get("clip_bpms")
    return isinstance(bpms, list) and len(bpms) > 0


def _run_sidecar_tempo(sidecar: Path, tempo_jsonl: Path) -> dict[str, Any]:
    expected = _load_sidecar_paths(sidecar)
    tempo_idx = _load_tempo_index(tempo_jsonl)
    missing: list[str] = []
    not_ok: list[str] = []
    for sp in expected:
        rec = tempo_idx.get(sp)
        if rec is None:
            missing.append(sp)
        elif not _tempo_ok(rec):
            not_ok.append(sp)
    bad = missing + not_ok
    return {
        "mode": "sidecar_vs_tempo",
        "sidecar": str(sidecar),
        "tempo_jsonl": str(tempo_jsonl),
        "expected_songs": len(expected),
        "ok_tempo_rows": len(expected) - len(bad),
        "missing_in_tempo_jsonl": len(missing),
        "present_but_not_ok": len(not_ok),
        "missing_paths_sample": missing[:30],
        "not_ok_paths_sample": not_ok[:30],
    }


def _run_merged(merged_path: Path) -> dict[str, Any]:
    if not merged_path.is_file():
        raise FileNotFoundError(f"Merged gold not found: {merged_path}")
    total = 0
    with_pt = 0
    null_paths: list[str] = []
    with merged_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            total += 1
            sp = row.get("source_path")
            sp_s = _norm_sp(sp) if isinstance(sp, str) else ""
            pt = row.get("program_tempo")
            if isinstance(pt, dict):
                with_pt += 1
            elif sp_s:
                null_paths.append(sp_s)
    return {
        "mode": "merged_audit",
        "merged_jsonl": str(merged_path),
        "rows": total,
        "with_program_tempo": with_pt,
        "without_program_tempo": len(null_paths),
        "without_program_tempo_sample": null_paths[:30],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Gold BPM / tempo coverage report.")
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=None,
        help="Gold sidecar JSONL; compare each source_path to --tempo-jsonl.",
    )
    parser.add_argument(
        "--tempo-jsonl",
        type=Path,
        default=settings.DATA_DIR / "eval" / "tempo_eval_song_predictions.jsonl",
        help="Song-level tempo eval JSONL (used with --sidecar).",
    )
    parser.add_argument(
        "--merged-jsonl",
        type=Path,
        default=None,
        help="If set, audit merged gold rows for program_tempo.bpm_mean (post-merge).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if any expected song lacks ok tempo (--sidecar) or any merged row lacks program_tempo.",
    )
    args = parser.parse_args()

    if args.merged_jsonl is not None and args.sidecar is not None:
        parser.error("Use either --merged-jsonl or --sidecar, not both.")
    if args.merged_jsonl is None and args.sidecar is None:
        parser.error("Pass --sidecar (pre-merge) or --merged-jsonl (post-merge).")

    if args.merged_jsonl is not None:
        report = _run_merged(args.merged_jsonl)
        fail = report["without_program_tempo"] > 0
    else:
        report = _run_sidecar_tempo(args.sidecar, args.tempo_jsonl)
        fail = (report["missing_in_tempo_jsonl"] + report["present_but_not_ok"]) > 0

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
