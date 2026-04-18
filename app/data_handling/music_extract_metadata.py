"""
Extract structured music metadata from filenames in data/music_db using Grok (xAI).

Each saved record:
  audio, text, mood, confidence

confidence is computed in code as:
  0.5 * S + 0.3 * R + 0.2 * D
(S, R, D are model-provided scores in [0, 1], not stored in the JSON file.)
Stored confidence values are rounded to two decimal places.

Output file:
    data/mapping/music_metadata.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Weights for confidence = W_S * S + W_R * R + W_D * D
CONFIDENCE_W_S = 0.5
CONFIDENCE_W_R = 0.3
CONFIDENCE_W_D = 0.2

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

from config import settings

DEFAULT_MODEL = "grok-4.20-non-reasoning"
DEFAULT_BASE_URL = "https://api.x.ai/v1"
DEFAULT_OUTPUT_PATH = settings.MAPPING_DIR / "music_metadata.json"
SYSTEM_PROMPT = (
    "You infer music-related cues only from the given file path or filename (no audio). "
    "Return one JSON object with exactly these keys: "
    "\"audio\", \"text\", \"mood\", \"S\", \"R\", \"D\". "
    "- \"audio\": string — must equal the exact path string the user provides. "
    "- \"text\": string — a short English line (one sentence preferred) describing likely style, "
    "vocals, instruments, and overall feel inferred from the filename; be conservative if uncertain. "
    "- \"mood\": string — a single concise English mood label (e.g. melancholic, upbeat) inferred "
    "from the filename, or JSON null if it cannot be inferred. "
    "- \"S\": number from 0 to 1 — semantic match: how well your text and mood align with the "
    "meaningful tokens in the filename (artist/title/genre-like fragments vs noise). "
    "- \"R\": number from 0 to 1 — source reliability: how informative and trustworthy the "
    "filename/path is as evidence (structured title vs random IDs or garbage). "
    "- \"D\": number from 0 to 1 — description richness: how specific and grounded your \"text\" is "
    "for this filename versus a vague generic sentence. "
    "Do not output confidence yourself; do not add any other keys. "
    "No markdown or code fences. Output valid JSON only."
)


def read_music_filenames(music_dir: Path) -> list[str]:
    """Read all file names from music_dir recursively, sorted and deduplicated."""
    if not music_dir.is_dir():
        raise NotADirectoryError(f"music directory does not exist: {music_dir}")

    filenames = {
        str(path.relative_to(music_dir)).replace("\\", "/")
        for path in music_dir.rglob("*")
        if path.is_file()
    }
    return sorted(filenames)


def _normalize_value(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _normalize_confidence(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0.0, min(1.0, float(value)))
    return 0.0


def _quantize_confidence(value: float) -> float:
    """Round stored confidence to two decimal places in [0, 1]."""
    return round(max(0.0, min(1.0, float(value))), 2)


def weighted_confidence(s: float, r: float, d: float) -> float:
    """confidence = 0.5*S + 0.3*R + 0.2*D (clamped to [0, 1], two decimal places)."""
    raw = CONFIDENCE_W_S * s + CONFIDENCE_W_R * r + CONFIDENCE_W_D * d
    return _quantize_confidence(raw)


def _failure_record(filename: str) -> dict[str, Any]:
    return {
        "audio": filename,
        "text": "",
        "mood": None,
        "confidence": _quantize_confidence(0.0),
    }


def _parse_model_json(raw_text: str, filename: str) -> dict[str, Any]:
    payload = json.loads(raw_text)
    if not isinstance(payload, dict):
        raise ValueError("Model response is not a JSON object")

    audio = _normalize_value(payload.get("audio")) or filename
    text_val = payload.get("text")
    text = text_val.strip() if isinstance(text_val, str) else ""
    mood = _normalize_value(payload.get("mood"))
    s = _normalize_confidence(payload.get("S"))
    r = _normalize_confidence(payload.get("R"))
    d = _normalize_confidence(payload.get("D"))
    confidence = weighted_confidence(s, r, d)
    return {
        "audio": audio,
        "text": text,
        "mood": mood,
        "confidence": confidence,
    }


async def extract_metadata_from_filename(
    client: AsyncOpenAI,
    filename: str,
    *,
    model: str,
    max_retries: int = 5,
    initial_backoff_seconds: float = 1.5,
) -> dict[str, Any]:
    """
    Extract full metadata record from one filename using Grok (xAI) with retries.
    """
    prompt = (
        "Infer metadata for this music file from the filename only.\n"
        "Return one JSON object with exactly these keys: "
        "\"audio\", \"text\", \"mood\", \"S\", \"R\", \"D\".\n\n"
        "Rules:\n"
        f"- \"audio\" must be this exact string: {json.dumps(filename)}\n"
        "- \"text\": one short English sentence like the example tone "
        '(e.g. "A slow, melancholic piano ballad with soft female vocals.").\n'
        '- \"mood\": one English mood word or short phrase, or null if unknown.\n'
        "- \"S\", \"R\", \"D\": each a number from 0 to 1 (see system message). "
        "The client will compute confidence as 0.5*S + 0.3*R + 0.2*D — do not output confidence.\n"
        "- No markdown or extra keys.\n"
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = await client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            raw = response.choices[0].message.content or "{}"
            record = _parse_model_json(raw, filename)
            record["audio"] = filename
            return record
        except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as exc:
            if attempt == max_retries:
                logging.error("xAI transient failure for '%s': %s", filename, exc)
                break
            sleep_s = initial_backoff_seconds * (2 ** (attempt - 1))
            logging.warning(
                "Retry %d/%d for '%s' after transient OpenAI error: %s",
                attempt,
                max_retries,
                filename,
                exc,
            )
            await asyncio.sleep(sleep_s)
        except (json.JSONDecodeError, ValueError, KeyError, IndexError, TypeError) as exc:
            logging.error("Invalid model JSON for '%s': %s", filename, exc)
            break
        except Exception as exc:  # noqa: BLE001
            logging.error("Unexpected error for '%s': %s", filename, exc)
            break

    # Never skip files; return a safe skeleton on failure.
    return _failure_record(filename)


async def _extract_with_semaphore(
    semaphore: asyncio.Semaphore,
    client: AsyncOpenAI,
    filename: str,
    *,
    model: str,
) -> dict[str, Any]:
    async with semaphore:
        return await extract_metadata_from_filename(client, filename, model=model)


async def process_missing_filenames(
    client: AsyncOpenAI,
    filenames: list[str],
    *,
    model: str,
    concurrency: int,
) -> dict[str, dict[str, Any]]:
    """
    Process missing filenames concurrently with bounded concurrency.
    Returns mapping: audio path -> metadata record.
    """
    if not filenames:
        return {}

    semaphore = asyncio.Semaphore(max(1, concurrency))
    tasks = [
        _extract_with_semaphore(semaphore, client, filename, model=model)
        for filename in filenames
    ]
    results = await asyncio.gather(*tasks)
    return {record["audio"]: record for record in results}


def save_metadata(output_path: Path, records: list[dict[str, Any]]) -> None:
    """Write deterministic JSON output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stable_records = sorted(records, key=lambda item: item["audio"])
    output_path.write_text(
        json.dumps(stable_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _record_from_loaded_item(item: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize one JSON object to the current schema (supports legacy rows)."""
    if "audio" in item and isinstance(item.get("audio"), str) and item["audio"].strip():
        audio = item["audio"].strip()
        text_val = item.get("text")
        text = text_val.strip() if isinstance(text_val, str) else ""

        mood = _normalize_value(item.get("mood"))
        if mood is None and isinstance(item.get("tags"), dict):
            mood = _normalize_value(item["tags"].get("mood"))

        conf = item.get("confidence")
        if conf is not None:
            confidence = _quantize_confidence(_normalize_confidence(conf))
        elif all(k in item for k in ("S", "R", "D")):
            confidence = weighted_confidence(
                _normalize_confidence(item.get("S")),
                _normalize_confidence(item.get("R")),
                _normalize_confidence(item.get("D")),
            )
        else:
            confidence = _quantize_confidence(_normalize_confidence(item.get("confidence")))

        return {
            "audio": audio,
            "text": text,
            "mood": mood,
            "confidence": confidence,
        }

    # Legacy: { "filename", "author", "title" }
    legacy = item.get("filename")
    if isinstance(legacy, str) and legacy.strip():
        audio = legacy.strip()
        return {
            "audio": audio,
            "text": "",
            "mood": None,
            "confidence": _quantize_confidence(0.0),
        }
    return None


def load_existing_metadata(output_path: Path) -> list[dict[str, Any]]:
    """
    Load existing metadata JSON array if present.
    Returns [] when file does not exist or is invalid.
    Legacy rows and older tag-based shapes are upgraded where possible.
    """
    if not output_path.exists():
        return []

    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            logging.warning("Existing metadata file is not a JSON array: %s", output_path)
            return []
    except (OSError, json.JSONDecodeError) as exc:
        logging.warning("Failed to read existing metadata file %s: %s", output_path, exc)
        return []

    records: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        rec = _record_from_loaded_item(item)
        if rec is not None:
            records.append(rec)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract music metadata (audio, text, mood, confidence) from filenames in data/music_db."
        )
    )
    parser.add_argument(
        "--music-dir",
        type=Path,
        default=settings.MUSIC_DB_DIR,
        help="Directory containing music files (default: data/music_db).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSON file path (default: data/mapping/music_metadata.json).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("XAI_MODEL", DEFAULT_MODEL),
        help=f"xAI model name (default env XAI_MODEL or {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("XAI_BASE_URL", DEFAULT_BASE_URL),
        help=f"xAI API base URL (default env XAI_BASE_URL or {DEFAULT_BASE_URL}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of files to process (for testing).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max concurrent API requests for missing files (default: 5).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help=(
            "Load existing music_metadata.json, then re-call the API for every filename in the "
            "union of (files on disk under music-dir) and (filenames already in the JSON), "
            "and write the merged, updated JSON."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    api_key = os.getenv("XAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "XAI_API_KEY is required in environment variables (recommended). "
            "OPENAI_API_KEY is also accepted for compatibility."
        )

    filenames_disk = read_music_filenames(args.music_dir)
    logging.info("Found %d files on disk under %s", len(filenames_disk), args.music_dir)

    existing_records = load_existing_metadata(args.output)
    existing_by_filename: dict[str, dict[str, Any]] = {
        row["audio"]: row for row in existing_records
    }
    logging.info("Loaded %d existing records from %s", len(existing_by_filename), args.output)

    if args.rebuild:
        # Union: keep JSON-only rows (e.g. file temporarily missing) and include new disk files.
        all_filenames = sorted(set(filenames_disk) | set(existing_by_filename.keys()))
        if args.limit is not None:
            all_filenames = all_filenames[: args.limit]
        missing_filenames = list(all_filenames)
        logging.info(
            "Rebuild mode: re-extracting via API for %d filenames (union of disk + JSON%s)",
            len(missing_filenames),
            f", limited to first {args.limit}" if args.limit is not None else "",
        )
    else:
        filenames = list(filenames_disk)
        if args.limit is not None:
            filenames = filenames[: args.limit]
        all_filenames = filenames
        missing_filenames = [name for name in filenames if name not in existing_by_filename]
        logging.info(
            "%d files already cached, %d files need API extraction",
            len(all_filenames) - len(missing_filenames),
            len(missing_filenames),
        )

    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")

    client = AsyncOpenAI(api_key=api_key, base_url=args.base_url)
    if missing_filenames:
        logging.info(
            "Starting async extraction for %d files with concurrency=%d",
            len(missing_filenames),
            args.concurrency,
        )
        new_records_by_filename = asyncio.run(
            process_missing_filenames(
                client,
                missing_filenames,
                model=args.model,
                concurrency=args.concurrency,
            )
        )
        existing_by_filename.update(new_records_by_filename)
        logging.info("Finished async extraction for %d files", len(missing_filenames))

    records = [existing_by_filename[name] for name in all_filenames]
    save_metadata(args.output, records)
    logging.info("Saved metadata for %d files to %s", len(records), args.output)


if __name__ == "__main__":
    main()
