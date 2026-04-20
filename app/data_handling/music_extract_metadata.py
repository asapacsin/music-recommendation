"""
Extract structured music metadata from filenames in data/music_db using Grok (xAI).

Each saved record:
  audio, text, mood, confidence

The model outputs confidence (0–1) from KB-style assessment of the filename; values are
quantized to two decimal places in code.

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
    "You describe a music file using only its path/filename (no audio). "
    "Treat your broad knowledge of how music tracks are named and labeled worldwide as a knowledge base (KB) "
    "that helps you read artist/title tokens, languages, and common release patterns—but you must still "
    "ground \"text\" and \"mood\" in what the filename string actually supports. "
    "Return one JSON object with exactly these keys: \"audio\", \"text\", \"mood\", \"confidence\". "
    "- \"audio\": string — must equal the exact path string the user provides. "
    "- \"text\": string — one short English sentence (prefer about 8–15 words). Overall feel and general "
    "musical character; stress mood, atmosphere, and energy; only high-certainty inferences from the filename; "
    "slightly rich but general; do NOT guess specific instruments, vocals, or genre unless clearly indicated "
    "in the filename; if the filename gives little information, use a simple, generic but meaningful sentence. "
    "- \"mood\": string or JSON null — one concise English mood label only when explicit English mood words "
    "or clearly interpretable mood tokens appear in the filename (e.g. sad, happy, chill); non-English or "
    "poetic titles → null unless such a token is unambiguously present; never infer mood from your own \"text\". "
    "- \"confidence\": number from 0 to 1 (you output it; the client may round to two decimals). "
    "This is how trustworthy a music description is when predicted from this filename using the KB: "
    "HIGH only when the KB plus the filename together give clear, trackable music information (recognizable "
    "song/artist/title-like naming you can connect to a sensible general description). "
    "FORCED LOW: if the KB cannot tie the string to any trackable music information (no usable song/title/artist "
    "signal the KB can anchor—e.g. opaque names, bare numbers, meaningless tokens), set confidence in the "
    "rough range 0.05–0.35 (do not use high values). "
    "FORCED ZERO: if the filename is not a valid plausible music filename at all (random hash only, obvious "
    "temp/encoder junk with no music naming, garbage stem, clearly not a music track label), set confidence "
    "to exactly 0. "
    "Do not output extra keys, markdown, or code fences. Output valid JSON only."
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
    confidence = _quantize_confidence(_normalize_confidence(payload.get("confidence")))
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
        "Infer metadata for this music file from the filename only (no audio).\n"
        "Use your KB of real-world music naming to read the string, but only assert mood/text the filename supports.\n"
        "Return one JSON object with exactly these keys: \"audio\", \"text\", \"mood\", \"confidence\".\n\n"
        "Fields:\n"
        f"- \"audio\": must be exactly: {json.dumps(filename)}\n"
        "- \"text\": one sentence (~8–15 words), general feel/energy; no invented instruments/vocals/genre; "
        "generic if little signal.\n"
        "- \"mood\": English mood word/token only if clearly in the filename; else null; not from your text.\n"
        "- \"confidence\": 0–1. Use KB + filename: high only with clear trackable music naming; "
        "if KB cannot anchor trackable music info → forced LOW (~0.05–0.35); "
        "if not a valid music filename at all → exactly 0.\n"
        "- Do not output S, R, or D. No markdown or extra keys.\n"
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


def _chunked(items: list[str], chunk_size: int) -> list[list[str]]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


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
            # Legacy rows: approximate former client-side blend
            s = _normalize_confidence(item.get("S"))
            r = _normalize_confidence(item.get("R"))
            d = _normalize_confidence(item.get("D"))
            confidence = _quantize_confidence(0.5 * s + 0.3 * r + 0.2 * d)
        else:
            confidence = _quantize_confidence(0.0)

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
        "--batch-size",
        type=int,
        default=400,
        help="Save checkpoint after every N extracted files (default: 400).",
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
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    client = AsyncOpenAI(api_key=api_key, base_url=args.base_url)
    if missing_filenames:
        batches = _chunked(missing_filenames, args.batch_size)
        total_batches = len(batches)
        logging.info(
            "Starting async extraction for %d files with concurrency=%d in %d batches (batch_size=%d)",
            len(missing_filenames),
            args.concurrency,
            total_batches,
            args.batch_size,
        )

        for i, batch in enumerate(batches, start=1):
            logging.info("Processing batch %d/%d with %d files", i, total_batches, len(batch))
            batch_records = asyncio.run(
                process_missing_filenames(
                    client,
                    batch,
                    model=args.model,
                    concurrency=args.concurrency,
                )
            )
            existing_by_filename.update(batch_records)

            # Checkpoint after every batch so previous progress is preserved.
            checkpoint_records = [
                existing_by_filename[name] for name in all_filenames if name in existing_by_filename
            ]
            save_metadata(args.output, checkpoint_records)
            logging.info(
                "Saved batch %d/%d checkpoint to %s (%d records)",
                i,
                total_batches,
                args.output,
                len(checkpoint_records),
            )

        logging.info("Finished async extraction for %d files", len(missing_filenames))

    records = [existing_by_filename[name] for name in all_filenames]
    save_metadata(args.output, records)
    logging.info("Saved metadata for %d files to %s", len(records), args.output)


if __name__ == "__main__":
    main()
