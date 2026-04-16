"""
Extract structured music metadata from filenames in data/music_db using Grok (xAI).

Output file:
    data/mapping/music_metadata.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError

from config import settings

DEFAULT_MODEL = "grok-4.20-non-reasoning"
DEFAULT_BASE_URL = "https://api.x.ai/v1"
DEFAULT_OUTPUT_PATH = settings.MAPPING_DIR / "music_metadata.json"
SYSTEM_PROMPT = (
    "You extract metadata from messy music filenames. "
    "Return strict JSON with keys: author, title. "
    "Use null for unknown values. "
    "Do not invent details that are not reasonably inferable."
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


def _parse_model_json(raw_text: str) -> tuple[str | None, str | None]:
    payload = json.loads(raw_text)
    if not isinstance(payload, dict):
        raise ValueError("Model response is not a JSON object")

    author = _normalize_value(payload.get("author"))
    title = _normalize_value(payload.get("title"))
    return author, title


def extract_metadata_from_filename(
    client: OpenAI,
    filename: str,
    *,
    model: str,
    max_retries: int = 5,
    initial_backoff_seconds: float = 1.5,
) -> dict[str, str | None]:
    """
    Extract author/title from one filename using Grok (xAI) with retries.
    """
    prompt = (
        "Extract metadata from this music filename.\n"
        "Return ONLY JSON with keys 'author' and 'title'.\n"
        "Rules:\n"
        "- If author is unknown, set author to null.\n"
        "- If title is unknown, set title to null.\n"
        "- Do not include extra keys.\n\n"
        f"filename: {filename}"
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            raw = response.choices[0].message.content or "{}"
            author, title = _parse_model_json(raw)
            return {"filename": filename, "author": author, "title": title}
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
            time.sleep(sleep_s)
        except (json.JSONDecodeError, ValueError, KeyError, IndexError, TypeError) as exc:
            logging.error("Invalid model JSON for '%s': %s", filename, exc)
            break
        except Exception as exc:  # noqa: BLE001
            logging.error("Unexpected error for '%s': %s", filename, exc)
            break

    # Never skip files; return nulls on failure.
    return {"filename": filename, "author": None, "title": None}


def save_metadata(output_path: Path, records: list[dict[str, str | None]]) -> None:
    """Write deterministic JSON output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stable_records = sorted(records, key=lambda item: item["filename"])
    output_path.write_text(
        json.dumps(stable_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract author/title metadata from filenames in data/music_db."
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

    filenames = read_music_filenames(args.music_dir)
    if args.limit is not None:
        filenames = filenames[: args.limit]

    total = len(filenames)
    logging.info("Found %d files in %s", total, args.music_dir)

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    records: list[dict[str, str | None]] = []
    for idx, filename in enumerate(filenames, start=1):
        record = extract_metadata_from_filename(client, filename, model=args.model)
        records.append(record)
        logging.info("Processed %d/%d: %s", idx, total, filename)

    save_metadata(args.output, records)
    logging.info("Saved metadata for %d files to %s", len(records), args.output)


if __name__ == "__main__":
    main()
