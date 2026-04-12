"""Smoke test for config.settings path resolution."""
import sys
from pathlib import Path

# Allow `python app/test_path.py` from repo root (cwd may not include project root)
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import settings


def main() -> None:
    assert settings.BASE_DIR.is_dir(), f"BASE_DIR missing: {settings.BASE_DIR}"
    assert (settings.BASE_DIR / "config" / "settings.py").is_file()

    print("BASE_DIR:", settings.BASE_DIR)
    print("DATA_DIR:", settings.DATA_DIR)
    print("MUSIC_DB_DIR:", settings.MUSIC_DB_DIR)
    print("MUSIC_MAP_FILE:", settings.MUSIC_MAP_FILE)
    print("BEST_MODEL_FILE:", settings.BEST_MODEL_FILE)
    print("OK — path management resolves correctly.")


if __name__ == "__main__":
    main()
