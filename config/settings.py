from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project root: two levels up from this file (config/ -> project root)
BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
DATA_DIR = BASE_DIR / "data"
MUSIC_DB_DIR = DATA_DIR / "music_db"
QUERY_INPUT_DIR = DATA_DIR / "input"
FAISS_INDEX_DIR = DATA_DIR / "index"
EMBEDDINGS_CACHE_DIR = DATA_DIR / "embeddings_cache"
FILE_NAME_DIR = DATA_DIR / "file_name"
MAPPING_DIR = DATA_DIR / "mapping"
LOG_DIR = DATA_DIR / "log"

# ---------------------------------------------------------------------------
# Index / mapping files
# ---------------------------------------------------------------------------
AUDIO_INDEX_FILE = FAISS_INDEX_DIR / "audio_index.faiss"
TEXT_INDEX_FILE = FAISS_INDEX_DIR / "text_index.faiss"
OL3_INDEX_FILE = FAISS_INDEX_DIR / "index.faiss"
FILE_PATHS_FILE = FILE_NAME_DIR / "file_paths.txt"
ID_PATH_MAPPING_FILE = MAPPING_DIR / "id_path_mapping.json"
MUSIC_MAP_FILE = MAPPING_DIR / "music_map.txt"

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
MODEL_DIR = BASE_DIR / "model"
CLAP_MODEL_FILE = MODEL_DIR / "clap" / "music_audioset_epoch_15_esc_90.14.pt"
LLM_MODEL_DIR = MODEL_DIR / "llama3"
BEST_MODEL_FILE = MODEL_DIR / "best_model.pt"

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------
TEST_DATA_DIR = BASE_DIR / "tests" / "data"

# ---------------------------------------------------------------------------
# Auto-create runtime directories
# ---------------------------------------------------------------------------
for _d in [MUSIC_DB_DIR, FAISS_INDEX_DIR, QUERY_INPUT_DIR,
           EMBEDDINGS_CACHE_DIR, FILE_NAME_DIR, MAPPING_DIR, LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
