import os
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
MUSIC_DB_15S_DIR = DATA_DIR / "music_db_15s"
QUERY_INPUT_DIR = DATA_DIR / "input"
FAISS_INDEX_DIR = DATA_DIR / "index"
EMBEDDINGS_CACHE_DIR = DATA_DIR / "embeddings_cache"
FILE_NAME_DIR = DATA_DIR / "file_name"
MAPPING_DIR = DATA_DIR / "mapping"
CLAP_TRAIN_JSONL = MAPPING_DIR / "clap_train_15s.jsonl"
CLAP_VAL_JSONL = MAPPING_DIR / "clap_val_15s.jsonl"
SELF_TRAIN_DATA_DIR = DATA_DIR / "self_train"
LOG_DIR = DATA_DIR / "log"

# ---------------------------------------------------------------------------
# Index / mapping files
# ---------------------------------------------------------------------------
AUDIO_INDEX_FILE = FAISS_INDEX_DIR / "audio_index.faiss"
TEXT_INDEX_FILE = FAISS_INDEX_DIR / "text_index.faiss"
OL3_INDEX_FILE = FAISS_INDEX_DIR / "index.faiss"
METADATA_TEXT_INDEX_FILE = FAISS_INDEX_DIR / "metadata_text_index.faiss"
FILE_PATHS_FILE = FILE_NAME_DIR / "file_paths.txt"
ID_PATH_MAPPING_FILE = MAPPING_DIR / "id_path_mapping.json"
METADATA_ID_MAPPING_FILE = MAPPING_DIR / "metadata_id_mapping.json"
MUSIC_MAP_FILE = MAPPING_DIR / "music_map.txt"
MUSIC_METADATA_FILE = MAPPING_DIR / "music_metadata.json"
HUMAN_PASS_WAY_FILE = MAPPING_DIR / "human_pass_way.json"
PROCESS_META_FILE = MAPPING_DIR / "process_meta.json"

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
MODEL_DIR = BASE_DIR / "model"
# All CLAP weights under ``model/clap/``: backbone .pt at top of clap/, fine-tunes under clap/finetune/.
CLAP_DIR = MODEL_DIR / "clap"
CLAP_PRETRAINED_BACKBONE_FILE = CLAP_DIR / "music_audioset_epoch_15_esc_90.14.pt"
# Eval / retrieval: defaults to backbone; set ``RAGWEB_CLAP_CHECKPOINT`` to a ``best_model.pt`` from fine-tune.
_clap_ckpt = os.environ.get("RAGWEB_CLAP_CHECKPOINT")
CLAP_MODEL_FILE = (
    Path(_clap_ckpt).expanduser().resolve()
    if _clap_ckpt
    else CLAP_PRETRAINED_BACKBONE_FILE
)
# Local Hugging Face snapshot for caption refinement (see scripts/download_llama31_8b.sh).
LLM_HF_REPO_ID = os.environ.get(
    "RAGWEB_LLM_HF_REPO_ID", "meta-llama/Meta-Llama-3.1-8B-Instruct"
)
_llm_dir_name = os.environ.get("RAGWEB_LLM_MODEL_DIR", "llama3.1-8b-instruct")
LLM_MODEL_DIR = MODEL_DIR / _llm_dir_name
LLM_LOAD_IN_4BIT = os.environ.get("RAGWEB_LLM_4BIT", "0").strip() in ("1", "true", "yes")
BEST_MODEL_FILE = MODEL_DIR / "best_model.pt"
# Fine-tuned CLAP checkpoints live next to the backbone under model/clap/.
FINETUNE_MODEL_DIR = CLAP_DIR / "finetune"
SELF_TRAIN_MODEL_DIR = CLAP_DIR / "self_train"


def finetune_checkpoint_path(run_id: str, seed: int) -> Path:
    """Path to ``best_model.pt`` for a multi-seed fine-tune run."""
    return FINETUNE_MODEL_DIR / run_id / f"seed_{seed}" / "best_model.pt"


def finetune_log_run_dir(run_id: str) -> Path:
    """Per-run training logs (summary, params, metrics — no large weights)."""
    return LOG_DIR / "finetune_runs" / run_id


def finetune_log_seed_dir(run_id: str, seed: int) -> Path:
    return finetune_log_run_dir(run_id) / f"seed_{seed}"


def self_train_run_data_dir(run_id: str) -> Path:
    """Per-run self-train artifacts (hard_mined, train_mixed, iter_metrics)."""
    return SELF_TRAIN_DATA_DIR / run_id


def self_train_iter_data_dir(run_id: str, iter_n: int) -> Path:
    return self_train_run_data_dir(run_id) / f"iter_{iter_n}"


def self_train_log_run_dir(run_id: str) -> Path:
    return LOG_DIR / "self_train_runs" / run_id


def self_train_log_iter_dir(run_id: str, iter_n: int) -> Path:
    return self_train_log_run_dir(run_id) / f"iter_{iter_n}"


def self_train_checkpoint_path(run_id: str, iter_n: int) -> Path:
    return SELF_TRAIN_MODEL_DIR / run_id / f"iter_{iter_n}" / "best_model.pt"


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------
TEST_DATA_DIR = BASE_DIR / "tests" / "data"

# ---------------------------------------------------------------------------
# Auto-create runtime directories
# ---------------------------------------------------------------------------
for _d in [MUSIC_DB_DIR, FAISS_INDEX_DIR, QUERY_INPUT_DIR,
           EMBEDDINGS_CACHE_DIR, FILE_NAME_DIR, MAPPING_DIR, LOG_DIR,
           LOG_DIR / "finetune_runs", LOG_DIR / "self_train_runs",
           CLAP_DIR, FINETUNE_MODEL_DIR, SELF_TRAIN_DATA_DIR, SELF_TRAIN_MODEL_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
