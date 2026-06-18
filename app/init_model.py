import glob
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import faiss
import laion_clap
import numpy as np
import torch

from config import settings


def set_seed(seed: int) -> None:
    """Deterministic-ish training for multi-seed studies (CUDA still has some nondeterminism)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# quantization
def int16_to_float32(x):
    return (x / 32767.0).astype('float32')


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype('int16')

def load_map(filenames):
    with open(settings.MUSIC_MAP_FILE, "r", encoding="utf-8") as f:
        descriptions = json.load(f)
    descriptions_list = [descriptions.get(file_name, "None") for file_name in filenames]
    return descriptions_list
    
def get_filename_list(music_path_list):
    filenames_list = [Path(path).name for path in music_path_list]
    return filenames_list

def _resolve_project_path(value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        resolved = p.resolve()
    else:
        resolved = (settings.BASE_DIR / p).resolve()
    staged_root = os.environ.get("RAGWEB_AUDIO_15S_ROOT", "").strip()
    if staged_root and "music_db_15s" in value.replace("\\", "/"):
        staged = Path(staged_root) / Path(value).name
        if staged.is_file():
            return staged.resolve()
    return resolved


def _load_clap_train_jsonl(
    jsonl_path: Path,
    *,
    with_cache_keys: bool = False,
) -> tuple[list[str], list[str]] | tuple[list[str], list[str], list[str]]:
    """Load (absolute audio paths, text) from ``clap_train_15s.jsonl`` rows."""
    if not jsonl_path.is_file():
        raise FileNotFoundError(
            f"Training manifest not found: {jsonl_path}\n"
            "Build it with: python -m app.data_handling.music_build_train_val_from_15s"
        )
    paths: list[str] = []
    texts: list[str] = []
    cache_keys: list[str] = []
    skipped = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                skipped += 1
                continue
            audio_value = row.get("audio_path")
            if not isinstance(audio_value, str) or not audio_value.strip():
                skipped += 1
                continue
            abs_path = _resolve_project_path(audio_value)
            if not abs_path.is_file():
                skipped += 1
                continue
            text = row.get("text")
            caption = text.strip() if isinstance(text, str) and text.strip() else "None"
            paths.append(str(abs_path))
            texts.append(caption)
            if with_cache_keys:
                from app.clap_audio_cache import normalize_audio_manifest_key

                cache_keys.append(normalize_audio_manifest_key(audio_value))
    if not paths:
        raise ValueError(f"No trainable rows in {jsonl_path} (skipped={skipped})")
    if skipped:
        print(f"warning: skipped {skipped} JSONL rows (missing audio_path or file)", flush=True)
    if with_cache_keys:
        return paths, texts, cache_keys
    return paths, texts


def load_training_pairs(
    params: dict[str, Any],
) -> tuple[list[str], list[str] | None, list[str] | None]:
    """Default: 15s train manifest. Returns (paths, texts, cache_keys|None)."""
    use_fallback = bool(params.get("use_music_db_fallback", False))
    jsonl_raw = params.get("train_jsonl")
    jsonl_path = (
        Path(jsonl_raw).expanduser()
        if jsonl_raw
        else settings.CLAP_TRAIN_JSONL
    )
    if not use_fallback and jsonl_path.is_file():
        paths, texts, keys = _load_clap_train_jsonl(jsonl_path.resolve(), with_cache_keys=True)
        return paths, texts, keys
    if use_fallback or not jsonl_path.is_file():
        paths = mock_path_list()
        if not paths:
            raise ValueError(
                f"No training audio: manifest missing ({jsonl_path}) and "
                f"{settings.MUSIC_DB_DIR} is empty. "
                "Build clap_train_15s.jsonl or symlink 15s audio into data/music_db."
            )
        print(
            f"warning: training from music_db glob ({len(paths)} files), not {jsonl_path}",
            flush=True,
        )
        return paths, None, None
    paths, texts, keys = _load_clap_train_jsonl(jsonl_path.resolve(), with_cache_keys=True)
    return paths, texts, keys


def mean_diagonal_similarity_batched(
    model,
    paths: list[str],
    texts: list[str] | None,
    *,
    batch_size: int = 8,
    audio_cache: Any | None = None,
    cache_keys: list[str] | None = None,
) -> float:
    """Mean diagonal cos(audio, text) without embedding the full manifest at once."""
    if not paths:
        return 0.0
    if texts is not None and len(texts) != len(paths):
        raise ValueError(f"paths/texts length mismatch: {len(paths)} vs {len(texts)}")
    if cache_keys is not None and len(cache_keys) != len(paths):
        raise ValueError(f"paths/cache_keys length mismatch: {len(paths)} vs {len(cache_keys)}")
    total = 0.0
    n = 0
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        batch_texts = texts[i : i + batch_size] if texts is not None else None
        batch_keys = cache_keys[i : i + batch_size] if cache_keys is not None else None
        audio_embed, text_embed = embed_pipeline(
            batch_paths,
            model,
            tensor_mode=True,
            texts=batch_texts,
            audio_cache=audio_cache,
            cache_keys=batch_keys,
        )
        audio = audio_embed.squeeze(1) if audio_embed.dim() > 2 else audio_embed
        text = text_embed.squeeze(1) if text_embed.dim() > 2 else text_embed
        audio = audio / audio.norm(dim=1, keepdim=True).clamp(min=1e-8)
        text = text / text.norm(dim=1, keepdim=True).clamp(min=1e-8)
        diag = (audio * text).sum(dim=1)
        total += float(diag.sum().detach().cpu().item())
        n += len(batch_paths)
    return total / n


def eval_manifest_mean_similarity(
    model,
    val_jsonl: Path | str,
    *,
    batch_size: int = 8,
    audio_cache: Any | None = None,
) -> float:
    """Mean diagonal cosine similarity on a CLAP JSONL manifest (in-memory model)."""
    paths, texts, cache_keys = _load_clap_train_jsonl(
        Path(val_jsonl).resolve(), with_cache_keys=True
    )
    return mean_diagonal_similarity_batched(
        model,
        paths,
        texts,
        batch_size=batch_size,
        audio_cache=audio_cache,
        cache_keys=cache_keys,
    )


def embed_pipeline(
    music_paths: list[str],
    model,
    tensor_mode: bool = False,
    texts: list[str] | None = None,
    audio_cache: Any | None = None,
    cache_keys: list[str] | None = None,
):
    audio_embed = None
    if audio_cache is not None and cache_keys is not None:
        audio_embed = audio_cache.project_audio_batch(
            model, cache_keys, tensor_mode=tensor_mode
        )
    if audio_embed is None:
        audio_embed = model.get_audio_embedding_from_filelist(
            x=music_paths, use_tensor=tensor_mode
        )
    if texts is None:
        filenames = get_filename_list(music_paths)
        texts = load_map(filenames)
    text_embed = model.get_text_embedding(x=texts, use_tensor=tensor_mode)
    return audio_embed, text_embed

def compute_avg_similarity(audio_embed, text_embed):
    audio_embed = torch.tensor(audio_embed)
    text_embed = torch.tensor(text_embed)
    audio_embed = audio_embed / audio_embed.norm(dim=1, keepdim=True)
    text_embed = text_embed / text_embed.norm(dim=1, keepdim=True)
    similarity = torch.mm(audio_embed, text_embed.t())
    similarity = torch.diagonal(similarity).mean()
    return similarity


def build_faiss_index(paths,audio_embed_list,text_embed_list):
    audio_embed = np.array(audio_embed_list).astype('float32')
    text_embed = np.array(text_embed_list).astype('float32')

    # Normalize embeddings
    audio_embed = normalize_embeddings(audio_embed)
    text_embed = normalize_embeddings(text_embed)

    # Build FAISS index
    audio_index = faiss.IndexFlatIP(audio_embed.shape[1])
    audio_index.add(audio_embed)

    text_index = faiss.IndexFlatIP(text_embed.shape[1])
    text_index.add(text_embed)

    # Save indices
    faiss.write_index(audio_index, str(settings.AUDIO_INDEX_FILE))
    faiss.write_index(text_index, str(settings.TEXT_INDEX_FILE))

    # Save id-path mapping
    id_path_mapping = {str(i): path for i, path in enumerate(paths)}
    with open(settings.ID_PATH_MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(id_path_mapping, f, indent=2)

    print("FAISS indices and id-path mapping saved")

def normalize_embeddings(embeddings):
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norm

def load_original_model():
    """Always start from the public pretrained backbone (ignores ``RAGWEB_CLAP_CHECKPOINT``)."""
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    model.load_ckpt(str(settings.CLAP_PRETRAINED_BACKBONE_FILE))
    return model


def load_model_from_checkpoint(init_checkpoint: str | Path | None = None):
    """Load backbone, optionally overlay ``model_state_dict`` from a fine-tuned ``best_model.pt``."""
    model = load_original_model()
    if not init_checkpoint:
        return model
    ckpt_path = Path(init_checkpoint).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"init_checkpoint not found: {ckpt_path}")
    blob = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(blob, dict) and "model_state_dict" in blob:
        model.model.load_state_dict(blob["model_state_dict"], strict=False)
        print(f"Loaded init_checkpoint: {ckpt_path}", flush=True)
    else:
        raise ValueError(f"init_checkpoint missing model_state_dict: {ckpt_path}")
    return model

def mock_path_list():
    paths = glob.glob(str(settings.MUSIC_DB_DIR / "*"))
    return paths

def get_embed(paths,model):
    batch_size = 8
    audio_embed_list = []
    text_embed_list = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i+batch_size]
        audio_embed, text_embed = embed_pipeline(batch_paths, model)
        audio_embed_list += list(audio_embed)
        text_embed_list += list(text_embed) 
    return audio_embed_list,text_embed_list

def general_pipeline():
    model = load_original_model()
    paths = mock_path_list()
    audio_embed_list, text_embed_list = get_embed(paths, model)
    build_faiss_index(paths, audio_embed_list, text_embed_list)
    audio_embed_list = normalize_embeddings(np.array(audio_embed_list)).tolist()
    text_embed_list = normalize_embeddings(np.array(text_embed_list)).tolist()
    
    similarity = compute_avg_similarity(audio_embed_list, text_embed_list)
    print(f"Total similarity: {similarity}")

def get_top_k_by_text_query(query_path, model, k=5):
    # Load mapping and index
    with open(settings.ID_PATH_MAPPING_FILE, "r", encoding="utf-8") as f:
        id_path_mapping = json.load(f)
    audio_index = faiss.read_index(str(settings.AUDIO_INDEX_FILE))

    from app import text_processing as tp

    # Get description for the query path
    filename = tp.get_music_name(query_path)
    with open(settings.MUSIC_MAP_FILE, "r", encoding="utf-8") as f:
        descriptions = json.load(f)
    query_text = descriptions.get(filename, "None")

    # Get embedding for the query text
    query_embed = model.get_text_embedding(x=[query_text], use_tensor=False)
    query_embed = normalize_embeddings(np.array(query_embed).astype('float32'))

    # Search in FAISS index
    D, I = audio_index.search(query_embed, k)
    top_k_paths = [id_path_mapping[str(idx)] for idx in I[0]]

    return top_k_paths

def top_k_sum(paths,model,k=5):
    num_match = 0 
    for path in paths:
        retrieved = set(get_top_k_by_text_query(path,k=k,model=model))
        if path in retrieved:
            num_match += 1
    return num_match/len(paths)

def cross_entropy_loss(logits, labels):
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=logits.size(1)).float()
    labels_one_hot = labels_one_hot.cuda()
    loss = -torch.sum(log_probs * labels_one_hot,dim=1) / logits.size(1)
    loss = loss.mean()
    return loss

def model_creation(params: dict[str, Any]) -> dict[str, Any]:
    """Contrastive fine-tune CLAP heads. Writes checkpoint to ``save_path`` and optional ``metrics_path``.

    Params may include: ``seed`` (int, default 42), ``metrics_path`` (str, default next to ``save_path``),
    ``init_checkpoint`` (str path to prior ``best_model.pt`` for multi-iter self-train).
    Returns a small dict with best epoch, best similarity, paths (for multi-seed drivers).
    """
    # 模型训练配置参数
    # params = {
    #     'learning_rate': 1e-4,
    #     'num_epochs': 5,
    #     'batch_size': 128,
    #     'temperature': 100,  # 对比学习温度系数
    #     'unfreeze_layers': {
    #         'audio_projection': True,
    #         'audio_transform': True,
    #         'text_projection': True,
    #         'text_transform': True
    #     },
    #     'save_path': "model/best_model.pt",
    #     'early_stopping': {
    #         'enabled': True,
    #         'metric': 'similarity',
    #         'mode': 'max'
    #     }
    # }
    seed = int(params.get("seed", 42))
    set_seed(seed)

    init_ckpt = params.get("init_checkpoint")
    model = load_model_from_checkpoint(init_ckpt)
    paths, train_texts, train_cache_keys = load_training_pairs(params)

    from app.clap_audio_cache import ClapAudioBackboneCache, resolve_audio_cache_dir

    cache_dir = resolve_audio_cache_dir(params.get("audio_cache_dir"))
    audio_cache: ClapAudioBackboneCache | None = None
    if cache_dir is not None:
        audio_cache = ClapAudioBackboneCache(cache_dir)
        if train_cache_keys:
            present, total = audio_cache.keys_for_manifest(train_cache_keys)
            if present == total:
                print(
                    f"Using precomputed backbone audio cache: {cache_dir} ({present} clips)",
                    flush=True,
                )
            else:
                print(
                    f"warning: audio cache incomplete ({present}/{total} at {cache_dir}); "
                    "falling back to MP3 decode for missing rows",
                    flush=True,
                )
        else:
            print("warning: audio cache configured but no manifest cache keys", flush=True)
    else:
        print("Audio cache not configured; training reads MP3 each batch", flush=True)

    print(f"Training on {len(paths)} clips", flush=True)
    # print("Model parameters and their requires_grad status:")
    # for name, param in model.named_parameters():
    #     print(name, param.shape, param.requires_grad)
    core_model = model.model
    for param in core_model.parameters():
        param.requires_grad = False  # 全部凍結
    
    # 根据params配置解冻指定层
    unfreeze_layers = params.get('unfreeze_layers', {})
    if unfreeze_layers.get('text_projection', False):
        for param in core_model.text_projection.parameters():
            param.requires_grad = True

    if unfreeze_layers.get('text_transform', False):
        for param in core_model.text_transform.parameters():
            param.requires_grad = True

    if unfreeze_layers.get('audio_projection', False):
        for param in core_model.audio_projection.parameters():
            param.requires_grad = True

    if unfreeze_layers.get('audio_transform', False):
        for param in core_model.audio_transform.parameters():
            param.requires_grad = True

    # 收集需要优化的参数
    trainable_params = []
    if unfreeze_layers.get('audio_projection', False):
        trainable_params += list(core_model.audio_projection.parameters())
    if unfreeze_layers.get('audio_transform', False):
        trainable_params += list(core_model.audio_transform.parameters())
    if unfreeze_layers.get('text_projection', False):
        trainable_params += list(core_model.text_projection.parameters())
    if unfreeze_layers.get('text_transform', False):
        trainable_params += list(core_model.text_transform.parameters())

    # Fine-tune the model
    learning_rate = params.get('learning_rate', 1e-4)
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
        
    num_epochs = params.get('num_epochs', 5)
    batch_size = params.get('batch_size', 128)
    temperature = params.get('temperature', 100)
    save_path = str(params.get('save_path', str(settings.BEST_MODEL_FILE)))
    save_parent = Path(save_path).resolve().parent
    save_parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(
        params.get("metrics_path", str(save_parent / "metrics.jsonl"))
    ).resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if metrics_path.is_file():
        metrics_path.unlink()

    # Early stopping / checkpoint selection
    early_stopping = params.get("early_stopping", {})
    early_stop_enabled = bool(early_stopping.get("enabled", False))
    early_stop_mode = early_stopping.get("mode", "max")
    patience = int(early_stopping.get("patience", 2))

    val_jsonl_raw = params.get("val_jsonl")
    val_jsonl_path = Path(val_jsonl_raw).resolve() if val_jsonl_raw else None
    use_val = val_jsonl_path is not None and val_jsonl_path.is_file()
    monitor = early_stopping.get(
        "monitor", "val_similarity" if use_val else "train_similarity"
    )
    val_batch_size = int(params.get("val_batch_size", min(batch_size, 32)))
    train_eval_batch_size = int(
        params.get("train_eval_batch_size", min(batch_size, 32))
    )
    min_epochs = int(early_stopping.get("min_epochs", 0))

    skip_train_sim_raw = params.get("skip_train_similarity_eval")
    if skip_train_sim_raw is None:
        skip_train_sim = monitor == "val_similarity" and use_val
    else:
        skip_train_sim = bool(skip_train_sim_raw)

    best_metric = float("-inf") if early_stop_mode == "max" else float("inf")
    best_train_similarity = float("-inf")
    best_val_similarity: float | None = None
    best_epoch = 0
    last_loss = 0.0
    epochs_without_improve = 0

    def _is_better(candidate: float, best: float) -> bool:
        if early_stop_mode == "max":
            return candidate > best
        return candidate < best

    for epoch in range(num_epochs):
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i : i + batch_size]
            batch_texts = train_texts[i : i + batch_size] if train_texts is not None else None
            batch_keys = (
                train_cache_keys[i : i + batch_size] if train_cache_keys is not None else None
            )
            batch_audio_embed, batch_text_embed = embed_pipeline(
                batch_paths,
                model,
                tensor_mode=True,
                texts=batch_texts,
                audio_cache=audio_cache,
                cache_keys=batch_keys,
            )
            logits = torch.mm(
                batch_audio_embed.squeeze(1), batch_text_embed.squeeze(1).t()
            ) * temperature
            device = batch_audio_embed.device
            labels = torch.arange(batch_audio_embed.size(0), device=device)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())

        if skip_train_sim:
            train_sim_f = float("nan")
            train_similarity = float("nan")
        else:
            train_sim_f = mean_diagonal_similarity_batched(
                model,
                paths,
                train_texts,
                batch_size=train_eval_batch_size,
                audio_cache=audio_cache,
                cache_keys=train_cache_keys,
            )
            train_similarity = train_sim_f
            best_train_similarity = max(best_train_similarity, train_sim_f)

        val_sim_f: float | None = None
        if use_val:
            val_sim_f = eval_manifest_mean_similarity(
                model,
                val_jsonl_path,
                batch_size=val_batch_size,
                audio_cache=audio_cache,
            )
            best_val_similarity = (
                val_sim_f
                if best_val_similarity is None
                else max(best_val_similarity, val_sim_f)
            )

        if monitor == "val_similarity" and val_sim_f is not None:
            checkpoint_metric = val_sim_f
        else:
            checkpoint_metric = train_sim_f

        train_str = f"{train_sim_f:.4f}" if math.isfinite(train_sim_f) else "n/a"
        val_str = f"{val_sim_f:.4f}" if val_sim_f is not None else "n/a"
        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {last_loss:.4f}, "
            f"TrainSim: {train_str}, ValSim: {val_str}, Monitor: {checkpoint_metric:.4f}",
            flush=True,
        )

        metric_record: dict[str, Any] = {
            "seed": seed,
            "epoch": epoch + 1,
            "loss": last_loss,
        }
        if math.isfinite(train_sim_f):
            metric_record["train_similarity"] = train_sim_f
            metric_record["similarity"] = train_sim_f
        if val_sim_f is not None:
            metric_record["val_similarity"] = val_sim_f
        metric_record["checkpoint_metric"] = checkpoint_metric
        metric_record["monitor"] = monitor

        with metrics_path.open("a", encoding="utf-8") as mf:
            mf.write(json.dumps(metric_record, ensure_ascii=False) + "\n")

        if _is_better(checkpoint_metric, best_metric):
            best_metric = checkpoint_metric
            best_epoch = epoch + 1
            epochs_without_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "seed": seed,
                    "model_state_dict": core_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_similarity": train_similarity,
                    "val_similarity": val_sim_f,
                    "similarity": train_similarity,
                    "checkpoint_metric": checkpoint_metric,
                    "monitor": monitor,
                },
                save_path,
            )
            print(
                f"保存最佳模型，monitor={monitor}，metric={checkpoint_metric:.4f}",
                flush=True,
            )
        else:
            epochs_without_improve += 1

        if (
            early_stop_enabled
            and use_val
            and monitor == "val_similarity"
            and epochs_without_improve >= patience
            and (epoch + 1) >= min_epochs
        ):
            print(
                f"Val early stop at epoch {epoch+1} (patience={patience}, "
                f"min_epochs={min_epochs})",
                flush=True,
            )
            break

    print(
        f"训练完成，最佳模型在第 {best_epoch} 轮，monitor={monitor}，metric={best_metric:.4f}",
        flush=True,
    )
    complete_path = metrics_path.parent / "training_complete.json"
    complete_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "best_epoch": best_epoch,
                "best_similarity": best_metric,
                "monitor": monitor,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_similarity": best_metric,
        "best_train_similarity": best_train_similarity,
        "best_val_similarity": best_val_similarity,
        "monitor": monitor,
        "save_path": save_path,
        "metrics_path": str(metrics_path),
    }

