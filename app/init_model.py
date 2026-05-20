import glob
import json
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
        return p.resolve()
    return (settings.BASE_DIR / p).resolve()


def _load_clap_train_jsonl(jsonl_path: Path) -> tuple[list[str], list[str]]:
    """Load (absolute audio paths, text) from ``clap_train_15s.jsonl`` rows."""
    if not jsonl_path.is_file():
        raise FileNotFoundError(
            f"Training manifest not found: {jsonl_path}\n"
            "Build it with: python -m app.data_handling.music_build_train_val_from_15s"
        )
    paths: list[str] = []
    texts: list[str] = []
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
    if not paths:
        raise ValueError(f"No trainable rows in {jsonl_path} (skipped={skipped})")
    if skipped:
        print(f"warning: skipped {skipped} JSONL rows (missing audio_path or file)", flush=True)
    return paths, texts


def load_training_pairs(params: dict[str, Any]) -> tuple[list[str], list[str] | None]:
    """Default: 15s train manifest. Set ``train_jsonl`` in params or use ``music_db`` fallback."""
    use_fallback = bool(params.get("use_music_db_fallback", False))
    jsonl_raw = params.get("train_jsonl")
    jsonl_path = (
        Path(jsonl_raw).expanduser()
        if jsonl_raw
        else settings.CLAP_TRAIN_JSONL
    )
    if not use_fallback and jsonl_path.is_file():
        return _load_clap_train_jsonl(jsonl_path.resolve())
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
        return paths, None
    return _load_clap_train_jsonl(jsonl_path.resolve())


def embed_pipeline(
    music_paths: list[str],
    model,
    tensor_mode: bool = False,
    texts: list[str] | None = None,
):
    audio_embed = model.get_audio_embedding_from_filelist(x=music_paths, use_tensor=tensor_mode)
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

    Params may include: ``seed`` (int, default 42), ``metrics_path`` (str, default next to ``save_path``).
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

    model = load_original_model()
    paths, train_texts = load_training_pairs(params)
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

    # 早停配置
    early_stopping = params.get('early_stopping', {})
    early_stop_enabled = early_stopping.get('enabled', False)
    early_stop_metric = early_stopping.get('metric', 'similarity')
    early_stop_mode = early_stopping.get('mode', 'max')
    
    best_similarity = float('-inf') if early_stop_mode == 'max' else float('inf')
    best_epoch = 0
    last_loss = 0.0

    for epoch in range(num_epochs):
        for i in range(0, len(paths), batch_size):
            loss = 0
            batch_paths = paths[i : i + batch_size]
            batch_texts = train_texts[i : i + batch_size] if train_texts is not None else None
            batch_audio_embed, batch_text_embed = embed_pipeline(
                batch_paths, model, tensor_mode=True, texts=batch_texts
            )
            # Contrastive loss 
            logits = torch.mm(batch_audio_embed.squeeze(1), batch_text_embed.squeeze(1).t()) * temperature
            labels = torch.arange(batch_audio_embed.size(0)).cuda()
            loss = torch.nn.functional.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())

        audio_embed_list, text_embed_list = embed_pipeline(
            paths, model, tensor_mode=True, texts=train_texts
        )
        similarity = compute_avg_similarity(audio_embed_list, text_embed_list)
        sim_f = float(similarity.detach().cpu().item())
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {last_loss:.4f}, Similarity: {sim_f:.4f}")
        with metrics_path.open("a", encoding="utf-8") as mf:
            mf.write(
                json.dumps(
                    {
                        "seed": seed,
                        "epoch": epoch + 1,
                        "loss": last_loss,
                        "similarity": sim_f,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        
        # 保存最佳模型
        should_save = False
        if early_stop_mode == 'max':
            if similarity > best_similarity:
                should_save = True
        else:
            if similarity < best_similarity:
                should_save = True
        
        if should_save:
            best_similarity = similarity
            best_epoch = epoch + 1
            torch.save(
                {
                    "epoch": epoch,
                    "seed": seed,
                    "model_state_dict": core_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "similarity": similarity,
                },
                save_path,
            )
            print(f"保存最佳模型，相似度: {similarity.item():.4f}")

    best_sim_f = (
        float(best_similarity.detach().cpu().item())
        if isinstance(best_similarity, torch.Tensor)
        else float(best_similarity)
    )
    print(f"训练完成，最佳模型在第 {best_epoch} 轮，相似度: {best_sim_f:.4f}")
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_similarity": best_sim_f,
        "save_path": save_path,
        "metrics_path": str(metrics_path),
    }

