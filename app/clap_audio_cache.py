"""
Precomputed CLAP HTSAT backbone audio features (before ``audio_projection``).

Training still runs ``audio_projection`` / ``text_projection`` with gradients; only
disk decode + frozen backbone forward are skipped.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from config import settings


def normalize_audio_manifest_key(audio_path: str) -> str:
    """Stable cache key from JSONL ``audio_path`` (relative or absolute)."""
    p = audio_path.strip().replace("\\", "/")
    marker = "music_db_15s/"
    if marker in p:
        p = p.split(marker, 1)[1]
    return p


def default_backbone_cache_dir() -> Path:
    tag = settings.CLAP_PRETRAINED_BACKBONE_FILE.stem
    return settings.EMBEDDINGS_CACHE_DIR / "clap_backbone" / tag


def resolve_audio_cache_dir(explicit: str | Path | None = None) -> Path | None:
    if explicit is not None and str(explicit).strip():
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("RAGWEB_CLAP_AUDIO_CACHE", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    default = default_backbone_cache_dir()
    if (default / "index.json").is_file():
        return default
    return None


def _blob_name(cache_key: str) -> str:
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:20]
    safe = cache_key.replace("/", "__")
    if len(safe) > 80:
        safe = safe[-80:]
    return f"{digest}_{safe}.npy"


class ClapAudioBackboneCache:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir.resolve()
        self.features_dir = self.cache_dir / "features"
        self.index_path = self.cache_dir / "index.json"
        self.meta_path = self.cache_dir / "meta.json"
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, str] = {}
        if self.index_path.is_file():
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                self._index = {str(k): str(v) for k, v in payload.items()}

    def has_key(self, cache_key: str) -> bool:
        rel = self._index.get(cache_key)
        if not rel:
            return False
        return (self.features_dir / rel).is_file()

    def keys_for_manifest(self, cache_keys: list[str]) -> tuple[int, int]:
        present = sum(1 for k in cache_keys if self.has_key(k))
        return present, len(cache_keys)

    def load_backbone_batch(
        self,
        cache_keys: list[str],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        missing = [k for k in cache_keys if not self.has_key(k)]
        if missing:
            raise KeyError(f"Missing {len(missing)} cached audio features (e.g. {missing[0]!r})")
        arrays: list[np.ndarray] = []
        for key in cache_keys:
            rel = self._index[key]
            vec = np.load(self.features_dir / rel)
            arrays.append(np.asarray(vec, dtype=np.float32).reshape(-1))
        stacked = np.stack(arrays, axis=0)
        return torch.from_numpy(stacked).to(device=device, dtype=torch.float32)

    def project_audio_batch(
        self,
        model: Any,
        cache_keys: list[str],
        *,
        tensor_mode: bool = True,
    ) -> torch.Tensor | None:
        if not cache_keys or not all(self.has_key(k) for k in cache_keys):
            return None
        try:
            device = next(model.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        backbone = self.load_backbone_batch(cache_keys, device=device)
        projected = model.model.audio_projection(backbone)
        projected = F.normalize(projected, dim=-1)
        if not tensor_mode:
            return projected.detach().cpu().numpy()
        return projected

    def write_entry(self, cache_key: str, backbone_vec: np.ndarray) -> None:
        rel = _blob_name(cache_key)
        out_path = self.features_dir / rel
        np.save(out_path, np.asarray(backbone_vec, dtype=np.float32))
        self._index[cache_key] = rel

    def save_index(self) -> None:
        self.index_path.write_text(
            json.dumps(self._index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def write_meta(self) -> None:
        meta = {
            "backbone": str(settings.CLAP_PRETRAINED_BACKBONE_FILE.resolve()),
            "feature": "encode_audio.embedding (pre audio_projection)",
            "manifest_key": "normalize_audio_manifest_key(audio_path)",
        }
        self.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def get_backbone_audio_features_from_filelist(
    model: Any,
    music_paths: list[str],
    *,
    use_tensor: bool = False,
) -> torch.Tensor | np.ndarray:
    """Frozen HTSAT output used as cache payload (before trainable ``audio_projection``)."""
    import librosa
    from laion_clap.training.data import get_audio_features, float32_to_int16, int16_to_float32

    model.model.eval()
    enable_fusion = bool(getattr(model, "enable_fusion", False))
    audio_input: list[dict[str, Any]] = []
    for f in music_paths:
        audio_waveform, _ = librosa.load(f, sr=48000)
        audio_waveform = int16_to_float32(float32_to_int16(audio_waveform))
        audio_waveform = torch.from_numpy(audio_waveform).float()
        temp_dict: dict[str, Any] = {}
        temp_dict = get_audio_features(
            temp_dict,
            audio_waveform,
            480000,
            data_truncating="fusion" if enable_fusion else "rand_trunc",
            data_filling="repeatpad",
            audio_cfg=model.model_cfg["audio_cfg"],
            require_grad=audio_waveform.requires_grad,
        )
        audio_input.append(temp_dict)

    device = next(model.model.parameters()).device
    input_dict: dict[str, torch.Tensor] = {}
    keys = audio_input[0].keys()
    for k in keys:
        input_dict[k] = torch.cat([d[k].unsqueeze(0) for d in audio_input], dim=0).to(device)
    with torch.no_grad():
        backbone = model.model.encode_audio(input_dict, device=device)["embedding"]
    if use_tensor:
        return backbone
    return backbone.detach().cpu().numpy()
