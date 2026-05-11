"""
Load CLAP (HTSAT-base) for evaluation scripts.

If ``settings.CLAP_MODEL_FILE`` points to a ``torch.save`` dict containing
``model_state_dict`` (as written by ``init_model.model_creation``), load the
public backbone from ``CLAP_PRETRAINED_BACKBONE_FILE`` then overlay those weights.
Otherwise call ``load_ckpt`` on ``CLAP_MODEL_FILE`` (original pretrained file).
"""
from __future__ import annotations

import laion_clap
import torch

from config import settings


def load_clap_module_httsat() -> laion_clap.CLAP_Module:
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    path = settings.CLAP_MODEL_FILE
    try:
        blob = torch.load(str(path), map_location="cpu")
    except Exception:
        blob = None

    if isinstance(blob, dict) and "model_state_dict" in blob:
        model.load_ckpt(str(settings.CLAP_PRETRAINED_BACKBONE_FILE))
        model.model.load_state_dict(blob["model_state_dict"], strict=False)
    else:
        model.load_ckpt(str(path))
    return model
