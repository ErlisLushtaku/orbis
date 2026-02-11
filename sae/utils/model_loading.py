"""Shared model loading utilities for the SAE module.

Centralizes Orbis model and SAE checkpoint loading to avoid duplication
across train_sae.py and semantic grounding scripts.
"""

from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

from .logging import get_logger

logger = get_logger(__name__)


def load_orbis_model(
    config_path: Union[str, Path],
    ckpt_path: Union[str, Path],
    device: torch.device,
) -> nn.Module:
    """Load pre-trained Orbis world model, frozen in eval mode.

    Args:
        config_path: Path to model config.yaml
        ckpt_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Loaded model in eval mode with frozen parameters
    """
    from omegaconf import OmegaConf
    from util import instantiate_from_config

    logger.info(f"Loading config from {config_path}")
    cfg_model = OmegaConf.load(str(config_path))

    logger.info("Instantiating model...")
    model = instantiate_from_config(cfg_model.model)

    logger.info(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    logger.info("Model loaded and frozen")
    return model


def load_sae(
    checkpoint_path: Union[str, Path],
    device: torch.device,
) -> "TopKSAE":
    """Load a trained SAE from checkpoint.

    Args:
        checkpoint_path: Path to SAE checkpoint (.pt)
        device: Device to load model on

    Returns:
        Loaded SAE in eval mode with frozen parameters
    """
    # Lazy import to avoid circular dependency: topk_sae -> utils -> model_loading -> topk_sae
    from ..topk_sae import TopKSAE, TopKSAEConfig

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    if "config" not in checkpoint:
        raise ValueError(f"Unknown checkpoint format: {checkpoint.keys()}")

    config = checkpoint["config"]
    if isinstance(config, dict):
        # Filter to only known fields for backward compatibility
        known_fields = {f.name for f in TopKSAEConfig.__dataclass_fields__.values()}
        config = TopKSAEConfig(**{k: v for k, v in config.items() if k in known_fields})
    elif isinstance(config, TopKSAEConfig):
        # Ensure old checkpoints (missing new fields) get defaults
        for field_name, field_obj in TopKSAEConfig.__dataclass_fields__.items():
            if not hasattr(config, field_name):
                setattr(config, field_name, field_obj.default)
    sae = TopKSAE(config)

    state_dict_key = "state_dict" if "state_dict" in checkpoint else "model_state_dict"
    sae.load_state_dict(checkpoint[state_dict_key])

    sae.to(device)
    sae.eval()
    for param in sae.parameters():
        param.requires_grad = False

    return sae
