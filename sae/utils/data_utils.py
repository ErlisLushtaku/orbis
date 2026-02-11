"""Shared data utility functions for the SAE module.

Centralizes data source detection and batch processing patterns
used across training, caching, and metrics code.
"""

from typing import Union

import torch


def is_webdataset_mode(data_source: str) -> bool:
    """Check if data_source uses WebDataset format (ends with '-webdataset')."""
    return data_source.endswith("-webdataset")


def get_base_data_source(data_source: str) -> str:
    """Get base data source name (without -webdataset suffix)."""
    if data_source.endswith("-webdataset"):
        return data_source[: -len("-webdataset")]
    return data_source


def extract_batch_images(batch: Union[list, tuple, dict, torch.Tensor]) -> torch.Tensor:
    """Extract image tensor from various batch formats.

    Handles the three batch formats used across the codebase:
    - list/tuple: images are the first element
    - dict: images under 'images' or 'image' key
    - raw tensor: returned as-is
    """
    if isinstance(batch, (list, tuple)):
        return batch[0]
    elif isinstance(batch, dict):
        return batch.get("images", batch.get("image"))
    return batch


def compute_ground_truth(
    model: torch.nn.Module,
    x: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Compute diffusion velocity field ground truth target.

    Implements: A(t) * x_last + B(t) * noise_last for video (5D)
    or A(t) * x + B(t) * noise for single-frame (4D).

    Args:
        model: Orbis model with A() and B() schedule functions
        x: Encoded frames (B, T, C, H, W) or (B, C, H, W)
        noise: Noise tensor matching x shape
        t: Timestep tensor (B,)

    Returns:
        Ground truth target tensor
    """
    a_t = model.A(t).view(-1, 1, 1, 1)
    b_t = model.B(t).view(-1, 1, 1, 1)
    if x.dim() == 5:
        return a_t * x[:, -1] + b_t * noise[:, -1]
    return a_t * x + b_t * noise
