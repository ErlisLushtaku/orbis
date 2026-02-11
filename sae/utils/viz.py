"""
Visualization utilities for SAE latent interpretability.

Provides reusable functions for multi-granularity SAE visualization:
- Patch heatmaps: spatial activation maps overlaid on frames
- Clip videos: mp4 sequences from frame arrays
- Frame PNGs: single-frame image saving
"""

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import matplotlib
import numpy as np
from PIL import Image

from sae.utils.constants import ORBIS_GRID_H, ORBIS_GRID_W

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def save_frame_png(frame_rgb: np.ndarray, output_path: Path) -> None:
    """Save a single raw frame as PNG.

    Args:
        frame_rgb: Raw frame array (H, W, 3) or (H, W), uint8.
        output_path: Destination path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(frame_rgb).convert("RGB")
    img.save(output_path)


def create_clip_video(
    frames_rgb: List[np.ndarray],
    output_path: Path,
    fps: int = 5,
) -> None:
    """Write a list of RGB frames to an mp4 video.

    Args:
        frames_rgb: List of (H, W, 3) uint8 numpy arrays.
        output_path: Destination .mp4 path.
        fps: Frames per second.
    """
    if not frames_rgb:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = frames_rgb[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for frame in frames_rgb:
        # Convert RGB to BGR for OpenCV
        bgr = cv2.cvtColor(
            np.ascontiguousarray(frame).astype(np.uint8), cv2.COLOR_RGB2BGR,
        )
        writer.write(bgr)

    writer.release()


def create_patch_heatmap(
    frame_rgb: np.ndarray,
    spatial_activations: np.ndarray,
    output_path: Path,
    grid_h: int = ORBIS_GRID_H,
    grid_w: int = ORBIS_GRID_W,
    title: Optional[str] = None,
) -> None:
    """Create a side-by-side image: original frame | spatial activation heatmap.

    Adapted from ViT-Prisma's image_patch_heatmap() for Orbis's non-square
    18x32 spatial grid (no CLS token).

    Args:
        frame_rgb: Raw frame (H, W, 3) uint8.
        spatial_activations: Per-token activations, shape (grid_h * grid_w,)
            or (grid_h, grid_w).
        output_path: Destination path for the side-by-side PNG.
        grid_h: Spatial grid height (default 18 for Orbis 288x512).
        grid_w: Spatial grid width (default 32 for Orbis 288x512).
        title: Optional title for the figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img_h, img_w = frame_rgb.shape[:2]

    # Reshape flat activations to spatial grid
    acts = np.asarray(spatial_activations, dtype=np.float32)
    if acts.ndim == 1:
        acts = acts.reshape(grid_h, grid_w)

    # Upscale activation grid to image resolution via nearest-neighbor
    scale_h = img_h / grid_h
    scale_w = img_w / grid_w
    heatmap = np.repeat(np.repeat(acts, int(scale_h), axis=0), int(scale_w), axis=1)

    # Handle rounding (crop to image size)
    heatmap = heatmap[:img_h, :img_w]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].imshow(frame_rgb)
    axes[0].set_title("Original Frame")
    axes[0].axis("off")

    im = axes[1].imshow(heatmap, cmap="viridis", aspect="auto")
    axes[1].set_title("SAE Latent Activation")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontsize=12)

    fig.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
