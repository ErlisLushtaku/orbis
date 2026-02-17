"""
Visualization utilities for SAE latent interpretability.

Provides reusable functions for multi-granularity SAE visualization:
- Overlay heatmaps: spatial activation maps overlaid on frames
- Top frames grids: grid of top-activating frames with overlaid heatmaps
- Clip videos: mp4 sequences from frame arrays
- Frame PNGs: single-frame image saving
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import cv2
import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import torch
from PIL import Image

from sae.utils.constants import ORBIS_GRID_H, ORBIS_GRID_W

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Heatmap overlay utilities
# Adapted from the overcomplete library by Thomas Fel
# (https://github.com/tfel/overcomplete)
# Sources: visualization/cmaps.py, visualization/plot_utils.py, data.py
# ---------------------------------------------------------------------------


def _to_npf32(tensor: Union[np.ndarray, torch.Tensor, Image.Image]) -> np.ndarray:
    """Convert torch, PIL, or numpy input to float32 numpy array."""
    if isinstance(tensor, np.ndarray) and tensor.dtype == np.float32:
        return tensor
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy().astype(np.float32)
    return np.array(tensor).astype(np.float32)


def _np_channel_last(arr: Union[np.ndarray, torch.Tensor, Image.Image]) -> np.ndarray:
    """Ensure channels-last numpy float32 array."""
    arr = _to_npf32(arr)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2:
        return arr[:, :, None]
    if arr.shape[0] == 3 or arr.shape[0] == 1:
        return np.moveaxis(arr, 0, -1)
    assert arr.ndim == 3, f"Expected 3 dimensions, but got {arr.shape}"
    return arr


def _normalize(image: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Min-max normalize image to 0-1 range."""
    image = np.array(image, dtype=np.float32)
    image -= image.min()
    image /= image.max() + eps
    return image


def _interpolate_heatmap(
    heatmap: np.ndarray,
    target_wh: tuple,
    interpolation: int = cv2.INTER_CUBIC,
) -> np.ndarray:
    """Bicubic-upscale a 2D heatmap to target (width, height).

    Args:
        heatmap: 2D float array (h, w).
        target_wh: Target size as (width, height) for cv2.resize.
        interpolation: OpenCV interpolation flag.
    """
    heatmap = np.asarray(heatmap, dtype=np.float32)
    assert heatmap.ndim == 2, f"Expected 2D heatmap, got shape {heatmap.shape}"
    return cv2.resize(heatmap, target_wh, interpolation=interpolation)


def create_alpha_cmap(
    color_input: Union[str, tuple],
    name: Optional[str] = None,
) -> mcolors.LinearSegmentedColormap:
    """Create a colormap with alpha gradient from 0 (transparent) to 1 (opaque).

    Args:
        color_input: A matplotlib colormap name (str) or an RGB tuple.
        name: Optional name for the resulting colormap.
    """
    if isinstance(color_input, str):
        base_cmap = matplotlib.colormaps[color_input]
    elif isinstance(color_input, tuple) and len(color_input) == 3:
        if np.max(color_input) > 1:
            color_input = (
                color_input[0] / 255,
                color_input[1] / 255,
                color_input[2] / 255,
            )
        if name is None:
            name = f"RGB{color_input}"
        base_cmap = mcolors.LinearSegmentedColormap.from_list(
            name, [color_input, color_input],
        )
    else:
        raise ValueError("color_input must be a colormap name (str) or RGB tuple.")

    alpha = np.linspace(0.0, 1.0, base_cmap.N)
    colors = base_cmap(np.arange(base_cmap.N))
    colors[:, -1] = alpha
    return mcolors.LinearSegmentedColormap.from_list(
        name or "alpha_cmap", colors,
    )


VIRIDIS_ALPHA = create_alpha_cmap("viridis")
JET_ALPHA = create_alpha_cmap("jet")


# ---------------------------------------------------------------------------
# Public API -- frame / video saving
# ---------------------------------------------------------------------------


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
        bgr = cv2.cvtColor(
            np.ascontiguousarray(frame).astype(np.uint8), cv2.COLOR_RGB2BGR,
        )
        writer.write(bgr)

    writer.release()


# ---------------------------------------------------------------------------
# Public API -- overlay heatmap visualizations
# ---------------------------------------------------------------------------


def create_overlay_heatmap(
    frame_rgb: np.ndarray,
    spatial_activations: np.ndarray,
    output_path: Path,
    grid_h: int = ORBIS_GRID_H,
    grid_w: int = ORBIS_GRID_W,
    cmap: Optional[mcolors.Colormap] = None,
    title: Optional[str] = None,
) -> None:
    """Create a frame with spatial activation heatmap overlaid directly on it.

    Uses bicubic interpolation for smooth upscaling and an alpha colormap
    so low-activation regions are transparent and high-activation regions
    are visibly colored on top of the original frame.

    Args:
        frame_rgb: Raw frame (H, W, 3) uint8.
        spatial_activations: Per-token activations, shape (grid_h * grid_w,)
            or (grid_h, grid_w).
        output_path: Destination path for the overlay PNG.
        grid_h: Spatial grid height (default 18 for Orbis 288x512).
        grid_w: Spatial grid width (default 32 for Orbis 288x512).
        cmap: Matplotlib colormap for the heatmap. Defaults to VIRIDIS_ALPHA.
        title: Optional title for the figure.
    """
    if cmap is None:
        cmap = VIRIDIS_ALPHA

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img_h, img_w = frame_rgb.shape[:2]

    acts = np.asarray(spatial_activations, dtype=np.float32)
    if acts.ndim == 1:
        acts = acts.reshape(grid_h, grid_w)

    heatmap = _interpolate_heatmap(acts, (img_w, img_h))

    frame_norm = _normalize(_np_channel_last(frame_rgb))

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.imshow(frame_norm)
    ax.imshow(heatmap, cmap=cmap, alpha=1.0)
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def create_top_frames_grid(
    frames_rgb: List[np.ndarray],
    spatial_maps: List[np.ndarray],
    output_path: Path,
    grid_h: int = ORBIS_GRID_H,
    grid_w: int = ORBIS_GRID_W,
    cmap: Optional[mcolors.Colormap] = None,
    title: Optional[str] = None,
    subtitles: Optional[List[str]] = None,
    grid_rows: int = 2,
    grid_cols: int = 5,
) -> None:
    """Create a grid of frames with overlaid activation heatmaps.

    Accepts pre-ranked frames (no internal re-ranking). Each subplot shows
    the original frame with its spatial activation heatmap overlaid using
    an alpha colormap and bicubic interpolation.

    Args:
        frames_rgb: List of (H, W, 3) uint8 frames, already ranked.
        spatial_maps: List of per-token activations, each (grid_h*grid_w,)
            or (grid_h, grid_w). Must match length of frames_rgb.
        output_path: Destination path for the grid PNG.
        grid_h: Spatial grid height.
        grid_w: Spatial grid width.
        cmap: Matplotlib colormap for heatmaps. Defaults to VIRIDIS_ALPHA.
        title: Optional suptitle for the figure.
        subtitles: Optional per-subplot titles (e.g. activation values).
        grid_rows: Number of rows in the grid.
        grid_cols: Number of columns in the grid.
    """
    if cmap is None:
        cmap = VIRIDIS_ALPHA

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_slots = grid_rows * grid_cols
    n_images = min(len(frames_rgb), n_slots)

    fig, axes = plt.subplots(
        grid_rows, grid_cols,
        figsize=(grid_cols * 4, grid_rows * 3 + 0.5),
    )
    if grid_rows == 1 and grid_cols == 1:
        axes = np.array([[axes]])
    elif grid_rows == 1 or grid_cols == 1:
        axes = np.atleast_2d(axes)
        if grid_rows == 1:
            axes = axes.reshape(1, -1)
        else:
            axes = axes.reshape(-1, 1)

    for i in range(n_slots):
        row, col = divmod(i, grid_cols)
        ax = axes[row, col]

        if i < n_images:
            frame = frames_rgb[i]
            img_h, img_w = frame.shape[:2]

            acts = np.asarray(spatial_maps[i], dtype=np.float32)
            if acts.ndim == 1:
                acts = acts.reshape(grid_h, grid_w)
            heatmap = _interpolate_heatmap(acts, (img_w, img_h))

            frame_norm = _normalize(_np_channel_last(frame))
            ax.imshow(frame_norm)
            ax.imshow(heatmap, cmap=cmap, alpha=1.0)

            if subtitles and i < len(subtitles):
                ax.set_title(subtitles[i], fontsize=9)

        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
