#!/usr/bin/env python3
"""
Feature Dashboard: Visualize what concepts each SAE latent has learned.

For each of the top-N most active latents, generates a 3x3 grid showing
the 9 video frames that caused that feature to fire most strongly.

Optionally overlays spatial heatmaps showing WHERE in the frame the feature fired.

Usage:
    python orbis/sae/feature_dashboard.py \
        --sae_checkpoint /path/to/sae.pt \
        --exp_dir /path/to/orbis_experiment \
        --data_path /path/to/Cityscapes \
        --output_dir feature_dashboard/ \
        --num_latents 50 \
        --frames_per_latent 9
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from einops import rearrange

# Add project root to path (scripts are in sae/scripts/, so go up 3 levels for project, 2 for orbis)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ORBIS_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ORBIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ORBIS_ROOT))

from omegaconf import OmegaConf
from util import instantiate_from_config
from sae.utils.logging import get_logger, setup_sae_logging

# Setup logging
setup_sae_logging()
logger = get_logger(__name__)

# Local imports (use absolute imports for script execution)
from sae.topk_sae import TopKSAE
from sae.activation_hooks import ActivationExtractor


@dataclass
class FeatureActivation:
    """Record of a feature activation."""
    frame_idx: int          # Index of the frame in the dataset
    spatial_idx: int        # Spatial position within the frame
    activation_value: float # Activation magnitude
    frame_path: str         # Path to the original image file


class FeatureTracker:
    """
    Track top activations for each feature across the dataset.
    
    For memory efficiency, only keeps the top-K activations per feature.
    """
    
    def __init__(self, d_sae: int, top_k: int = 9):
        self.d_sae = d_sae
        self.top_k = top_k
        
        # For each feature, store list of (activation_value, frame_idx, spatial_idx, frame_path)
        self.feature_activations: Dict[int, List[Tuple[float, int, int, str]]] = \
            defaultdict(list)
        
        # Also track total activation mass per feature (for ranking)
        self.feature_total_activation = torch.zeros(d_sae)
    
    def update(
        self,
        sparse_acts: torch.Tensor,  # (batch, spatial, d_sae) or (batch*spatial, d_sae)
        frame_indices: List[int],
        frame_paths: List[str],
        spatial_size: Tuple[int, int],  # (H, W) of spatial grid
    ):
        """
        Update feature tracking with new batch of activations.
        
        Args:
            sparse_acts: Sparse activations from SAE
            frame_indices: Global indices of frames in this batch
            frame_paths: Paths to original images
            spatial_size: Spatial dimensions of the activation map
        """
        # Handle shape
        if sparse_acts.dim() == 2:
            # Reshape to (batch, spatial, d_sae)
            batch_size = len(frame_indices)
            spatial_tokens = sparse_acts.shape[0] // batch_size
            sparse_acts = sparse_acts.view(batch_size, spatial_tokens, -1)
        
        batch_size, spatial_tokens, d_sae = sparse_acts.shape
        H, W = spatial_size
        
        # Update total activation mass
        self.feature_total_activation += sparse_acts.sum(dim=(0, 1)).cpu()
        
        # Find non-zero activations
        for b in range(batch_size):
            frame_idx = frame_indices[b]
            frame_path = frame_paths[b]
            
            for s in range(spatial_tokens):
                # Get activations at this position
                acts = sparse_acts[b, s]  # (d_sae,)
                
                # Find active features
                active_mask = acts > 0
                active_features = torch.where(active_mask)[0]
                
                for feat_idx in active_features.tolist():
                    act_val = acts[feat_idx].item()
                    
                    # Add to this feature's list
                    self.feature_activations[feat_idx].append(
                        (act_val, frame_idx, s, frame_path)
                    )
                    
                    # Keep only top-K for memory efficiency
                    if len(self.feature_activations[feat_idx]) > self.top_k * 2:
                        self.feature_activations[feat_idx].sort(reverse=True)
                        self.feature_activations[feat_idx] = \
                            self.feature_activations[feat_idx][:self.top_k]
    
    def get_top_features(self, n: int) -> List[int]:
        """Get indices of top-N features by total activation mass."""
        _, indices = torch.topk(self.feature_total_activation, min(n, self.d_sae))
        return indices.tolist()
    
    def get_top_activations(self, feature_idx: int) -> List[FeatureActivation]:
        """Get top activations for a specific feature."""
        activations = self.feature_activations.get(feature_idx, [])
        activations.sort(reverse=True)
        
        return [
            FeatureActivation(
                frame_idx=frame_idx,
                spatial_idx=spatial_idx,
                activation_value=act_val,
                frame_path=frame_path,
            )
            for act_val, frame_idx, spatial_idx, frame_path in activations[:self.top_k]
        ]


def create_activation_heatmap(
    activation_value: float,
    spatial_idx: int,
    spatial_size: Tuple[int, int],
    image_size: Tuple[int, int],
) -> np.ndarray:
    """
    Create a heatmap showing where in the image the feature activated.
    
    Args:
        activation_value: Activation strength (for coloring)
        spatial_idx: Flattened spatial index
        spatial_size: (H, W) of the activation grid
        image_size: (H, W) of the output heatmap
        
    Returns:
        Heatmap array of shape (image_H, image_W)
    """
    H, W = spatial_size
    img_H, img_W = image_size
    
    # Create activation map at spatial resolution
    act_map = np.zeros((H, W), dtype=np.float32)
    
    # Convert flat index to 2D
    h_idx = spatial_idx // W
    w_idx = spatial_idx % W
    
    if 0 <= h_idx < H and 0 <= w_idx < W:
        act_map[h_idx, w_idx] = activation_value
    
    # Gaussian blur for smoother visualization
    from scipy.ndimage import gaussian_filter
    act_map = gaussian_filter(act_map, sigma=1.0)
    
    # Resize to image size
    act_map_pil = Image.fromarray((act_map * 255).astype(np.uint8))
    act_map_resized = act_map_pil.resize((img_W, img_H), Image.BILINEAR)
    
    return np.array(act_map_resized, dtype=np.float32) / 255.0


def create_feature_grid(
    activations: List[FeatureActivation],
    feature_idx: int,
    spatial_size: Tuple[int, int],
    image_size: Tuple[int, int] = (288, 512),
    show_heatmap: bool = True,
    grid_rows: int = 3,
    grid_cols: int = 3,
) -> plt.Figure:
    """
    Create a grid visualization for a single feature.
    
    Args:
        activations: List of top activations for this feature
        feature_idx: Index of the feature being visualized
        spatial_size: Spatial dimensions of activation map
        image_size: Size of input images (H, W)
        show_heatmap: Whether to overlay activation heatmaps
        grid_rows: Number of rows in the grid
        grid_cols: Number of columns in the grid
        
    Returns:
        Matplotlib figure
    """
    n_images = grid_rows * grid_cols
    
    fig = plt.figure(figsize=(grid_cols * 4, grid_rows * 3 + 0.5))
    gs = gridspec.GridSpec(grid_rows, grid_cols, figure=fig)
    
    fig.suptitle(f"Feature #{feature_idx}", fontsize=14, fontweight='bold')
    
    for i, act in enumerate(activations[:n_images]):
        if i >= n_images:
            break
        
        row = i // grid_cols
        col = i % grid_cols
        ax = fig.add_subplot(gs[row, col])
        
        # Load and display image
        try:
            img = Image.open(act.frame_path).convert("RGB")
            img = img.resize((image_size[1], image_size[0]))  # (W, H) for PIL
            img_array = np.array(img)
            
            ax.imshow(img_array)
            
            # Overlay heatmap if requested
            if show_heatmap:
                heatmap = create_activation_heatmap(
                    act.activation_value,
                    act.spatial_idx,
                    spatial_size,
                    image_size,
                )
                # Normalize heatmap for display
                if heatmap.max() > 0:
                    heatmap = heatmap / heatmap.max()
                
                # Create colored overlay
                ax.imshow(heatmap, cmap='hot', alpha=0.4)
                
        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading\n{act.frame_path}", 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title(f"act={act.activation_value:.2f}", fontsize=9)
        ax.axis('off')
    
    # Fill remaining slots with empty axes
    for i in range(len(activations), n_images):
        row = i // grid_cols
        col = i % grid_cols
        ax = fig.add_subplot(gs[row, col])
        ax.axis('off')
    
    plt.tight_layout()
    return fig


@torch.no_grad()
def generate_feature_dashboard(
    model: nn.Module,
    sae: TopKSAE,
    dataloader: DataLoader,
    layer_idx: int,
    device: torch.device,
    output_dir: Path,
    num_latents: int = 50,
    frames_per_latent: int = 9,
    t_noise: float = 0.0,
    frame_rate: int = 5,
    spatial_size: Tuple[int, int] = (18, 32),  # From Orbis config
    image_size: Tuple[int, int] = (288, 512),
    show_heatmap: bool = True,
    max_batches: Optional[int] = None,
):
    """
    Generate the full feature dashboard.
    
    Args:
        model: Orbis world model
        sae: Trained SAE model
        dataloader: DataLoader for images
        layer_idx: Which layer to extract from
        device: Device to run on
        output_dir: Directory to save visualizations
        num_latents: Number of top latents to visualize
        frames_per_latent: Number of frames per latent (should be perfect square)
        t_noise: Noise timestep
        frame_rate: Frame rate conditioning
        spatial_size: Spatial dimensions of activation map
        image_size: Size of input images
        show_heatmap: Whether to show spatial heatmaps
        max_batches: Maximum batches to process
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    sae.eval()
    sae = sae.to(device)
    
    extractor = ActivationExtractor(model, layer_idx=layer_idx, flatten_spatial=False)
    tracker = FeatureTracker(sae.d_sae, top_k=frames_per_latent)
    
    logger.info(f" Scanning dataset for top activations...")
    
    global_frame_idx = 0
    
    iterator = enumerate(dataloader)
    if max_batches is not None:
        total = min(max_batches, len(dataloader))
    else:
        total = len(dataloader)
    
    for batch_idx, batch in tqdm(iterator, total=total, desc="Scanning"):
        if max_batches is not None and batch_idx >= max_batches:
            break
        
        # Handle batch format
        if isinstance(batch, (list, tuple)):
            imgs = batch[0]
        elif isinstance(batch, dict):
            imgs = batch.get('images', batch.get('image'))
        else:
            imgs = batch
        
        imgs = imgs.to(device)
        batch_size = imgs.shape[0]
        
        # Get frame paths from dataset
        if hasattr(dataloader.dataset, 'images'):
            # Cityscapes-style dataset
            start_idx = batch_idx * dataloader.batch_size
            frame_paths = [
                dataloader.dataset.images[start_idx + i] 
                for i in range(batch_size)
            ]
        else:
            frame_paths = [f"frame_{global_frame_idx + i}" for i in range(batch_size)]
        
        frame_indices = list(range(global_frame_idx, global_frame_idx + batch_size))
        global_frame_idx += batch_size
        
        # Encode frames
        x = model.encode_frames(imgs)
        
        # Setup for forward pass
        b = x.shape[0]
        t = torch.full((b,), t_noise, device=device)
        
        if t_noise > 0:
            target_t, _ = model.add_noise(x, t)
        else:
            target_t = x
        
        if target_t.dim() == 4:
            target_t = target_t.unsqueeze(1)
        
        fr = torch.full((b,), frame_rate, device=device)
        
        # Run with activation capture
        with extractor.capture():
            if target_t.shape[1] > 1:
                context = target_t[:, :-1]
                target = target_t[:, -1:]
            else:
                context = None
                target = target_t
            
            _ = model.vit(target, context, t, frame_rate=fr)
        
        # Get activations: list of (B, F, N, D) tensors
        acts_list = extractor.get_activations()
        
        # Process each activation tensor
        for acts in acts_list:
            # acts shape: (B, F, N, D) - take first frame
            if acts.dim() == 4:
                acts = acts[:, 0]  # (B, N, D)
            
            # Flatten for SAE encoding
            B, N, D = acts.shape
            acts_flat = rearrange(acts, 'b n d -> (b n) d')
            
            # Encode through SAE
            sparse_acts = sae.get_feature_activations(acts_flat.to(sae.W_enc.device))
            
            # Reshape back
            sparse_acts = rearrange(sparse_acts, '(b n) f -> b n f', b=B, n=N)
            
            # Update tracker
            tracker.update(
                sparse_acts.cpu(),
                frame_indices,
                frame_paths,
                spatial_size,
            )
    
    # Get top features
    logger.info(f" Generating visualizations for top {num_latents} features...")
    top_features = tracker.get_top_features(num_latents)
    
    # Determine grid size
    grid_size = int(np.ceil(np.sqrt(frames_per_latent)))
    
    # Generate visualizations
    feature_info = []
    for i, feat_idx in enumerate(tqdm(top_features, desc="Creating grids")):
        activations = tracker.get_top_activations(feat_idx)
        
        if not activations:
            continue
        
        # Create and save grid
        fig = create_feature_grid(
            activations,
            feat_idx,
            spatial_size,
            image_size,
            show_heatmap=show_heatmap,
            grid_rows=grid_size,
            grid_cols=grid_size,
        )
        
        save_path = output_dir / f"latent_{feat_idx:05d}.png"
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Record info
        feature_info.append({
            "feature_idx": feat_idx,
            "rank": i + 1,
            "total_activation": float(tracker.feature_total_activation[feat_idx]),
            "num_activations": len(activations),
            "top_activation_value": activations[0].activation_value if activations else 0,
        })
    
    # Save summary
    summary = {
        "num_features_visualized": len(feature_info),
        "frames_per_feature": frames_per_latent,
        "layer_idx": layer_idx,
        "features": feature_info,
    }
    
    with open(output_dir / "dashboard_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f" Feature dashboard saved to {output_dir}")
    logger.info(f"  Visualized {len(feature_info)} features")
    logger.info(f"  Top feature: #{top_features[0]} with activation mass {tracker.feature_total_activation[top_features[0]]:.2f}")


def main(args):
    """Main function for feature dashboard generation."""

    device = torch.device(args.device)
    logger.info(f" Using device: {device}")

    # Load SAE
    logger.info(f" Loading SAE from {args.sae_checkpoint}")
    sae = TopKSAE.load(args.sae_checkpoint, device=device)
    logger.info(f" SAE: {sae}")

    # Load Orbis model
    exp_dir = Path(args.exp_dir)
    config_path = exp_dir / args.config
    ckpt_path = exp_dir / args.ckpt

    logger.info(f" Loading Orbis from {exp_dir}")
    cfg_model = OmegaConf.load(config_path)
    model = instantiate_from_config(cfg_model.model)
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    # Freeze model
    for param in model.parameters():
        param.requires_grad = False

    # Create dataloader based on data source
    if args.data_source == "cityscapes":
        from semantic_stage2 import CityScapes

        dataset = CityScapes(
            size=tuple(args.input_size),
            data_path=args.data_path,
            split="val",
            mode="fine",
            target_type="semantic",
        )
    elif args.data_source == "hdf5":
        from data.custom_multiframe import (
            MultiHDF5DatasetMultiFrameFromJSONFrameRateWrapper,
        )

        dataset = MultiHDF5DatasetMultiFrameFromJSONFrameRateWrapper(
            size=tuple(args.input_size),
            samples_json=args.val_json,
            num_frames=args.num_frames,
            stored_data_frame_rate=args.stored_frame_rate,
            frame_rate=args.frame_rate,
            num_samples=args.max_samples or 500,
        )
    else:
        raise ValueError(f"Unknown data source: {args.data_source}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Get spatial size from model config
    if hasattr(model.vit, 'input_size'):
        spatial_size = tuple(model.vit.input_size)
    else:
        spatial_size = (18, 32)  # Default from Orbis config

    # Generate dashboard
    output_dir = Path(args.output_dir)

    generate_feature_dashboard(
        model=model,
        sae=sae,
        dataloader=dataloader,
        layer_idx=args.layer,
        device=device,
        output_dir=output_dir,
        num_latents=args.num_latents,
        frames_per_latent=args.frames_per_latent,
        t_noise=args.t_noise,
        frame_rate=args.frame_rate,
        spatial_size=spatial_size,
        image_size=tuple(args.input_size),
        show_heatmap=args.show_heatmap,
        max_batches=args.max_batches,
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Feature Dashboard for SAE interpretability"
    )

    # Required paths
    parser.add_argument("--sae_checkpoint", type=str, required=True,
                        help="Path to trained SAE checkpoint")
    parser.add_argument(
        "--exp_dir", type=str, required=True, help="Path to Orbis experiment directory"
    )
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save dashboard outputs")

    # Data source selection
    parser.add_argument(
        "--data_source",
        type=str,
        default="hdf5",
        choices=["cityscapes", "hdf5"],
        help="Data source: 'hdf5' (native Orbis data) or 'cityscapes'",
    )

    # Cityscapes options
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to Cityscapes dataset (required if data_source=cityscapes)",
    )

    # HDF5 options (native Orbis data)
    parser.add_argument(
        "--val_json",
        type=str,
        default=None,
        help="Path to validation samples JSON (required if data_source=hdf5)",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=6,
        help="Number of frames per sample (for HDF5 data)",
    )
    parser.add_argument(
        "--stored_frame_rate",
        type=int,
        default=5,
        help="Frame rate of stored HDF5 data",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=500,
        help="Maximum samples from HDF5 validation set",
    )

    # Orbis model paths (relative to exp_dir)
    parser.add_argument("--ckpt", type=str, default="checkpoints/last.ckpt",
                        help="Checkpoint path relative to exp_dir")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Config path relative to exp_dir")

    # Dashboard parameters
    parser.add_argument("--num_latents", type=int, default=50,
                        help="Number of top latents to visualize")
    parser.add_argument("--frames_per_latent", type=int, default=9,
                        help="Number of frames per latent grid")
    parser.add_argument("--show_heatmap", action="store_true", default=True,
                        help="Show spatial activation heatmaps")
    parser.add_argument("--no_heatmap", action="store_false", dest="show_heatmap",
                        help="Disable spatial activation heatmaps")

    # Activation extraction
    parser.add_argument("--layer", type=int, default=12,
                        help="Which transformer layer to extract from")
    parser.add_argument("--t_noise", type=float, default=0.0,
                        help="Noise timestep for denoising")
    parser.add_argument("--frame_rate", type=int, default=5,
                        help="Frame rate conditioning value")

    # Data
    parser.add_argument("--input_size", type=int, nargs=2, default=[288, 512],
                        help="Input image size (H W)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for data loading")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Maximum batches to process (for debugging)")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
