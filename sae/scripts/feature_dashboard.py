#!/usr/bin/env python3
"""
Feature Dashboard: Visualize what concepts each SAE latent has learned.

For each of the top-N most active latents, generates a grid showing
the video frames that caused that feature to fire most strongly,
with spatial activation heatmaps overlaid on each frame.

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
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
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

setup_sae_logging()
logger = get_logger(__name__)

from sae.topk_sae import TopKSAE
from sae.activation_hooks import ActivationExtractor
from sae.utils.viz import create_top_frames_grid


@dataclass
class FeatureActivation:
    """Record of a feature activation with full spatial profile."""
    frame_idx: int
    spatial_map: np.ndarray     # Full spatial activation profile (spatial_tokens,)
    activation_value: float     # Mean activation across spatial tokens (ranking score)
    frame_path: str


class FeatureTracker:
    """Track top activations for each feature across the dataset.

    For each feature, stores the top-K frames ranked by mean spatial
    activation, along with the full spatial activation map for overlay
    heatmap visualization.
    """

    def __init__(self, d_sae: int, top_k: int = 9):
        self.d_sae = d_sae
        self.top_k = top_k
        # Per feature: list of (mean_act, frame_idx, spatial_map, frame_path)
        self.feature_activations: Dict[int, List[Tuple[float, int, np.ndarray, str]]] = \
            defaultdict(list)
        self.feature_total_activation = torch.zeros(d_sae)

    def update(
        self,
        sparse_acts: torch.Tensor,
        frame_indices: List[int],
        frame_paths: List[str],
        spatial_size: Tuple[int, int],
    ) -> None:
        """Update feature tracking with a new batch of activations.

        Args:
            sparse_acts: Shape (batch, spatial, d_sae) or (batch*spatial, d_sae).
            frame_indices: Global indices of frames in this batch.
            frame_paths: Paths to original images.
            spatial_size: (H, W) of the spatial grid.
        """
        if sparse_acts.dim() == 2:
            batch_size = len(frame_indices)
            spatial_tokens = sparse_acts.shape[0] // batch_size
            sparse_acts = sparse_acts.view(batch_size, spatial_tokens, -1)

        batch_size = sparse_acts.shape[0]
        self.feature_total_activation += sparse_acts.sum(dim=(0, 1)).cpu()

        for b in range(batch_size):
            frame_acts = sparse_acts[b]  # (spatial, d_sae)
            mean_acts = frame_acts.mean(dim=0)  # (d_sae,)
            active_features = torch.where(mean_acts > 0)[0]

            frame_idx = frame_indices[b]
            frame_path = frame_paths[b]

            for feat_idx in active_features.tolist():
                score = mean_acts[feat_idx].item()
                spatial_map = frame_acts[:, feat_idx].cpu().numpy()

                self.feature_activations[feat_idx].append(
                    (score, frame_idx, spatial_map, frame_path),
                )

                if len(self.feature_activations[feat_idx]) > self.top_k * 2:
                    self.feature_activations[feat_idx].sort(
                        key=lambda x: x[0], reverse=True,
                    )
                    self.feature_activations[feat_idx] = \
                        self.feature_activations[feat_idx][:self.top_k]

    def get_top_features(self, n: int) -> List[int]:
        """Get indices of top-N features by total activation mass."""
        _, indices = torch.topk(self.feature_total_activation, min(n, self.d_sae))
        return indices.tolist()

    def get_top_activations(self, feature_idx: int) -> List[FeatureActivation]:
        """Get top activations for a specific feature, sorted by mean activation."""
        activations = self.feature_activations.get(feature_idx, [])
        activations.sort(key=lambda x: x[0], reverse=True)

        return [
            FeatureActivation(
                frame_idx=frame_idx,
                spatial_map=spatial_map,
                activation_value=score,
                frame_path=frame_path,
            )
            for score, frame_idx, spatial_map, frame_path in activations[:self.top_k]
        ]


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
    spatial_size: Tuple[int, int] = (18, 32),
    image_size: Tuple[int, int] = (288, 512),
    show_heatmap: bool = True,
    max_batches: Optional[int] = None,
):
    """Generate the full feature dashboard.

    Args:
        model: Orbis world model.
        sae: Trained SAE model.
        dataloader: DataLoader for images.
        layer_idx: Which layer to extract from.
        device: Device to run on.
        output_dir: Directory to save visualizations.
        num_latents: Number of top latents to visualize.
        frames_per_latent: Number of frames per latent grid.
        t_noise: Noise timestep.
        frame_rate: Frame rate conditioning.
        spatial_size: Spatial dimensions of activation map.
        image_size: Size of input images.
        show_heatmap: Whether to show spatial heatmaps.
        max_batches: Maximum batches to process.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    sae.eval()
    sae = sae.to(device)

    extractor = ActivationExtractor(model, layer_idx=layer_idx, flatten_spatial=False)
    tracker = FeatureTracker(sae.d_sae, top_k=frames_per_latent)

    logger.info("Scanning dataset for top activations...")

    global_frame_idx = 0

    iterator = enumerate(dataloader)
    if max_batches is not None:
        total = min(max_batches, len(dataloader))
    else:
        total = len(dataloader)

    for batch_idx, batch in tqdm(iterator, total=total, desc="Scanning"):
        if max_batches is not None and batch_idx >= max_batches:
            break

        if isinstance(batch, (list, tuple)):
            imgs = batch[0]
        elif isinstance(batch, dict):
            imgs = batch.get('images', batch.get('image'))
        else:
            imgs = batch

        imgs = imgs.to(device)
        batch_size = imgs.shape[0]

        if hasattr(dataloader.dataset, 'images'):
            start_idx = batch_idx * dataloader.batch_size
            frame_paths = [
                dataloader.dataset.images[start_idx + i]
                for i in range(batch_size)
            ]
        else:
            frame_paths = [f"frame_{global_frame_idx + i}" for i in range(batch_size)]

        frame_indices = list(range(global_frame_idx, global_frame_idx + batch_size))
        global_frame_idx += batch_size

        x = model.encode_frames(imgs)

        b = x.shape[0]
        t = torch.full((b,), t_noise, device=device)

        if t_noise > 0:
            target_t, _ = model.add_noise(x, t)
        else:
            target_t = x

        if target_t.dim() == 4:
            target_t = target_t.unsqueeze(1)

        fr = torch.full((b,), frame_rate, device=device)

        with extractor.capture():
            if target_t.shape[1] > 1:
                context = target_t[:, :-1]
                target = target_t[:, -1:]
            else:
                context = None
                target = target_t

            _ = model.vit(target, context, t, frame_rate=fr)

        acts_list = extractor.get_activations()

        for acts in acts_list:
            if acts.dim() == 4:
                acts = acts[:, 0]  # (B, N, D)

            B, N, D = acts.shape
            acts_flat = rearrange(acts, 'b n d -> (b n) d')

            sparse_acts = sae.get_feature_activations(acts_flat.to(sae.W_enc.device))
            sparse_acts = rearrange(sparse_acts, '(b n) f -> b n f', b=B, n=N)

            tracker.update(
                sparse_acts.cpu(),
                frame_indices,
                frame_paths,
                spatial_size,
            )

    logger.info(f"Generating visualizations for top {num_latents} features...")
    top_features = tracker.get_top_features(num_latents)

    grid_size = int(np.ceil(np.sqrt(frames_per_latent)))

    feature_info = []
    for i, feat_idx in enumerate(tqdm(top_features, desc="Creating grids")):
        activations = tracker.get_top_activations(feat_idx)
        if not activations:
            continue

        frames_rgb = []
        spatial_maps = []
        subtitles = []
        for act in activations:
            try:
                img = Image.open(act.frame_path).convert("RGB")
                img = img.resize((image_size[1], image_size[0]))
                frames_rgb.append(np.array(img))
                spatial_maps.append(act.spatial_map)
                subtitles.append(f"act={act.activation_value:.2f}")
            except Exception:
                logger.warning(f"Failed to load frame: {act.frame_path}")

        if frames_rgb and show_heatmap:
            save_path = output_dir / f"latent_{feat_idx:05d}.png"
            create_top_frames_grid(
                frames_rgb,
                spatial_maps,
                save_path,
                grid_h=spatial_size[0],
                grid_w=spatial_size[1],
                title=f"Feature #{feat_idx}",
                subtitles=subtitles,
                grid_rows=grid_size,
                grid_cols=grid_size,
            )

        feature_info.append({
            "feature_idx": feat_idx,
            "rank": i + 1,
            "total_activation": float(tracker.feature_total_activation[feat_idx]),
            "num_activations": len(activations),
            "top_activation_value": activations[0].activation_value if activations else 0,
        })

    summary = {
        "num_features_visualized": len(feature_info),
        "frames_per_feature": frames_per_latent,
        "layer_idx": layer_idx,
        "features": feature_info,
    }

    with open(output_dir / "dashboard_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Feature dashboard saved to {output_dir}")
    logger.info(f"  Visualized {len(feature_info)} features")
    logger.info(
        f"  Top feature: #{top_features[0]} with activation mass "
        f"{tracker.feature_total_activation[top_features[0]]:.2f}",
    )


def main(args):
    """Main function for feature dashboard generation."""

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    logger.info(f"Loading SAE from {args.sae_checkpoint}")
    sae = TopKSAE.load(args.sae_checkpoint, device=device)
    logger.info(f"SAE: {sae}")

    exp_dir = Path(args.exp_dir)
    config_path = exp_dir / args.config
    ckpt_path = exp_dir / args.ckpt

    logger.info(f"Loading Orbis from {exp_dir}")
    cfg_model = OmegaConf.load(config_path)
    model = instantiate_from_config(cfg_model.model)
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

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

    if hasattr(model.vit, 'input_size'):
        spatial_size = tuple(model.vit.input_size)
    else:
        spatial_size = (18, 32)

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

    parser.add_argument("--sae_checkpoint", type=str, required=True,
                        help="Path to trained SAE checkpoint")
    parser.add_argument(
        "--exp_dir", type=str, required=True, help="Path to Orbis experiment directory"
    )
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save dashboard outputs")

    parser.add_argument(
        "--data_source",
        type=str,
        default="hdf5",
        choices=["cityscapes", "hdf5"],
        help="Data source: 'hdf5' (native Orbis data) or 'cityscapes'",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to Cityscapes dataset (required if data_source=cityscapes)",
    )

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

    parser.add_argument("--ckpt", type=str, default="checkpoints/last.ckpt",
                        help="Checkpoint path relative to exp_dir")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Config path relative to exp_dir")

    parser.add_argument("--num_latents", type=int, default=50,
                        help="Number of top latents to visualize")
    parser.add_argument("--frames_per_latent", type=int, default=9,
                        help="Number of frames per latent grid")
    parser.add_argument("--show_heatmap", action="store_true", default=True,
                        help="Show spatial activation heatmaps")
    parser.add_argument("--no_heatmap", action="store_false", dest="show_heatmap",
                        help="Disable spatial activation heatmaps")

    parser.add_argument("--layer", type=int, default=12,
                        help="Which transformer layer to extract from")
    parser.add_argument("--t_noise", type=float, default=0.0,
                        help="Noise timestep for denoising")
    parser.add_argument("--frame_rate", type=int, default=5,
                        help="Frame rate conditioning value")

    parser.add_argument("--input_size", type=int, nargs=2, default=[288, 512],
                        help="Input image size (H W)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for data loading")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Maximum batches to process (for debugging)")

    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
