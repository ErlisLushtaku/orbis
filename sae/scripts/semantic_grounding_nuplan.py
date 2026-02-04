"""
Post-Training Semantic Grounding Analysis for Orbis SAE using NuPlan data.

Analyzes trained SAE latents by computing correlations with NuPlan odometry data
(speed, acceleration, yaw rate) on a held-out test set.

Usage:
    python sae/scripts/semantic_grounding_nuplan.py \
        --exp_dir /path/to/orbis_288x512 \
        --sae_checkpoint /path/to/sae/best.pt \
        --nuplan_data_dir /path/to/nuplan_videos \
        --output_dir /path/to/analysis_output
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from scipy import stats

# Add orbis root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sae.topk_sae import TopKSAE, TopKSAEConfig
from sae.activation_hooks import ActivationExtractor
from sae.logging_utils import get_logger, setup_sae_logging
from data.nuplan.nuplan_dataset import NuPlanOrbisMultiFrame
from util import instantiate_from_config

# Setup logging
setup_sae_logging()
logger = get_logger(__name__)


# Odometry fields to analyze
ODOMETRY_FIELDS = [
    "target_speed",
    "target_speed_kmh", 
    "target_acceleration",
    "target_yaw_rate",
    "speed_mean",
    "acceleration_mean",
    "yaw_rate_mean",
    "is_stopped_rate",
    "is_turning_rate",
]


@dataclass
class FrameActivation:
    """Activation record for a single frame."""
    video_id: str
    frame_idx: int
    latent_activations: np.ndarray
    odometry: Dict[str, float]


@dataclass 
class CorrelationResult:
    """Correlation between a latent and odometry field."""
    latent_idx: int
    field_name: str
    pearson_r: float
    p_value: float
    num_samples: int


@dataclass
class PureFeature:
    """A latent that strongly correlates with exactly one odometry field."""
    latent_idx: int
    primary_field: str
    primary_correlation: float
    other_correlations: Dict[str, float]
    top_frames: List[Dict[str, Any]] = field(default_factory=list)


def load_orbis_model(config_path: Path, ckpt_path: Path, device: torch.device) -> nn.Module:
    """Load the frozen Orbis world model."""
    from omegaconf import OmegaConf
    
    logger.info(f" Loading config from {config_path}")
    cfg_model = OmegaConf.load(config_path)
    
    logger.info(f" Instantiating model...")
    model = instantiate_from_config(cfg_model.model)
    
    logger.info(f" Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    logger.info(f" Model loaded and frozen")
    return model


def load_sae(checkpoint_path: Path, device: torch.device) -> TopKSAE:
    """Load trained SAE from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'config' in checkpoint:
        config = checkpoint['config']
        if isinstance(config, dict):
            config = TopKSAEConfig(**config)
        sae = TopKSAE(config)
        state_dict_key = 'state_dict' if 'state_dict' in checkpoint else 'model_state_dict'
        sae.load_state_dict(checkpoint[state_dict_key])
    else:
        raise ValueError(f"Unknown checkpoint format: {checkpoint.keys()}")
    
    sae.to(device)
    sae.eval()
    for param in sae.parameters():
        param.requires_grad = False
    
    return sae


class NuPlanTestDataset(torch.utils.data.Dataset):
    """Test dataset for NuPlan that returns clips with odometry."""
    
    def __init__(
        self,
        data_dir: str,
        size=(288, 512),
        target_frame_rate: int = 5,
        stored_frame_rate: int = 10,
        num_frames: int = 6,
        num_videos: Optional[int] = None,
        skip_videos: int = 0,  # Skip first N videos (for train/val split)
    ):
        # Use NuPlanOrbisMultiFrame internally
        self.dataset = NuPlanOrbisMultiFrame(
            data_dir=data_dir,
            num_frames=num_frames,
            stored_data_frame_rate=stored_frame_rate,
            target_frame_rate=target_frame_rate,
            size=size,
            num_videos=num_videos + skip_videos if num_videos else None,
            include_odometry=True,
            debug=False,
        )
        
        # Skip training videos for test set
        if skip_videos > 0 and self.dataset._video_clips:
            # Calculate clips to skip
            skip_clips = sum(nc for _, nc, _ in self.dataset._video_clips[:skip_videos])
            self.start_idx = skip_clips
            self.end_idx = len(self.dataset)
        else:
            self.start_idx = 0
            self.end_idx = len(self.dataset)
        
        logger.info(f" Using clips {self.start_idx} to {self.end_idx} ({self.end_idx - self.start_idx} total)")
    
    def __len__(self):
        return self.end_idx - self.start_idx
    
    def __getitem__(self, idx):
        sample = self.dataset[self.start_idx + idx]
        
        # Extract odometry fields
        odometry = {
            field: sample.get(field, np.nan)
            for field in ODOMETRY_FIELDS
        }
        
        return {
            "image": sample["images"],  # (T, 3, H, W)
            "video_id": sample["video_id"],
            "frame_idx": sample["frame_idx"],
            "odometry": odometry,
        }


def custom_collate(batch):
    """Custom collate for NuPlan data."""
    images = torch.stack([item["image"] for item in batch])
    video_ids = [item["video_id"] for item in batch]
    frame_idxs = [item["frame_idx"] for item in batch]
    odometry_list = [item["odometry"] for item in batch]
    
    return {
        "images": images,
        "video_ids": video_ids,
        "frame_idxs": frame_idxs,
        "odometry": odometry_list,
    }


def save_activations_cache(activations: List[FrameActivation], cache_path: Path):
    """Save collected activations to cache."""
    cache_data = {
        "version": 1,
        "num_samples": len(activations),
        "num_latents": activations[0].latent_activations.shape[0] if activations else 0,
        "samples": [
            {"video_id": a.video_id, "frame_idx": a.frame_idx, "odometry": a.odometry}
            for a in activations
        ],
        "activations": np.stack([a.latent_activations for a in activations], axis=0),
    }
    
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache_data, cache_path)
    size_mb = cache_path.stat().st_size / (1024 * 1024)
    logger.info(f" Saved {len(activations)} samples to {cache_path} ({size_mb:.1f} MB)")


def load_activations_cache(cache_path: Path) -> List[FrameActivation]:
    """Load activations from cache."""
    cache_data = torch.load(cache_path, weights_only=False)
    
    activations = []
    for i, sample in enumerate(cache_data["samples"]):
        activations.append(FrameActivation(
            video_id=sample["video_id"],
            frame_idx=sample["frame_idx"],
            latent_activations=cache_data["activations"][i],
            odometry=sample["odometry"],
        ))
    
    logger.info(f" Loaded {len(activations)} samples from {cache_path}")
    return activations


@torch.inference_mode()
def collect_activations(
    orbis_model: nn.Module,
    sae: TopKSAE,
    dataloader: DataLoader,
    layer_idx: int,
    device: torch.device,
    frame_rate: int = 5,
    cache_path: Optional[Path] = None,
) -> List[FrameActivation]:
    """Collect SAE latent activations for test clips."""
    
    if cache_path and cache_path.exists():
        logger.info(f" Found cached activations at {cache_path}")
        return load_activations_cache(cache_path)
    
    extractor = ActivationExtractor(orbis_model, layer_idx=layer_idx, flatten_spatial=True)
    all_activations = []
    
    for batch in tqdm(dataloader, desc="Collecting activations"):
        imgs = batch["images"].to(device)
        video_ids = batch["video_ids"]
        frame_idxs = batch["frame_idxs"]
        odometry_list = batch["odometry"]
        
        b = imgs.shape[0]
        
        if imgs.dim() == 4:
            imgs = imgs.unsqueeze(1)
        
        # Encode frames
        x = orbis_model.encode_frames(imgs)
        
        t_noise = torch.zeros(b, device=device)
        fr = torch.full((b,), frame_rate, device=device)
        
        # Split context and target
        if x.shape[1] > 1:
            context = x[:, :-1]
            target = x[:, -1:]
        else:
            context = x
            target = x
        
        # Extract activations
        with extractor.capture():
            _ = orbis_model.vit(target, context, t_noise, frame_rate=fr)
        
        acts = extractor.get_activations()
        spatial_tokens = acts.shape[0] // b
        acts = acts.view(b, spatial_tokens, -1)
        
        # SAE encoding
        sae_acts = sae.encode(acts.float())
        sae_acts_mean = sae_acts.mean(dim=1).cpu().numpy()
        
        for i in range(b):
            all_activations.append(FrameActivation(
                video_id=video_ids[i],
                frame_idx=frame_idxs[i],
                latent_activations=sae_acts_mean[i],
                odometry=odometry_list[i],
            ))
    
    if cache_path:
        save_activations_cache(all_activations, cache_path)
    
    return all_activations


def compute_correlations(activations: List[FrameActivation]) -> Dict[str, List[CorrelationResult]]:
    """Compute correlations between latents and odometry fields."""
    num_latents = activations[0].latent_activations.shape[0]
    num_samples = len(activations)
    
    logger.info(f" Computing correlations for {num_latents} latents, {num_samples} samples")
    
    latent_matrix = np.stack([a.latent_activations for a in activations], axis=0)
    
    odom_matrix = np.array([
        [a.odometry.get(field, np.nan) for field in ODOMETRY_FIELDS]
        for a in activations
    ])
    
    results = {field: [] for field in ODOMETRY_FIELDS}
    
    for field_idx, field_name in enumerate(ODOMETRY_FIELDS):
        field_values = odom_matrix[:, field_idx]
        valid_mask = ~np.isnan(field_values)
        
        if valid_mask.sum() < 50:
            logger.info(f"  Skipping {field_name}: only {valid_mask.sum()} valid samples")
            continue
        
        valid_values = field_values[valid_mask]
        valid_latents = latent_matrix[valid_mask]
        
        for latent_idx in range(num_latents):
            latent_values = valid_latents[:, latent_idx]
            
            if np.std(latent_values) < 1e-8:
                continue
            
            r, p = stats.pearsonr(latent_values, valid_values)
            
            results[field_name].append(CorrelationResult(
                latent_idx=latent_idx,
                field_name=field_name,
                pearson_r=r,
                p_value=p,
                num_samples=int(valid_mask.sum()),
            ))
        
        results[field_name].sort(key=lambda x: abs(x.pearson_r), reverse=True)
    
    return results


def find_pure_features(
    correlations: Dict[str, List[CorrelationResult]],
    primary_threshold: float = 0.3,
    secondary_threshold: float = 0.15,
    top_k: int = 10,
) -> List[PureFeature]:
    """Find latents strongly correlated with one field but not others."""
    
    latent_correlations = {}
    for field, results in correlations.items():
        for result in results:
            if result.latent_idx not in latent_correlations:
                latent_correlations[result.latent_idx] = {}
            latent_correlations[result.latent_idx][field] = result.pearson_r
    
    pure_features = []
    
    for latent_idx, field_corrs in latent_correlations.items():
        if not field_corrs:
            continue
        
        primary_field = max(field_corrs, key=lambda f: abs(field_corrs[f]))
        primary_corr = field_corrs[primary_field]
        
        if abs(primary_corr) < primary_threshold:
            continue
        
        other_corrs = {f: c for f, c in field_corrs.items() if f != primary_field}
        max_other = max(abs(c) for c in other_corrs.values()) if other_corrs else 0
        
        if max_other < secondary_threshold:
            pure_features.append(PureFeature(
                latent_idx=latent_idx,
                primary_field=primary_field,
                primary_correlation=primary_corr,
                other_correlations=other_corrs,
            ))
    
    pure_features.sort(key=lambda x: abs(x.primary_correlation), reverse=True)
    return pure_features[:top_k]


def find_top_activating_frames(
    activations: List[FrameActivation],
    latent_idx: int,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Find frames with highest activation for a latent."""
    
    frame_scores = [
        (i, activations[i].latent_activations[latent_idx])
        for i in range(len(activations))
    ]
    frame_scores.sort(key=lambda x: x[1], reverse=True)
    
    top_frames = []
    for idx, score in frame_scores[:top_k]:
        act = activations[idx]
        top_frames.append({
            "video_id": act.video_id,
            "frame_idx": act.frame_idx,
            "activation": float(score),
            "odometry": act.odometry,
        })
    
    return top_frames


def generate_report(
    correlations: Dict[str, List[CorrelationResult]],
    pure_features: List[PureFeature],
    output_path: Path,
):
    """Generate analysis report."""
    
    report = {
        "summary": {
            "num_odometry_fields": len(ODOMETRY_FIELDS),
            "num_pure_features": len(pure_features),
        },
        "top_correlations": {},
        "pure_features": [],
    }
    
    for field, results in correlations.items():
        report["top_correlations"][field] = [
            {
                "latent_idx": r.latent_idx,
                "pearson_r": float(r.pearson_r),
                "p_value": float(r.p_value),
            }
            for r in results[:20]
        ]
    
    for pf in pure_features:
        report["pure_features"].append({
            "latent_idx": pf.latent_idx,
            "primary_field": pf.primary_field,
            "primary_correlation": float(pf.primary_correlation),
            "other_correlations": {k: float(v) for k, v in pf.other_correlations.items()},
            "top_frames": pf.top_frames,
        })
    
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f" Saved to {output_path}")


def generate_markdown_report(
    correlations: Dict[str, List[CorrelationResult]],
    pure_features: List[PureFeature],
    activations: List[FrameActivation],
    output_path: Path,
):
    """Generate human-readable markdown report."""
    
    lines = [
        "# NuPlan SAE Semantic Grounding Analysis",
        "",
        "## Summary",
        f"- Total samples analyzed: {len(activations)}",
        f"- SAE latent dimension: {activations[0].latent_activations.shape[0]}",
        f"- Pure features found: {len(pure_features)}",
        "",
        "## Top Correlations by Odometry Field",
        "",
    ]
    
    for field in ODOMETRY_FIELDS:
        if field not in correlations or not correlations[field]:
            continue
        
        # Get human-readable field name
        field_display = {
            "target_speed": "Speed (m/s)",
            "target_speed_kmh": "Speed (km/h)",
            "target_acceleration": "Acceleration (m/s²)",
            "target_yaw_rate": "Yaw Rate (rad/s)",
            "speed_mean": "Clip Average Speed",
            "acceleration_mean": "Clip Average Acceleration",
            "yaw_rate_mean": "Clip Average Yaw Rate",
            "is_stopped_rate": "Stopped Fraction",
            "is_turning_rate": "Turning Fraction",
        }.get(field, field)
        
        lines.append(f"### {field_display}")
        lines.append("")
        lines.append("| Latent | Correlation | p-value |")
        lines.append("|--------|-------------|---------|")
        
        for r in correlations[field][:10]:
            lines.append(f"| {r.latent_idx} | {r.pearson_r:+.3f} | {r.p_value:.2e} |")
        
        lines.append("")
    
    lines.append("## Pure Features")
    lines.append("")
    lines.append("Latents that strongly correlate with one odometry field but not others:")
    lines.append("")
    
    for pf in pure_features:
        field_display = pf.primary_field.replace("_", " ").title()
        lines.append(f"### Latent {pf.latent_idx}: {field_display}")
        lines.append(f"- Primary correlation: **{pf.primary_correlation:+.3f}**")
        lines.append(f"- Other correlations: {', '.join(f'{k}: {v:+.2f}' for k, v in pf.other_correlations.items())}")
        lines.append("")
        
        if pf.top_frames:
            lines.append("**Top activating frames:**")
            lines.append("")
            for i, frame in enumerate(pf.top_frames[:5]):
                odom = frame["odometry"]
                lines.append(f"{i+1}. `{frame['video_id']}` frame {frame['frame_idx']}")
                lines.append(f"   - Activation: {frame['activation']:.3f}")
                lines.append(f"   - Speed: {odom.get('target_speed_kmh', 0):.1f} km/h")
                lines.append(f"   - Acceleration: {odom.get('target_acceleration', 0):.2f} m/s²")
                lines.append(f"   - Yaw rate: {odom.get('target_yaw_rate', 0):.4f} rad/s")
            lines.append("")
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    logger.info(f" Saved markdown to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="NuPlan SAE Semantic Grounding Analysis")
    
    parser.add_argument("--exp_dir", type=str, required=True, help="Orbis experiment directory")
    parser.add_argument("--sae_checkpoint", type=str, required=True, help="Path to SAE checkpoint")
    parser.add_argument("--nuplan_data_dir", type=str, required=True, help="NuPlan data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--layer", type=int, default=12, help="Transformer layer")
    parser.add_argument("--num_frames", type=int, default=6, help="Frames per clip")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_test_videos", type=int, default=100, help="Number of test videos")
    parser.add_argument("--skip_videos", type=int, default=0, help="Skip first N videos (for train/test split)")
    parser.add_argument("--top_k", type=int, default=20, help="Top-K features to report")
    parser.add_argument("--no_cache", action="store_true", help="Don't use cache")
    parser.add_argument("--rebuild_cache", action="store_true", help="Force rebuild cache")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    exp_dir = Path(args.exp_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set TK_WORK_DIR for tokenizer
    orbis_root = Path(__file__).resolve().parents[2]
    tk_work_dir = orbis_root / "logs_tk"
    os.environ["TK_WORK_DIR"] = str(tk_work_dir)
    
    # Load models
    orbis_model = load_orbis_model(
        exp_dir / "config.yaml",
        exp_dir / "checkpoints" / "last.ckpt",
        device,
    )
    
    sae = load_sae(Path(args.sae_checkpoint), device)
    logger.info(f" SAE loaded: {sae.config.d_in} -> {sae.config.d_sae} (k={sae.config.k})")
    
    # Create test dataset
    test_dataset = NuPlanTestDataset(
        data_dir=args.nuplan_data_dir,
        num_frames=args.num_frames,
        num_videos=args.num_test_videos,
        skip_videos=args.skip_videos,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate,
    )
    
    # Collect activations
    cache_path = None if args.no_cache else output_dir / "activations_cache.pt"
    if args.rebuild_cache and cache_path and cache_path.exists():
        cache_path.unlink()
    
    activations = collect_activations(
        orbis_model, sae, test_loader, args.layer, device,
        cache_path=cache_path,
    )
    
    # Compute correlations
    correlations = compute_correlations(activations)
    
    # Find pure features
    pure_features = find_pure_features(correlations, top_k=args.top_k)
    
    # Add top frames to pure features
    for pf in pure_features:
        pf.top_frames = find_top_activating_frames(activations, pf.latent_idx, top_k=5)
    
    # Generate reports
    generate_report(correlations, pure_features, output_dir / "analysis_results.json")
    generate_markdown_report(correlations, pure_features, activations, output_dir / "analysis_report.md")
    
    logger.info(f" Analysis complete!")
    logger.info(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
