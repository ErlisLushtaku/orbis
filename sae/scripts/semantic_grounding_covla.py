"""
Post-Training Semantic Grounding Analysis for Orbis SAE.

This script analyzes trained SAE latents by computing correlations with
CoVLA metadata fields (tunnel, highway, pedestrian, risk confidence scores)
on a held-out test set.

Requirements:
- Trained SAE checkpoint
- Orbis world model checkpoint
- Held-out test videos (not used in SAE training)
- CoVLA captions with confidence scores

Usage:
    python sae/semantic_grounding.py \
        --exp_dir /path/to/orbis_288x512 \
        --sae_checkpoint /path/to/sae/best.pt \
        --test_videos_dir /path/to/test_videos \
        --test_captions_dir /path/to/test_captions \
        --output_dir /path/to/analysis_output \
        --top_k 10 \
        --batch_size 4
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from scipy import stats

# Add orbis root to path for imports (scripts are in sae/scripts/, go up 2 levels)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sae.topk_sae import TopKSAE, TopKSAEConfig
from sae.activation_hooks import ActivationExtractor
from sae.caching import prepare_activation_cache, load_activation_cache, resolve_cache_dtype
from sae.logging_utils import get_logger, setup_sae_logging
from data.covla.covla_dataset import CoVLAOrbisMultiFrame
from util import instantiate_from_config

# Setup logging
setup_sae_logging()
logger = get_logger(__name__)


# Metadata fields to analyze (confidence scores)
METADATA_FIELDS = [
    "is_tunnel_yes_rate",
    "is_highway_yes_rate",
    "has_pedestrian_yes_rate",
    "risk_yes_rate",
]

# Additional fields for context
CONTEXT_FIELDS = [
    "weather_rate",
    "road_rate",
]


@dataclass
class FrameActivation:
    """Activation record for a single frame."""
    video_id: str
    frame_idx: int
    latent_activations: np.ndarray  # (num_latents,) - mean over spatial
    metadata: Dict[str, float]
    rich_caption: str = ""


@dataclass 
class CorrelationResult:
    """Correlation between a latent and metadata field."""
    latent_idx: int
    field_name: str
    pearson_r: float
    p_value: float
    num_samples: int


@dataclass
class PureFeature:
    """A latent that strongly correlates with exactly one metadata field."""
    latent_idx: int
    primary_field: str
    primary_correlation: float
    other_correlations: Dict[str, float]
    top_frames: List[Dict[str, Any]] = field(default_factory=list)


def load_orbis_model(
    config_path: Path,
    ckpt_path: Path,
    device: torch.device,
) -> nn.Module:
    """Load the frozen Orbis world model."""
    from omegaconf import OmegaConf
    
    logger.info(f" Loading config from {config_path}")
    cfg_model = OmegaConf.load(config_path)
    
    logger.info(f" Instantiating model...")
    model = instantiate_from_config(cfg_model.model)
    
    # Load checkpoint
    logger.info(f" Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    
    model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    
    # Freeze all parameters
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
        # Handle both 'state_dict' and 'model_state_dict' keys
        state_dict_key = 'state_dict' if 'state_dict' in checkpoint else 'model_state_dict'
        sae.load_state_dict(checkpoint[state_dict_key])
    else:
        raise ValueError(f"Unknown checkpoint format: {checkpoint.keys()}")
    
    sae.to(device)
    sae.eval()
    
    for param in sae.parameters():
        param.requires_grad = False
    
    return sae


def load_frame_metadata(captions_dir: Path, video_id: str, frame_idx: int) -> Dict[str, Any]:
    """Load metadata for a specific frame from caption file."""
    caption_file = captions_dir / f"{video_id}.jsonl"
    
    if not caption_file.exists():
        return {}
    
    with open(caption_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            # Each line is {frame_idx: {metadata}}
            frame_key = str(frame_idx)
            if frame_key in entry:
                return entry[frame_key]
    
    return {}


class TestDataset(torch.utils.data.Dataset):
    """
    Dataset for test videos that returns CLIPS of frames with metadata.
    
    Returns sequences of frames to match training context window.
    Metadata corresponds to the LAST (target) frame in each sequence.
    """
    
    def __init__(
        self,
        videos_dir: str,
        captions_dir: str,
        split_file: Optional[str] = None,
        size: Tuple[int, int] = (288, 512),
        target_frame_rate: int = 5,
        stored_frame_rate: int = 20,
        num_frames: int = 6,  # Total frames per clip (context + target)
        max_videos: Optional[int] = None,
    ):
        from decord import VideoReader, cpu
        from torchvision import transforms
        
        self.videos_dir = Path(videos_dir)
        self.captions_dir = Path(captions_dir)
        self.size = size
        self.num_frames = num_frames
        self.target_frame_rate = target_frame_rate
        self.stored_frame_rate = stored_frame_rate
        self.frame_interval = stored_frame_rate // target_frame_rate  # e.g., 20/5 = 4
        
        # Get video IDs from split file or directory
        if split_file:
            with open(split_file, 'r') as f:
                self.video_ids = [json.loads(line)["video_id"] for line in f]
            logger.info(f" Loaded {len(self.video_ids)} video IDs from split file")
        else:
            video_files = sorted(self.videos_dir.glob("*.mp4"))
            self.video_ids = [v.stem for v in video_files]
        
        if max_videos:
            self.video_ids = self.video_ids[:max_videos]
        
        # Build index of all clips to process
        # Each entry is (video_id, target_frame_idx) where target is the LAST frame of the clip
        self.clip_index: List[Tuple[str, int]] = []
        
        # Minimum frame index to have enough context frames
        min_frame_for_context = (self.num_frames - 1) * self.frame_interval
        
        logger.info(f" Indexing {len(self.video_ids)} test videos (context={num_frames-1}, target=1)...")
        for video_id in tqdm(self.video_ids, desc="Indexing"):
            video_path = self.videos_dir / f"{video_id}.mp4"
            try:
                vr = VideoReader(str(video_path), ctx=cpu(0))
                total_frames = len(vr)
                # Start from min_frame to ensure we have enough context
                for target_frame_idx in range(min_frame_for_context, total_frames, self.frame_interval):
                    self.clip_index.append((video_id, target_frame_idx))
            except Exception as e:
                logger.warning(f" Could not read {video_path}: {e}")
        
        logger.info(f" Total clips to process: {len(self.clip_index)}")
        
        # Setup transforms (aspect-ratio aware resize + center crop)
        source_size = (1928, 1208)  # CoVLA resolution (W, H)
        resize_size = self._get_resize_size(source_size, size)
        
        self.transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
        ])
    
    @staticmethod
    def _get_resize_size(source_size: tuple, target_size: tuple) -> tuple:
        """Calculate resize dimensions for aspect-ratio-aware scaling."""
        src_w, src_h = source_size
        tgt_h, tgt_w = target_size
        scale_w = tgt_w / src_w
        scale_h = tgt_h / src_h
        scale = max(scale_w, scale_h)
        return (int(src_h * scale), int(src_w * scale))
    
    def __len__(self) -> int:
        return len(self.clip_index)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        from decord import VideoReader, cpu
        
        video_id, target_frame_idx = self.clip_index[idx]
        video_path = self.videos_dir / f"{video_id}.mp4"
        
        vr = VideoReader(str(video_path), ctx=cpu(0))
        
        # Calculate frame indices for the sequence
        # Goes from oldest to newest: [context_0, context_1, ..., context_n-1, target]
        frame_indices = [
            target_frame_idx - (i * self.frame_interval)
            for i in range(self.num_frames - 1, -1, -1)
        ]
        
        # Load batch of frames efficiently
        frames = vr.get_batch(frame_indices).asnumpy()
        
        # Transform each frame
        tensor_frames = []
        for frame in frames:
            img = Image.fromarray(frame).convert("RGB")
            tensor = self.transform(img)
            tensor = tensor * 2 - 1  # [0,1] -> [-1,1]
            tensor_frames.append(tensor)
        
        # Stack: (T, 3, H, W) where T = num_frames
        clip_tensor = torch.stack(tensor_frames)
        
        # Load metadata for the TARGET frame (last frame in sequence)
        metadata = load_frame_metadata(self.captions_dir, video_id, target_frame_idx)
        
        return {
            "image": clip_tensor,  # (T, 3, H, W)
            "video_id": video_id,
            "frame_idx": target_frame_idx,  # Target frame index
            "metadata": metadata,
        }


def custom_collate(batch: List[Dict]) -> Dict:
    """Custom collate that handles metadata dicts and multi-frame clips."""
    # Stack images: handles both (3, H, W) and (T, 3, H, W)
    images = torch.stack([item["image"] for item in batch])
    video_ids = [item["video_id"] for item in batch]
    frame_idxs = [item["frame_idx"] for item in batch]
    metadata_list = [item["metadata"] for item in batch]
    
    return {
        "images": images,  # (B, T, 3, H, W) or (B, 3, H, W)
        "video_ids": video_ids,
        "frame_idxs": frame_idxs,
        "metadata": metadata_list,
    }


def save_sample_metadata(
    metadata_list: List[Dict[str, Any]],
    cache_dir: Path,
) -> None:
    """Save per-sample metadata alongside Orbis activation cache."""
    metadata_path = cache_dir / "_sample_metadata.pt"
    torch.save({"samples": metadata_list, "num_samples": len(metadata_list)}, metadata_path)
    logger.info(f" Saved metadata for {len(metadata_list)} samples to {metadata_path}")


def load_sample_metadata(cache_dir: Path) -> List[Dict[str, Any]]:
    """Load per-sample metadata from cache directory."""
    metadata_path = cache_dir / "_sample_metadata.pt"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Sample metadata not found at {metadata_path}")
    data = torch.load(metadata_path, weights_only=False)
    logger.info(f" Loaded metadata for {data['num_samples']} samples from {metadata_path}")
    return data["samples"]


def get_test_cache_dir(
    exp_dir: Path,
    data_source: str,
    layer_idx: int,
) -> Path:
    """Get the standard cache directory for test set activations.
    
    Returns: logs_sae/sae_cache/{data_source}/{model_name}/layer_{layer}/test/
    """
    orbis_root = Path(__file__).resolve().parents[2]
    model_name = exp_dir.name
    cache_dir = orbis_root / "logs_sae" / "sae_cache" / data_source / model_name / f"layer_{layer_idx}" / "test"
    return cache_dir


@torch.inference_mode()
def cache_orbis_activations_with_metadata(
    orbis_model: nn.Module,
    dataloader: DataLoader,
    cache_dir: Path,
    layer_idx: int,
    device: torch.device,
    frame_rate: int = 5,
    rebuild: bool = False,
) -> Tuple[List[Path], List[Dict[str, Any]]]:
    """
    Cache raw Orbis activations (same format as train/val) plus per-sample metadata.
    
    This stores:
    - Orbis activations in batch_XXXXXX.pt files (same as training cache)
    - Per-sample metadata in _sample_metadata.pt
    
    Returns:
        Tuple of (list of activation file paths, list of sample metadata dicts)
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_file = cache_dir / "_meta.json"
    sample_meta_file = cache_dir / "_sample_metadata.pt"
    
    # Check for existing cache
    existing_files = sorted(cache_dir.glob("batch_*.pt"))
    if existing_files and meta_file.exists() and sample_meta_file.exists() and not rebuild:
        logger.info(f" Using existing Orbis activation cache from {cache_dir} ({len(existing_files)} files)")
        sample_metadata = load_sample_metadata(cache_dir)
        return existing_files, sample_metadata
    
    if rebuild:
        logger.info(f" Rebuilding Orbis activation cache at {cache_dir}")
    else:
        logger.info(f" Creating Orbis activation cache at {cache_dir}")
    
    # Clean stale cache
    for path in cache_dir.glob("batch_*.pt"):
        path.unlink()
    if meta_file.exists():
        meta_file.unlink()
    if sample_meta_file.exists():
        sample_meta_file.unlink()
    
    orbis_model.eval()
    extractor = ActivationExtractor(orbis_model, layer_idx=layer_idx, flatten_spatial=True)
    
    saved_paths: List[Path] = []
    all_sample_metadata: List[Dict[str, Any]] = []
    total_tokens = 0
    hidden_dim: Optional[int] = None
    dtype = torch.float16
    dtype_name = "float16"
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Caching Orbis activations")):
        imgs = batch["images"].to(device)  # (B, T, 3, H, W)
        video_ids = batch["video_ids"]
        frame_idxs = batch["frame_idxs"]
        metadata_list = batch["metadata"]
        
        b = imgs.shape[0]
        
        # Handle both single-frame (B, 3, H, W) and multi-frame (B, T, 3, H, W) inputs
        if imgs.dim() == 4:
            imgs = imgs.unsqueeze(1)
        
        t_frames = imgs.shape[1]
        
        # Encode frames through tokenizer
        x = orbis_model.encode_frames(imgs)
        
        # Prepare for transformer (no noise for analysis)
        t_noise = torch.zeros(b, device=device)
        fr = torch.full((b,), frame_rate, device=device)
        
        # Split into context and target
        if t_frames > 1:
            context = x[:, :-1]
            target = x[:, -1:]
        else:
            warnings.warn("Single-frame input detected - activations may be out-of-distribution")
            context = x
            target = x
        
        # Extract activations from transformer
        with extractor.capture():
            _ = orbis_model.vit(target, context, t_noise, frame_rate=fr)
        
        # Get activations: (batch * spatial, hidden_dim)
        acts = extractor.get_activations()
        acts = acts.to(dtype=dtype).cpu()
        
        if hidden_dim is None:
            hidden_dim = acts.shape[-1]
        
        # Save activations to disk (same format as training cache)
        path = cache_dir / f"batch_{batch_idx:06d}.pt"
        torch.save({"activations": acts}, path)
        saved_paths.append(path)
        total_tokens += acts.shape[0]
        
        # Collect per-sample metadata
        spatial_tokens = acts.shape[0] // b
        for i in range(b):
            metadata = metadata_list[i]
            meta_scores = {
                field: metadata.get(field, np.nan)
                for field in METADATA_FIELDS + CONTEXT_FIELDS
            }
            all_sample_metadata.append({
                "video_id": video_ids[i],
                "frame_idx": frame_idxs[i],
                "metadata": meta_scores,
                "rich_caption": metadata.get("rich_caption", ""),
                "spatial_tokens": spatial_tokens,
            })
    
    # Save cache metadata
    meta = {
        "num_files": len(saved_paths),
        "total_tokens": total_tokens,
        "hidden_dim": hidden_dim,
        "dtype": dtype_name,
        "layer_idx": layer_idx,
        "t_noise": 0.0,
        "frame_rate": frame_rate,
    }
    import json as json_module
    with meta_file.open("w", encoding="utf-8") as f:
        json_module.dump(meta, f, indent=2)
    
    # Save sample metadata
    save_sample_metadata(all_sample_metadata, cache_dir)
    
    logger.info(f" Saved {len(saved_paths)} files with {total_tokens:,} total tokens")
    
    return sorted(saved_paths), all_sample_metadata


@torch.inference_mode()
def collect_activations(
    orbis_model: nn.Module,
    sae: TopKSAE,
    dataloader: DataLoader,
    layer_idx: int,
    device: torch.device,
    frame_rate: int = 5,
    cache_dir: Optional[Path] = None,
    rebuild_cache: bool = False,
) -> List[FrameActivation]:
    """
    Process test video clips and collect SAE latent activations per target frame.
    
    Caches raw Orbis activations in sae_cache/test/ (same format as train/val).
    Encodes through SAE on-the-fly when loading from cache.
    
    Args:
        orbis_model: Orbis world model
        sae: Trained SAE model
        dataloader: DataLoader returning clips with metadata
        layer_idx: Transformer layer to extract from
        device: Device to run on
        frame_rate: Frame rate conditioning value
        cache_dir: Directory for Orbis activation cache (logs_sae/sae_cache/.../test/)
        rebuild_cache: Force rebuild of cache even if it exists
    
    Returns list of FrameActivation objects with SAE latent activations and metadata.
    """
    # Cache or load Orbis activations
    if cache_dir is not None:
        cache_files, sample_metadata = cache_orbis_activations_with_metadata(
            orbis_model=orbis_model,
            dataloader=dataloader,
            cache_dir=cache_dir,
            layer_idx=layer_idx,
            device=device,
            frame_rate=frame_rate,
            rebuild=rebuild_cache,
        )
        
        # Load cached activations and encode through SAE
        logger.info(f" Encoding cached activations through SAE...")
        all_activations: List[FrameActivation] = []
        sample_idx = 0
        
        for cache_file in tqdm(cache_files, desc="Processing cached activations"):
            data = torch.load(cache_file, weights_only=True)
            acts = data["activations"].to(device).float()  # (batch * spatial, hidden_dim)
            
            # Determine batch size from metadata
            # Each sample has spatial_tokens entries in the activations
            batch_start = sample_idx
            while sample_idx < len(sample_metadata):
                meta = sample_metadata[sample_idx]
                spatial_tokens = meta["spatial_tokens"]
                
                # Check if we have enough activations left for this sample
                acts_needed = (sample_idx - batch_start + 1) * spatial_tokens
                if acts_needed > acts.shape[0]:
                    break
                
                # Extract this sample's activations
                start_idx = (sample_idx - batch_start) * spatial_tokens
                end_idx = start_idx + spatial_tokens
                sample_acts = acts[start_idx:end_idx]  # (spatial, hidden_dim)
                
                # Encode through SAE
                sae_acts = sae.encode(sample_acts)  # (spatial, num_latents)
                sae_acts_mean = sae_acts.mean(dim=0).cpu().numpy()  # (num_latents,)
                
                frame_act = FrameActivation(
                    video_id=meta["video_id"],
                    frame_idx=meta["frame_idx"],
                    latent_activations=sae_acts_mean,
                    metadata=meta["metadata"],
                    rich_caption=meta["rich_caption"],
                )
                all_activations.append(frame_act)
                sample_idx += 1
        
        return all_activations
    
    # No caching - compute directly (fallback)
    extractor = ActivationExtractor(orbis_model, layer_idx=layer_idx, flatten_spatial=True)
    all_activations: List[FrameActivation] = []
    
    for batch in tqdm(dataloader, desc="Collecting activations"):
        imgs = batch["images"].to(device)
        video_ids = batch["video_ids"]
        frame_idxs = batch["frame_idxs"]
        metadata_list = batch["metadata"]
        
        b = imgs.shape[0]
        
        if imgs.dim() == 4:
            imgs = imgs.unsqueeze(1)
        
        t_frames = imgs.shape[1]
        x = orbis_model.encode_frames(imgs)
        
        t_noise = torch.zeros(b, device=device)
        fr = torch.full((b,), frame_rate, device=device)
        
        if t_frames > 1:
            context = x[:, :-1]
            target = x[:, -1:]
        else:
            warnings.warn("Single-frame input detected")
            context = x
            target = x
        
        with extractor.capture():
            _ = orbis_model.vit(target, context, t_noise, frame_rate=fr)
        
        acts = extractor.get_activations()
        spatial_tokens = acts.shape[0] // b
        acts = acts.view(b, spatial_tokens, -1)
        
        sae_acts = sae.encode(acts.float())
        sae_acts_mean = sae_acts.mean(dim=1).cpu().numpy()
        
        for i in range(b):
            metadata = metadata_list[i]
            meta_scores = {
                field: metadata.get(field, np.nan)
                for field in METADATA_FIELDS + CONTEXT_FIELDS
            }
            
            frame_act = FrameActivation(
                video_id=video_ids[i],
                frame_idx=frame_idxs[i],
                latent_activations=sae_acts_mean[i],
                metadata=meta_scores,
                rich_caption=metadata.get("rich_caption", ""),
            )
            all_activations.append(frame_act)
    
    return all_activations


def compute_correlations(
    activations: List[FrameActivation],
    min_samples: int = 100,
) -> Dict[str, List[CorrelationResult]]:
    """
    Compute Pearson correlation between each latent and each metadata field.
    
    Returns dict mapping field_name to list of CorrelationResults (sorted by |r|).
    """
    num_latents = activations[0].latent_activations.shape[0]
    num_samples = len(activations)
    
    logger.info(f" Computing correlations for {num_latents} latents across {num_samples} samples")
    
    # Build arrays for vectorized computation
    # latent_matrix: (num_samples, num_latents)
    latent_matrix = np.stack([a.latent_activations for a in activations], axis=0)
    
    # metadata_matrix: (num_samples, num_fields)
    metadata_matrix = np.array([
        [a.metadata.get(field, np.nan) for field in METADATA_FIELDS]
        for a in activations
    ])
    
    results: Dict[str, List[CorrelationResult]] = {field: [] for field in METADATA_FIELDS}
    
    for field_idx, field_name in enumerate(tqdm(METADATA_FIELDS, desc="Computing correlations")):
        field_values = metadata_matrix[:, field_idx]
        
        # Filter out NaN values
        valid_mask = ~np.isnan(field_values)
        if valid_mask.sum() < min_samples:
            logger.info(f"  Warning: {field_name} has only {valid_mask.sum()} valid samples")
            continue
        
        valid_field = field_values[valid_mask]
        valid_latents = latent_matrix[valid_mask]
        
        # Compute correlation for each latent
        for latent_idx in range(num_latents):
            latent_values = valid_latents[:, latent_idx]
            
            # Skip if latent has no variance
            if np.std(latent_values) < 1e-8:
                continue
            
            r, p = stats.pearsonr(latent_values, valid_field)
            
            results[field_name].append(CorrelationResult(
                latent_idx=latent_idx,
                field_name=field_name,
                pearson_r=r,
                p_value=p,
                num_samples=int(valid_mask.sum()),
            ))
        
        # Sort by absolute correlation
        results[field_name].sort(key=lambda x: abs(x.pearson_r), reverse=True)
    
    return results


def find_pure_features(
    correlations: Dict[str, List[CorrelationResult]],
    high_threshold: float = 0.5,
    low_threshold: float = 0.2,
) -> List[PureFeature]:
    """
    Find 'pure features': latents with high correlation (|r| > high_threshold) 
    with exactly one field and low correlation (|r| < low_threshold) with others.
    """
    # Build correlation lookup: latent_idx -> {field: r}
    latent_correlations: Dict[int, Dict[str, float]] = {}
    
    for field_name, results in correlations.items():
        for result in results:
            if result.latent_idx not in latent_correlations:
                latent_correlations[result.latent_idx] = {}
            latent_correlations[result.latent_idx][field_name] = result.pearson_r
    
    pure_features: List[PureFeature] = []
    
    for latent_idx, field_corrs in latent_correlations.items():
        # Find fields with high correlation
        high_corr_fields = [
            (field, r) for field, r in field_corrs.items()
            if abs(r) >= high_threshold
        ]
        
        # Check if exactly one high correlation
        if len(high_corr_fields) != 1:
            continue
        
        primary_field, primary_r = high_corr_fields[0]
        
        # Check all others are low
        other_corrs = {
            field: r for field, r in field_corrs.items()
            if field != primary_field
        }
        
        if all(abs(r) < low_threshold for r in other_corrs.values()):
            pure_features.append(PureFeature(
                latent_idx=latent_idx,
                primary_field=primary_field,
                primary_correlation=primary_r,
                other_correlations=other_corrs,
            ))
    
    # Sort by primary correlation strength
    pure_features.sort(key=lambda x: abs(x.primary_correlation), reverse=True)
    
    return pure_features


def find_top_activating_latents(
    activations: List[FrameActivation],
    top_n: int = 50,
    metric: str = "max",
) -> List[Dict[str, Any]]:
    """
    Find latents with the highest overall activation across all frames.
    
    This helps discover concepts that may not correlate with predefined metadata fields.
    
    Args:
        activations: List of frame activations
        top_n: Number of top latents to return
        metric: How to rank latents - "max" (maximum activation) or "mean" (mean activation)
    
    Returns:
        List of dicts with latent_idx, score, and statistics
    """
    num_latents = activations[0].latent_activations.shape[0]
    
    # Build latent matrix: (num_samples, num_latents)
    latent_matrix = np.stack([a.latent_activations for a in activations], axis=0)
    
    # Compute score per latent
    if metric == "max":
        scores = np.max(latent_matrix, axis=0)
    elif metric == "mean":
        scores = np.mean(latent_matrix, axis=0)
    elif metric == "std":
        # High variance latents are often interesting
        scores = np.std(latent_matrix, axis=0)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Get top indices
    top_indices = np.argsort(scores)[::-1][:top_n]
    
    results = []
    for latent_idx in top_indices:
        latent_acts = latent_matrix[:, latent_idx]
        results.append({
            "latent_idx": int(latent_idx),
            "score": float(scores[latent_idx]),
            "max": float(np.max(latent_acts)),
            "mean": float(np.mean(latent_acts)),
            "std": float(np.std(latent_acts)),
            "sparsity": float(np.mean(latent_acts > 0)),  # fraction of frames where active
        })
    
    return results


def analyze_top_latents(
    activations: List[FrameActivation],
    videos_dir: Path,
    output_dir: Path,
    top_n_latents: int = 20,
    top_k_frames: int = 10,
    metric: str = "max",
) -> str:
    """
    Analyze top activating latents overall (not tied to specific metadata fields).
    
    Saves frames and generates a report for manual inspection of learned concepts.
    """
    from decord import VideoReader, cpu
    
    logger.info(f" Finding top {top_n_latents} latents by {metric} activation...")
    top_latents = find_top_activating_latents(activations, top_n=top_n_latents, metric=metric)
    
    # Create output directory
    top_latents_dir = output_dir / "top_latents"
    top_latents_dir.mkdir(parents=True, exist_ok=True)
    
    lines = [
        "# Top Activating Latents Analysis",
        "",
        f"**Metric:** {metric}",
        f"**Total latents analyzed:** {activations[0].latent_activations.shape[0]}",
        f"**Total frames:** {len(activations)}",
        "",
        "This report shows latents with the highest overall activation, regardless of correlation with metadata.",
        "These may represent visual concepts not captured by the predefined metadata fields.",
        "",
        "---",
        "",
    ]
    
    for i, latent_info in enumerate(tqdm(top_latents, desc="Processing top latents")):
        latent_idx = latent_info["latent_idx"]
        
        lines.extend([
            f"## #{i+1}: Latent {latent_idx}",
            "",
            f"- **{metric.capitalize()} activation:** {latent_info['score']:.4f}",
            f"- **Max:** {latent_info['max']:.4f}",
            f"- **Mean:** {latent_info['mean']:.4f}",
            f"- **Std:** {latent_info['std']:.4f}",
            f"- **Sparsity:** {latent_info['sparsity']:.2%} of frames active",
            "",
        ])
        
        # Find top activating frames
        top_frames = find_top_activating_frames(activations, latent_idx, top_k_frames)
        
        # Save frames
        frame_dir = top_latents_dir / f"latent_{latent_idx}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        
        for j, frame_info in enumerate(top_frames):
            video_path = videos_dir / f"{frame_info['video_id']}.mp4"
            if video_path.exists():
                try:
                    vr = VideoReader(str(video_path), ctx=cpu(0))
                    frame = vr[frame_info["frame_idx"]].asnumpy()
                    img = Image.fromarray(frame)
                    save_path = frame_dir / f"rank{j+1}_{frame_info['video_id']}_frame{frame_info['frame_idx']}.png"
                    img.save(save_path)
                except Exception as e:
                    logger.warning(f" Could not save frame: {e}")
        
        # Add frames and captions to report
        lines.append("### Top Activating Frames")
        lines.append("")
        
        for j, frame_info in enumerate(top_frames):
            video_id = frame_info["video_id"]
            frame_idx = frame_info["frame_idx"]
            activation = frame_info["activation_value"]
            caption = frame_info["rich_caption"] if frame_info["rich_caption"] else "*No caption*"
            
            img_path = f"top_latents/latent_{latent_idx}/rank{j+1}_{video_id}_frame{frame_idx}.png"
            
            lines.extend([
                f"**Rank {j+1}:** `{video_id}` frame {frame_idx} (activation: {activation:.4f})",
                f"",
                f"![frame]({img_path})",
                "",
                f"> {caption}",
                "",
            ])
        
        lines.append("---")
        lines.append("")
    
    # Save latent info as JSON
    with open(output_dir / "top_latents.json", 'w') as f:
        json.dump(top_latents, f, indent=2)
    
    report = "\n".join(lines)
    
    # Save report
    report_path = output_dir / "top_latents_analysis.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"report] Top latents analysis saved to {report_path}")
    
    return report


def find_top_activating_frames(
    activations: List[FrameActivation],
    latent_idx: int,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Find top-K frames that most strongly activate a given latent."""
    # Sort by activation value for this latent
    sorted_acts = sorted(
        activations,
        key=lambda x: x.latent_activations[latent_idx],
        reverse=True,
    )
    
    top_frames = []
    for act in sorted_acts[:top_k]:
        top_frames.append({
            "video_id": act.video_id,
            "frame_idx": act.frame_idx,
            "activation_value": float(act.latent_activations[latent_idx]),
            "rich_caption": act.rich_caption,
            "metadata": act.metadata,
        })
    
    return top_frames


def save_top_frames(
    top_frames: List[Dict[str, Any]],
    videos_dir: Path,
    output_dir: Path,
    latent_idx: int,
    field_name: str,
) -> List[str]:
    """Save images of top-activating frames."""
    from decord import VideoReader, cpu
    
    frame_dir = output_dir / "frames" / f"latent_{latent_idx}_{field_name}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    for i, frame_info in enumerate(top_frames):
        video_path = videos_dir / f"{frame_info['video_id']}.mp4"
        
        if not video_path.exists():
            continue
        
        try:
            vr = VideoReader(str(video_path), ctx=cpu(0))
            frame = vr[frame_info["frame_idx"]].asnumpy()
            img = Image.fromarray(frame)
            
            save_path = frame_dir / f"rank{i+1}_{frame_info['video_id']}_frame{frame_info['frame_idx']}.png"
            img.save(save_path)
            saved_paths.append(str(save_path))
        except Exception as e:
            logger.warning(f" Could not save frame: {e}")
    
    return saved_paths


def generate_report(
    correlations: Dict[str, List[CorrelationResult]],
    pure_features: List[PureFeature],
    activations: List[FrameActivation],
    videos_dir: Path,
    output_dir: Path,
    top_k: int = 10,
    top_latents_per_field: int = 3,
) -> str:
    """Generate comprehensive analysis report."""
    
    report_lines = [
        "=" * 80,
        "SEMANTIC GROUNDING ANALYSIS REPORT",
        "=" * 80,
        "",
        f"Total frames analyzed: {len(activations)}",
        f"Number of latents: {activations[0].latent_activations.shape[0]}",
        "",
    ]
    
    # Per-field analysis
    for field_name in METADATA_FIELDS:
        report_lines.extend([
            "",
            "-" * 60,
            f"FIELD: {field_name}",
            "-" * 60,
        ])
        
        field_results = correlations.get(field_name, [])
        
        if not field_results:
            report_lines.append("  No valid correlations computed.")
            continue
        
        report_lines.append(f"\nTop {top_latents_per_field} correlated latents:")
        
        for i, result in enumerate(field_results[:top_latents_per_field]):
            report_lines.extend([
                f"\n  #{i+1}: Latent {result.latent_idx}",
                f"      Pearson r = {result.pearson_r:.4f} (p = {result.p_value:.2e})",
                f"      N = {result.num_samples}",
            ])
            
            # Find top activating frames for this latent
            top_frames = find_top_activating_frames(activations, result.latent_idx, top_k)
            
            # Save frames
            saved_paths = save_top_frames(
                top_frames, videos_dir, output_dir,
                result.latent_idx, field_name,
            )
            
            report_lines.append(f"      Top {top_k} activating frames saved to: {saved_paths[0] if saved_paths else 'N/A'}")
            
            # Show top 3 captions
            report_lines.append("      Top activating frame captions:")
            for j, frame in enumerate(top_frames[:3]):
                caption = frame["rich_caption"][:100] + "..." if len(frame["rich_caption"]) > 100 else frame["rich_caption"]
                report_lines.append(f"        {j+1}. [{frame['video_id']}:{frame['frame_idx']}] {caption}")
    
    # Pure features section
    report_lines.extend([
        "",
        "=" * 60,
        "PURE FEATURES (high correlation with one field only)",
        "=" * 60,
        f"\nFound {len(pure_features)} pure features",
    ])
    
    # Group by primary field
    by_field: Dict[str, List[PureFeature]] = {field: [] for field in METADATA_FIELDS}
    for pf in pure_features:
        by_field[pf.primary_field].append(pf)
    
    for field_name, features in by_field.items():
        report_lines.append(f"\n{field_name}: {len(features)} pure latents")
        for pf in features[:3]:
            report_lines.append(
                f"  - Latent {pf.latent_idx}: r={pf.primary_correlation:.3f}, "
                f"others: {', '.join(f'{k}={v:.2f}' for k,v in pf.other_correlations.items())}"
            )
    
    report = "\n".join(report_lines)
    
    # Save report
    report_path = output_dir / "semantic_grounding_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"report] Saved to {report_path}")
    
    return report


def generate_detailed_frame_analysis(
    correlations: Dict[str, List[CorrelationResult]],
    activations: List[FrameActivation],
    output_dir: Path,
    top_k: int = 10,
    top_latents_per_field: int = 3,
) -> str:
    """
    Generate detailed analysis of top activating frames with full captions.
    
    Creates a markdown file that can be easily reviewed for manual analysis.
    """
    lines = [
        "# Detailed Frame Analysis",
        "",
        f"**Total frames analyzed:** {len(activations)}",
        f"**Number of latents:** {activations[0].latent_activations.shape[0]}",
        "",
        "---",
        "",
    ]
    
    for field_name in METADATA_FIELDS:
        lines.extend([
            f"## {field_name}",
            "",
        ])
        
        field_results = correlations.get(field_name, [])
        
        if not field_results:
            lines.append("*No valid correlations computed.*\n")
            continue
        
        for i, result in enumerate(field_results[:top_latents_per_field]):
            latent_idx = result.latent_idx
            r = result.pearson_r
            p = result.p_value
            
            lines.extend([
                f"### Latent {latent_idx} (r = {r:.4f}, p = {p:.2e})",
                "",
            ])
            
            # Find top activating frames
            top_frames = find_top_activating_frames(activations, latent_idx, top_k)
            
            frame_dir = output_dir / "frames" / f"latent_{latent_idx}_{field_name}"
            
            # List each frame with full details
            for j, frame in enumerate(top_frames):
                video_id = frame["video_id"]
                frame_idx = frame["frame_idx"]
                activation = frame["activation_value"]
                caption = frame["rich_caption"] if frame["rich_caption"] else "*No caption available*"
                
                # Image path (relative for markdown)
                img_path = f"frames/latent_{latent_idx}_{field_name}/rank{j+1}_{video_id}_frame{frame_idx}.png"
                
                # Metadata scores
                meta = frame.get("metadata", {})
                meta_items = []
                for k, v in meta.items():
                    if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                        meta_items.append(f"{k}={v:.2f}")
                meta_str = ", ".join(meta_items)
                
                lines.extend([
                    f"**Rank {j+1}:** `{video_id}` frame {frame_idx}",
                    f"- Activation: {activation:.4f}",
                    f"- Metadata: {meta_str}" if meta_str else "- Metadata: N/A",
                    f"- Image: ![frame]({img_path})",
                    "",
                    f"> {caption}",
                    "",
                ])
            
            lines.append("---\n")
    
    report = "\n".join(lines)
    
    # Save as markdown
    report_path = output_dir / "detailed_frame_analysis.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"report] Detailed frame analysis saved to {report_path}")
    
    return report


def main(args: argparse.Namespace):
    """Main analysis function."""
    
    device = torch.device(args.device)
    logger.info(f" Using device: {device}")
    
    exp_dir = Path(args.exp_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    logger.info("Loading Orbis model...")
    config_path = exp_dir / args.config
    ckpt_path = exp_dir / args.ckpt
    orbis_model = load_orbis_model(config_path, ckpt_path, device)
    
    logger.info("Loading SAE...")
    sae = load_sae(Path(args.sae_checkpoint), device)
    logger.info(f"  SAE config: d_in={sae.config.d_in}, d_sae={sae.config.d_sae}, k={sae.config.k}")
    
    # Create test dataset
    logger.info("Creating test dataset...")
    test_dataset = TestDataset(
        videos_dir=args.test_videos_dir,
        captions_dir=args.test_captions_dir,
        split_file=args.split_file,
        size=tuple(args.input_size),
        target_frame_rate=args.frame_rate,
        stored_frame_rate=args.stored_frame_rate,
        num_frames=args.num_frames,  # Context window to match training
        max_videos=args.max_videos,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate,
        pin_memory=True,
    )
    
    # Determine cache directory (same location as train/val caches)
    cache_dir = None
    if not args.no_cache:
        cache_dir = get_test_cache_dir(
            exp_dir=exp_dir,
            data_source="covla",  # TODO: make this configurable if needed
            layer_idx=args.layer,
        )
        logger.info(f" Test activation cache directory: {cache_dir}")
    
    # Collect activations (with caching)
    logger.info("Collecting SAE activations...")
    activations = collect_activations(
        orbis_model=orbis_model,
        sae=sae,
        dataloader=test_loader,
        layer_idx=args.layer,
        device=device,
        frame_rate=args.frame_rate,
        cache_dir=cache_dir,
        rebuild_cache=args.rebuild_cache,
    )
    
    # Compute correlations
    logger.info("Computing correlations...")
    correlations = compute_correlations(activations)
    
    # Find pure features
    logger.info("Finding pure features...")
    pure_features = find_pure_features(
        correlations,
        high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
    )
    logger.info(f"  Found {len(pure_features)} pure features")
    
    # Save correlation data
    corr_data = {
        field: [asdict(r) for r in results]
        for field, results in correlations.items()
    }
    with open(output_dir / "correlations.json", 'w') as f:
        json.dump(corr_data, f, indent=2)
    
    # Save pure features
    pure_data = [asdict(pf) for pf in pure_features]
    with open(output_dir / "pure_features.json", 'w') as f:
        json.dump(pure_data, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
    
    # Generate report
    logger.info("Generating report...")
    report = generate_report(
        correlations=correlations,
        pure_features=pure_features,
        activations=activations,
        videos_dir=Path(args.test_videos_dir),
        output_dir=output_dir,
        top_k=args.top_k,
        top_latents_per_field=args.top_latents_per_field,
    )
    
    logger.info(f"\n{report}")
    
    # Generate detailed frame analysis with full captions
    logger.info("Generating detailed frame analysis...")
    generate_detailed_frame_analysis(
        correlations=correlations,
        activations=activations,
        output_dir=output_dir,
        top_k=args.top_k,
        top_latents_per_field=args.top_latents_per_field,
    )
    
    # Analyze top activating latents overall (not tied to metadata)
    logger.info("Analyzing top activating latents...")
    analyze_top_latents(
        activations=activations,
        videos_dir=Path(args.test_videos_dir),
        output_dir=output_dir,
        top_n_latents=args.top_n_latents,
        top_k_frames=args.top_k,
        metric=args.latent_metric,
    )
    
    logger.info(f"done] Analysis complete. Results saved to {output_dir}")


def generate_report_from_saved(
    output_dir: Path,
    captions_dir: Path,
    top_k: int = 10,
    top_latents_per_field: int = 3,
) -> str:
    """
    Generate detailed report from previously saved analysis results.
    
    This reconstructs the detailed report from:
    - correlations.json
    - Saved frame images (which contain video_id and frame_idx in filename)
    - Caption files
    
    Can be used to regenerate reports without re-running inference.
    """
    output_dir = Path(output_dir)
    captions_dir = Path(captions_dir)
    
    # Load correlations
    with open(output_dir / "correlations.json", 'r') as f:
        corr_data = json.load(f)
    
    lines = [
        "# Detailed Frame Analysis",
        "",
        "---",
        "",
    ]
    
    for field_name in METADATA_FIELDS:
        lines.extend([
            f"## {field_name}",
            "",
        ])
        
        field_results = corr_data.get(field_name, [])
        
        if not field_results:
            lines.append("*No valid correlations computed.*\n")
            continue
        
        for i, result in enumerate(field_results[:top_latents_per_field]):
            latent_idx = result['latent_idx']
            r = result['pearson_r']
            p = result['p_value']
            
            lines.extend([
                f"### Latent {latent_idx} (r = {r:.4f}, p = {p:.2e})",
                "",
            ])
            
            # Check for saved frames
            frame_dir = output_dir / "frames" / f"latent_{latent_idx}_{field_name}"
            
            if not frame_dir.exists():
                lines.append("*No saved frames found.*\n")
                continue
            
            frames = sorted(frame_dir.glob("*.png"))
            
            for j, frame_path in enumerate(frames[:top_k]):
                # Parse filename: rank{N}_{video_id}_frame{idx}.png
                parts = frame_path.stem.split('_')
                rank = parts[0]  # rank1, rank2, etc.
                video_id = parts[1]
                frame_idx = int(parts[2].replace('frame', ''))
                
                # Load caption from JSONL
                caption = "*No caption available*"
                metadata_str = "N/A"
                
                caption_file = captions_dir / f"{video_id}.jsonl"
                if caption_file.exists():
                    with open(caption_file, 'r') as f:
                        for line in f:
                            entry = json.loads(line)
                            frame_key = str(frame_idx)
                            if frame_key in entry:
                                meta = entry[frame_key]
                                caption = meta.get('rich_caption', '*No caption*')
                                
                                # Extract metadata scores
                                meta_items = []
                                for k in METADATA_FIELDS:
                                    v = meta.get(k)
                                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                                        meta_items.append(f"{k}={v:.2f}")
                                metadata_str = ", ".join(meta_items) if meta_items else "N/A"
                                break
                
                # Image path (relative for markdown)
                img_rel_path = f"frames/latent_{latent_idx}_{field_name}/{frame_path.name}"
                
                lines.extend([
                    f"**{rank.capitalize()}:** `{video_id}` frame {frame_idx}",
                    f"- Metadata: {metadata_str}",
                    f"- Image: ![frame]({img_rel_path})",
                    "",
                    f"> {caption}",
                    "",
                ])
            
            lines.append("---\n")
    
    report = "\n".join(lines)
    
    # Save as markdown
    report_path = output_dir / "detailed_frame_analysis.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"report] Detailed frame analysis saved to {report_path}")
    
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Grounding Analysis for SAE")
    
    # Mode flag
    parser.add_argument("--from_saved", action="store_true",
                        help="Generate detailed report from saved analysis (no inference)")
    
    # Model paths (not required if --from_saved)
    parser.add_argument("--exp_dir", type=str, default=None,
                        help="Orbis experiment directory")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Config file name")
    parser.add_argument("--ckpt", type=str, default="checkpoints/last.ckpt",
                        help="Checkpoint file name")
    parser.add_argument("--sae_checkpoint", type=str, default=None,
                        help="Path to trained SAE checkpoint")
    
    # Data paths
    parser.add_argument("--test_videos_dir", type=str, default=None,
                        help="Directory containing test video files")
    parser.add_argument("--test_captions_dir", type=str, required=True,
                        help="Directory containing test caption files")
    parser.add_argument("--split_file", type=str, default=None,
                        help="Path to split file (jsonl with video_id per line)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save analysis results")
    
    # Model settings
    parser.add_argument("--layer", type=int, default=12,
                        help="Transformer layer to analyze")
    parser.add_argument("--input_size", type=int, nargs=2, default=[288, 512],
                        help="Input image size (H W)")
    parser.add_argument("--frame_rate", type=int, default=5,
                        help="Target frame rate")
    parser.add_argument("--stored_frame_rate", type=int, default=20,
                        help="Frame rate of stored videos")
    parser.add_argument("--num_frames", type=int, default=6,
                        help="Context window size: total frames per clip (must match SAE training)")
    
    # Analysis settings
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of top activating frames per latent")
    parser.add_argument("--top_latents_per_field", type=int, default=3,
                        help="Number of top correlated latents per field to report")
    parser.add_argument("--high_threshold", type=float, default=0.5,
                        help="Threshold for 'high' correlation for pure features")
    parser.add_argument("--low_threshold", type=float, default=0.2,
                        help="Threshold for 'low' correlation for pure features")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="Maximum number of test videos to process")
    
    # Top latents analysis (concepts not tied to metadata)
    parser.add_argument("--top_n_latents", type=int, default=20,
                        help="Number of top activating latents to analyze")
    parser.add_argument("--latent_metric", type=str, default="max",
                        choices=["max", "mean", "std"],
                        help="Metric to rank latents: max/mean activation or std (variance)")
    
    # Runtime
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    
    # Caching (Orbis activations cached in logs_sae/sae_cache/.../test/)
    parser.add_argument("--no_cache", action="store_true",
                        help="Disable Orbis activation caching (always recompute)")
    parser.add_argument("--rebuild_cache", action="store_true",
                        help="Force rebuild of Orbis activation cache even if it exists")
    
    args = parser.parse_args()
    
    if args.from_saved:
        # Just regenerate the detailed report from saved data
        logger.info("Generating detailed report from saved analysis...")
        generate_report_from_saved(
            output_dir=Path(args.output_dir),
            captions_dir=Path(args.test_captions_dir),
            top_k=args.top_k,
            top_latents_per_field=args.top_latents_per_field,
        )
    else:
        # Validate required arguments for full analysis
        if not args.exp_dir:
            parser.error("--exp_dir is required for full analysis")
        if not args.sae_checkpoint:
            parser.error("--sae_checkpoint is required for full analysis")
        if not args.test_videos_dir:
            parser.error("--test_videos_dir is required for full analysis")
        main(args)
