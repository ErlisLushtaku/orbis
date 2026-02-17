"""
Post-Training Semantic Grounding Analysis for Orbis SAE.

Analyzes trained SAE latents by computing correlations with CoVLA metadata
fields (tunnel, highway, pedestrian, risk confidence scores) on a held-out
test set.

Produces multi-granularity visualizations for manual concept discovery:
- Clips (mp4): temporal driving behavior
- Frames (PNG): target frame snapshots
- Patches (heatmap): spatial activation maps showing where in the image a latent fires

Usage:
    python sae/scripts/semantic_grounding_covla.py \
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
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from PIL import Image

# Add orbis root to path for imports (scripts are in sae/scripts/, go up 2 levels)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sae.topk_sae import TopKSAE
from sae.activation_hooks import ActivationExtractor
from sae.scripts.grounding_analysis import (
    CorrelationResult,
    PureFeature,
    compute_correlations as _compute_correlations,
    find_pure_features,
    find_top_activating_frames as _find_top_activating_frames,
    find_top_activating_latents,
)
from sae.utils.constants import ORBIS_GRID_H, ORBIS_GRID_W
from sae.utils.logging import get_logger, setup_sae_logging
from sae.utils.model_loading import load_orbis_model, load_sae
from sae.utils.spatial_tracker import SpatialEntry, SpatialTopKTracker
from sae.utils.viz import create_clip_video, create_overlay_heatmap, save_frame_png
from data.covla.covla_dataset import CoVLAOrbisMultiFrame

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


def create_val_dataset_covla(
    videos_dir: str,
    captions_dir: str,
    input_size: Tuple[int, int] = (288, 512),
    num_frames: int = 6,
    stored_frame_rate: int = 20,
    frame_rate: int = 5,
    num_videos: Optional[int] = None,
    val_split: float = 0.1,
) -> Tuple[Subset, "CoVLAOrbisMultiFrame"]:
    """Create the val dataset using the same split logic as SAE training.

    Returns:
        Tuple of (val_subset, full_dataset) so the wrapper can access dataset internals.
    """
    full_dataset = CoVLAOrbisMultiFrame(
        num_frames=num_frames,
        stored_data_frame_rate=stored_frame_rate,
        target_frame_rate=frame_rate,
        size=input_size,
        captions_dir=captions_dir,
        videos_dir=videos_dir,
        num_samples=num_videos,
        debug=False,
    )

    total_videos = full_dataset.num_videos
    clips_per_video = full_dataset.clips_per_video

    num_val_videos = max(1, int(total_videos * val_split))
    num_train_videos = total_videos - num_val_videos

    train_end_idx = num_train_videos * clips_per_video
    val_indices = list(range(train_end_idx, len(full_dataset)))

    val_subset = Subset(full_dataset, val_indices)

    logger.info(
        f"CoVLA val split: {total_videos} total videos, "
        f"{num_train_videos} train / {num_val_videos} val, "
        f"{len(val_subset)} val clips"
    )

    return val_subset, full_dataset


class CoVLAValWrapper(Dataset):  # type: ignore[type-arg]
    """Wraps a CoVLAOrbisMultiFrame val Subset to produce the format expected by custom_collate.

    CoVLAOrbisMultiFrame returns {images, caption, video_id, frame_rate}.
    The grounding pipeline expects {image, video_id, frame_idx, clip_frame_indices, metadata}.
    This wrapper bridges the gap by computing clip frame indices and loading per-frame metadata.
    """

    def __init__(
        self,
        val_subset: Subset,
        full_dataset: "CoVLAOrbisMultiFrame",
        captions_dir: str,
    ):
        self.val_subset = val_subset
        self.full_dataset = full_dataset
        self.captions_dir = Path(captions_dir)

    def __len__(self) -> int:
        return len(self.val_subset)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.val_subset[idx]

        # Recover the full-dataset flat index to compute clip frame indices
        full_idx = self.val_subset.indices[idx]
        video_idx = full_idx // self.full_dataset.clips_per_video
        clip_idx = full_idx % self.full_dataset.clips_per_video
        video_id = self.full_dataset.video_ids[video_idx]

        # Get clip frame indices (requires total_frames from video)
        from decord import VideoReader, cpu
        video_path = os.path.join(self.full_dataset.videos_dir, f"{video_id}.mp4")
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        del vr

        frame_indices = self.full_dataset._get_clip_frame_indices(clip_idx, total_frames)
        target_frame_idx = frame_indices[-1]

        metadata = load_frame_metadata(self.captions_dir, video_id, target_frame_idx)

        return {
            "image": sample["images"],  # (T, 3, H, W)
            "video_id": video_id,
            "frame_idx": target_frame_idx,
            "clip_frame_indices": frame_indices,
            "metadata": metadata,
        }



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
            "clip_frame_indices": frame_indices,
            "metadata": metadata,
        }


def custom_collate(batch: List[Dict]) -> Dict:
    """Custom collate that handles metadata dicts and multi-frame clips."""
    images = torch.stack([item["image"] for item in batch])
    video_ids = [item["video_id"] for item in batch]
    frame_idxs = [item["frame_idx"] for item in batch]
    clip_frame_indices = [item["clip_frame_indices"] for item in batch]
    metadata_list = [item["metadata"] for item in batch]
    
    return {
        "images": images,  # (B, T, 3, H, W) or (B, 3, H, W)
        "video_ids": video_ids,
        "frame_idxs": frame_idxs,
        "clip_frame_indices": clip_frame_indices,
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
        clip_frame_indices = batch["clip_frame_indices"]
        metadata_list = batch["metadata"]
        
        b = imgs.shape[0]
        
        # Handle both single-frame (B, 3, H, W) and multi-frame (B, T, 3, H, W) inputs
        if imgs.dim() == 4:
            imgs = imgs.unsqueeze(1)
        
        t_frames = imgs.shape[1]
        
        # Encode frames and apply sigma_min noise (matches cache creation)
        x = orbis_model.encode_frames(imgs)
        t_noise = torch.zeros(b, device=device)
        x_noised, _ = orbis_model.add_noise(x, t_noise)
        fr = torch.full((b,), frame_rate, device=device)
        
        # Split into context and target
        if t_frames > 1:
            context = x_noised[:, :-1]
            target = x_noised[:, -1:]
        else:
            warnings.warn("Single-frame input detected - activations may be out-of-distribution")
            context = x_noised
            target = x_noised
        
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
                "clip_frame_indices": clip_frame_indices[i],
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
    spatial_cache_path: Optional[Path] = None,
    top_k_frames: int = 10,
) -> Tuple[List[FrameActivation], Optional[SpatialTopKTracker]]:
    """
    Process test video clips and collect SAE latent activations per target frame.
    
    Single-pass collection: computes spatially-averaged activations for
    correlation analysis AND maintains per-latent top-K spatial activation
    maps for visualization.
    
    Returns:
        Tuple of (activations list, spatial tracker or None).
    """
    tracker = None

    # Check for existing spatial cache
    if spatial_cache_path and spatial_cache_path.exists() and not rebuild_cache:
        tracker = SpatialTopKTracker.load(spatial_cache_path)

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
        
        logger.info(" Encoding cached activations through SAE...")
        all_activations: List[FrameActivation] = []
        sample_idx = 0
        d_sae = sae.config.d_sae
        need_tracker = tracker is None and spatial_cache_path is not None
        if need_tracker:
            tracker = SpatialTopKTracker(num_latents=d_sae, top_k=top_k_frames)
        
        for cache_file in tqdm(cache_files, desc="Processing cached activations"):
            data = torch.load(cache_file, weights_only=True)
            acts = data["activations"].to(device).float()
            
            batch_start = sample_idx
            while sample_idx < len(sample_metadata):
                meta = sample_metadata[sample_idx]
                spatial_tokens = meta["spatial_tokens"]
                
                acts_needed = (sample_idx - batch_start + 1) * spatial_tokens
                if acts_needed > acts.shape[0]:
                    break
                
                start_idx = (sample_idx - batch_start) * spatial_tokens
                end_idx = start_idx + spatial_tokens
                sample_acts = acts[start_idx:end_idx]  # (spatial, hidden_dim)
                
                sae_acts = sae.get_feature_activations(sample_acts)  # (spatial, num_latents)
                sae_acts_mean = sae_acts.mean(dim=0).cpu().numpy()
                
                # Update spatial tracker (target frame only = last 576 tokens)
                if need_tracker and tracker is not None:
                    num_spatial = ORBIS_GRID_H * ORBIS_GRID_W  # 576
                    target_spatial = sae_acts[-num_spatial:, :].cpu().numpy()  # (576, num_latents)
                    tracker_meta = [{
                        "video_id": meta["video_id"],
                        "frame_idx": meta["frame_idx"],
                        "clip_frame_indices": meta.get("clip_frame_indices", []),
                        "metadata": meta["metadata"],
                    }]
                    tracker.update(
                        sae_acts_mean[np.newaxis, :],
                        target_spatial[np.newaxis, :, :],
                        tracker_meta,
                    )
                
                frame_act = FrameActivation(
                    video_id=meta["video_id"],
                    frame_idx=meta["frame_idx"],
                    latent_activations=sae_acts_mean,
                    metadata=meta["metadata"],
                    rich_caption=meta["rich_caption"],
                )
                all_activations.append(frame_act)
                sample_idx += 1
        
        if spatial_cache_path and need_tracker and tracker is not None:
            tracker.save(spatial_cache_path)
        
        return all_activations, tracker
    
    # No caching - compute directly (fallback)
    extractor = ActivationExtractor(orbis_model, layer_idx=layer_idx, flatten_spatial=True)
    all_activations: List[FrameActivation] = []
    d_sae = sae.config.d_sae
    need_tracker = tracker is None and spatial_cache_path is not None
    if need_tracker:
        tracker = SpatialTopKTracker(num_latents=d_sae, top_k=top_k_frames)
    
    for batch in tqdm(dataloader, desc="Collecting activations"):
        imgs = batch["images"].to(device)
        video_ids = batch["video_ids"]
        frame_idxs = batch["frame_idxs"]
        clip_frame_indices = batch["clip_frame_indices"]
        metadata_list = batch["metadata"]
        
        b = imgs.shape[0]
        
        if imgs.dim() == 4:
            imgs = imgs.unsqueeze(1)
        
        t_frames = imgs.shape[1]
        x = orbis_model.encode_frames(imgs)
        t_noise = torch.zeros(b, device=device)
        x_noised, _ = orbis_model.add_noise(x, t_noise)
        fr = torch.full((b,), frame_rate, device=device)
        
        if t_frames > 1:
            context = x_noised[:, :-1]
            target = x_noised[:, -1:]
        else:
            warnings.warn("Single-frame input detected")
            context = x_noised
            target = x_noised
        
        with extractor.capture():
            _ = orbis_model.vit(target, context, t_noise, frame_rate=fr)
        
        acts = extractor.get_activations()
        spatial_tokens = acts.shape[0] // b
        acts = acts.view(b, spatial_tokens, -1)
        
        sae_acts = sae.get_feature_activations(acts.float())  # (B, N, d_sae)
        sae_acts_mean = sae_acts.mean(dim=1).cpu().numpy()
        
        # Update spatial tracker (target frame only = last 576 tokens)
        if need_tracker and tracker is not None:
            num_spatial = ORBIS_GRID_H * ORBIS_GRID_W  # 576
            target_spatial = sae_acts[:, -num_spatial:, :].cpu().numpy()  # (B, 576, d_sae)
            batch_metadata = [
                {
                    "video_id": video_ids[i],
                    "frame_idx": frame_idxs[i],
                    "clip_frame_indices": clip_frame_indices[i],
                    "metadata": {
                        f_name: metadata_list[i].get(f_name, np.nan)
                        for f_name in METADATA_FIELDS + CONTEXT_FIELDS
                    },
                }
                for i in range(b)
            ]
            tracker.update(sae_acts_mean, target_spatial, batch_metadata)
        
        for i in range(b):
            metadata = metadata_list[i]
            meta_scores = {
                f_name: metadata.get(f_name, np.nan)
                for f_name in METADATA_FIELDS + CONTEXT_FIELDS
            }
            
            frame_act = FrameActivation(
                video_id=video_ids[i],
                frame_idx=frame_idxs[i],
                latent_activations=sae_acts_mean[i],
                metadata=meta_scores,
                rich_caption=metadata.get("rich_caption", ""),
            )
            all_activations.append(frame_act)
    
    if spatial_cache_path and need_tracker and tracker is not None:
        tracker.save(spatial_cache_path)
    
    return all_activations, tracker


def _collect_metadata_covla(
    dataloader: DataLoader,
) -> List[Dict[str, Any]]:
    """Collect per-sample metadata from the test dataset (CPU only)."""
    all_metadata: List[Dict[str, Any]] = []
    for batch in tqdm(dataloader, desc="Collecting metadata"):
        for i in range(len(batch["video_ids"])):
            meta = batch["metadata"][i]
            meta_scores = {
                f_name: meta.get(f_name, np.nan)
                for f_name in METADATA_FIELDS + CONTEXT_FIELDS
            }
            all_metadata.append({
                "video_id": batch["video_ids"][i],
                "frame_idx": batch["frame_idxs"][i],
                "clip_frame_indices": batch["clip_frame_indices"][i],
                "metadata": meta_scores,
                "rich_caption": meta.get("rich_caption", ""),
            })
    return all_metadata


@torch.inference_mode()
def collect_activations_from_val_cache(
    sae: TopKSAE,
    val_cache_dir: Path,
    metadata: List[Dict[str, Any]],
    device: torch.device,
    spatial_cache_path: Optional[Path] = None,
    top_k_frames: int = 10,
) -> Tuple[List[FrameActivation], Optional[SpatialTopKTracker]]:
    """Load pre-cached Orbis activations from WebDataset val shards and encode through SAE.

    Skips the Orbis model entirely, using raw activations from the
    sae_cache val directory created during training.
    """
    import io
    import webdataset as wds

    tracker = None

    # Check existing spatial cache
    if spatial_cache_path and spatial_cache_path.exists():
        tracker = SpatialTopKTracker.load(spatial_cache_path)
        # Still need to build FrameActivation list, but skip tracker rebuild

    # Load _meta.json
    meta_file = val_cache_dir / "_meta.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"Val cache metadata not found: {meta_file}")

    import json as json_module
    with meta_file.open("r") as f:
        cache_meta = json_module.load(f)

    num_cache_files = cache_meta["num_files"]
    total_tokens = cache_meta["total_tokens"]
    hidden_dim = cache_meta["hidden_dim"]
    cache_batch_size = cache_meta.get("batch_size", 4)

    # Verify sample count
    expected_samples = num_cache_files * cache_batch_size
    mismatch = abs(expected_samples - len(metadata))
    if mismatch > cache_batch_size:
        raise ValueError(
            f"Sample count mismatch too large: cache has {num_cache_files} files x "
            f"{cache_batch_size} samples = {expected_samples}, "
            f"but metadata has {len(metadata)} samples (diff={mismatch}). "
            f"The val cache was likely created from a different dataset split."
        )
    elif mismatch > 0:
        logger.info(
            f"Sample count near-match: cache={expected_samples}, metadata={len(metadata)} "
            f"(diff={mismatch}, last batch may be partial)"
        )

    logger.info(
        f"Loading val cache from {val_cache_dir}: "
        f"{num_cache_files} files, {total_tokens:,} tokens, "
        f"hidden_dim={hidden_dim}, batch_size={cache_batch_size}"
    )

    # Custom pt decoder
    def _pt_decoder(data: bytes) -> torch.Tensor:
        stream = io.BytesIO(data)
        loaded = torch.load(stream, weights_only=True, map_location="cpu")
        if isinstance(loaded, dict) and "activations" in loaded:
            return loaded["activations"]
        return loaded

    # Load shards sequentially
    shard_pattern = sorted(val_cache_dir.glob("*.tar"))
    if not shard_pattern:
        raise FileNotFoundError(f"No .tar shards found in {val_cache_dir}")

    dataset = wds.WebDataset(
        [str(p) for p in shard_pattern],
        shardshuffle=0,
        empty_check=False,
    ).decode(wds.handle_extension("pt", _pt_decoder)).to_tuple("pt")

    d_sae = sae.config.d_sae
    num_spatial = ORBIS_GRID_H * ORBIS_GRID_W
    need_tracker = tracker is None
    if need_tracker:
        tracker = SpatialTopKTracker(num_latents=d_sae, top_k=top_k_frames)

    all_activations: List[FrameActivation] = []
    sample_idx = 0

    for (acts_flat,) in tqdm(dataset, desc="Encoding cached activations", total=num_cache_files):
        acts_flat = acts_flat.to(device).float()

        tokens_per_sample = acts_flat.shape[0] // cache_batch_size
        actual_batch_size = acts_flat.shape[0] // tokens_per_sample
        acts_batched = acts_flat.view(actual_batch_size, tokens_per_sample, hidden_dim)

        sae_acts = sae.get_feature_activations(acts_batched)
        sae_acts_mean = sae_acts.mean(dim=1).cpu().numpy()

        # Spatial tracker: target frame only
        if need_tracker:
            target_spatial = sae_acts[:, -num_spatial:, :].cpu().numpy()

        batch_metadata = []
        for i in range(actual_batch_size):
            if sample_idx + i >= len(metadata):
                break
            meta = metadata[sample_idx + i]
            batch_metadata.append({
                "video_id": meta["video_id"],
                "frame_idx": meta["frame_idx"],
                "clip_frame_indices": meta["clip_frame_indices"],
                "metadata": meta["metadata"],
            })

        if need_tracker and batch_metadata:
            tracker.update(
                sae_acts_mean[:len(batch_metadata)],
                target_spatial[:len(batch_metadata)],
                batch_metadata,
            )

        for i in range(actual_batch_size):
            if sample_idx >= len(metadata):
                break
            meta = metadata[sample_idx]
            all_activations.append(FrameActivation(
                video_id=meta["video_id"],
                frame_idx=meta["frame_idx"],
                latent_activations=sae_acts_mean[i],
                metadata=meta["metadata"],
                rich_caption=meta.get("rich_caption", ""),
            ))
            sample_idx += 1

    logger.info(f"Encoded {len(all_activations)} samples from val cache")

    if spatial_cache_path and need_tracker and tracker is not None:
        tracker.save(spatial_cache_path)

    return all_activations, tracker


def _build_matrices(
    activations: List[FrameActivation],
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """Build matrix representations from CoVLA FrameActivation list."""
    latent_matrix = np.stack([a.latent_activations for a in activations], axis=0)
    field_matrix = np.array([
        [a.metadata.get(f, np.nan) for f in METADATA_FIELDS]
        for a in activations
    ])
    sample_metadata = [
        {
            "video_id": a.video_id,
            "frame_idx": a.frame_idx,
            "rich_caption": a.rich_caption,
            "metadata": a.metadata,
        }
        for a in activations
    ]
    return latent_matrix, field_matrix, sample_metadata


def compute_correlations(
    activations: List[FrameActivation],
    min_samples: int = 100,
) -> Dict[str, List[CorrelationResult]]:
    """Thin wrapper around shared compute_correlations for CoVLA FrameActivation list."""
    latent_matrix, field_matrix, _ = _build_matrices(activations)
    return _compute_correlations(latent_matrix, field_matrix, METADATA_FIELDS, min_samples)


def find_top_activating_frames(
    activations: List[FrameActivation],
    latent_idx: int,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Thin wrapper for CoVLA -- adds rich_caption and activation_value keys."""
    latent_matrix, _, sample_metadata = _build_matrices(activations)
    results = _find_top_activating_frames(latent_matrix, latent_idx, sample_metadata, top_k)
    # Add CoVLA-specific key aliases for backward compatibility with report generators
    for r in results:
        r["activation_value"] = r["activation"]
    return results


def _load_frame_from_video(videos_dir: Path, video_id: str, frame_idx: int) -> Optional[np.ndarray]:
    """Load a single frame from a CoVLA mp4 video."""
    from decord import VideoReader, cpu
    video_path = videos_dir / f"{video_id}.mp4"
    if not video_path.exists():
        return None
    try:
        vr = VideoReader(str(video_path), ctx=cpu(0))
        return vr[frame_idx].asnumpy()
    except Exception as e:
        logger.warning(f"Could not load frame {frame_idx} from {video_path}: {e}")
        return None


def _load_clip_from_video(
    videos_dir: Path, video_id: str, frame_indices: List[int],
) -> List[np.ndarray]:
    """Load multiple frames for a clip from a CoVLA mp4 video."""
    from decord import VideoReader, cpu
    video_path = videos_dir / f"{video_id}.mp4"
    if not video_path.exists():
        return []
    try:
        vr = VideoReader(str(video_path), ctx=cpu(0))
        total = len(vr)
        valid_indices = [min(idx, total - 1) for idx in frame_indices]
        return [vr[idx].asnumpy() for idx in valid_indices]
    except Exception as e:
        logger.warning(f"Could not load clip from {video_path}: {e}")
        return []


def _save_covla_latent_visualizations(
    entries: List[SpatialEntry],
    videos_dir: Path,
    output_dir: Path,
    latent_idx: int,
    captions_dir: Optional[Path] = None,
) -> List[str]:
    """Save clip/frame/heatmap artifacts for a CoVLA latent's top entries."""
    saved_paths = []
    for rank, entry in enumerate(entries):
        prefix = f"rank{rank + 1}_{entry.video_id}_frame{entry.frame_idx}"

        frame = _load_frame_from_video(videos_dir, entry.video_id, entry.frame_idx)
        if frame is None:
            continue

        # Target frame PNG
        frame_path = output_dir / f"{prefix}.png"
        save_frame_png(frame, frame_path)
        saved_paths.append(str(frame_path))

        # Overlay heatmap
        heatmap_path = output_dir / f"{prefix}_heatmap.png"
        title = (
            f"Latent {latent_idx} | rank {rank + 1} | "
            f"act={entry.score:.3f}"
        )
        create_overlay_heatmap(frame, entry.spatial_map, heatmap_path, title=title)

        # Clip video
        if entry.clip_frame_indices:
            clip_frames = _load_clip_from_video(
                videos_dir, entry.video_id, entry.clip_frame_indices,
            )
            if clip_frames:
                clip_path = output_dir / f"{prefix}_clip.mp4"
                create_clip_video(clip_frames, clip_path)

    return saved_paths


def analyze_top_latents(
    activations: List[FrameActivation],
    videos_dir: Path,
    output_dir: Path,
    tracker: Optional[SpatialTopKTracker] = None,
    top_n_latents: int = 20,
    top_k_frames: int = 10,
    metric: str = "max",
    captions_dir: Optional[Path] = None,
) -> str:
    """Analyze top activating latents with multi-granularity visualizations.

    When a spatial tracker is available, saves heatmaps and clip videos in
    addition to frame PNGs.
    """
    logger.info(f"Finding top {top_n_latents} latents by {metric} activation...")
    latent_matrix = np.stack([a.latent_activations for a in activations], axis=0)
    top_latents = find_top_activating_latents(latent_matrix, top_n=top_n_latents, metric=metric)

    top_latents_dir = output_dir / "top_latents"
    top_latents_dir.mkdir(parents=True, exist_ok=True)

    has_tracker = tracker is not None

    lines = [
        "# Top Activating Latents Analysis",
        "",
        f"**Metric:** {metric}",
        f"**Total latents analyzed:** {latent_matrix.shape[1]}",
        f"**Total frames:** {latent_matrix.shape[0]}",
        f"**Visualizations:** {'heatmaps + clips' if has_tracker else 'frames only'}",
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

        latent_dir = top_latents_dir / f"latent_{latent_idx}"
        latent_dir.mkdir(parents=True, exist_ok=True)

        if has_tracker:
            entries = tracker.get_top_k(latent_idx)[:top_k_frames]
            _save_covla_latent_visualizations(
                entries, videos_dir, latent_dir, latent_idx, captions_dir,
            )

            lines.append("### Top Activating Clips")
            lines.append("")
            for rank, entry in enumerate(entries):
                vid = entry.video_id
                fidx = entry.frame_idx
                prefix = f"rank{rank + 1}_{vid}_frame{fidx}"
                img_rel = f"top_latents/latent_{latent_idx}/{prefix}.png"
                heatmap_rel = f"top_latents/latent_{latent_idx}/{prefix}_heatmap.png"

                # Try to load caption
                caption = ""
                if captions_dir:
                    meta_data = load_frame_metadata(captions_dir, vid, fidx)
                    caption = meta_data.get("rich_caption", "")

                lines.extend([
                    f"**Rank {rank + 1}:** `{vid}` frame {fidx} (activation: {entry.score:.4f})",
                    "",
                    f"![frame]({img_rel})",
                    "",
                    f"![heatmap]({heatmap_rel})",
                    "",
                ])
                if caption:
                    lines.extend([f"> {caption}", ""])
        else:
            # Fallback: frame-only visualization
            top_frames = find_top_activating_frames(activations, latent_idx, top_k_frames)
            for j, frame_info in enumerate(top_frames):
                frame = _load_frame_from_video(videos_dir, frame_info["video_id"], frame_info["frame_idx"])
                if frame is not None:
                    save_path = latent_dir / f"rank{j+1}_{frame_info['video_id']}_frame{frame_info['frame_idx']}.png"
                    save_frame_png(frame, save_path)

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
                    "",
                    f"![frame]({img_path})",
                    "",
                    f"> {caption}",
                    "",
                ])

        lines.extend(["---", ""])

    with open(output_dir / "top_latents.json", "w") as f:
        json.dump(top_latents, f, indent=2)

    report = "\n".join(lines)
    report_path = output_dir / "top_latents_analysis.md"
    with open(report_path, "w") as f:
        f.write(report)

    logger.info(f"Top latents analysis saved to {report_path}")
    return report


def save_top_frames(
    top_frames: List[Dict[str, Any]],
    videos_dir: Path,
    output_dir: Path,
    latent_idx: int,
    field_name: str,
    tracker: Optional[SpatialTopKTracker] = None,
    top_k_frames: int = 10,
) -> List[str]:
    """Save images (and optionally heatmaps/clips) of top-activating frames."""
    frame_dir = output_dir / "frames" / f"latent_{latent_idx}_{field_name}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    # If tracker available, use it for full visualization
    if tracker is not None:
        entries = tracker.get_top_k(latent_idx)[:top_k_frames]
        if entries:
            return _save_covla_latent_visualizations(
                entries, videos_dir, frame_dir, latent_idx,
            )

    # Fallback: frame-only
    saved_paths = []
    for i, frame_info in enumerate(top_frames):
        frame = _load_frame_from_video(videos_dir, frame_info["video_id"], frame_info["frame_idx"])
        if frame is not None:
            save_path = frame_dir / f"rank{i+1}_{frame_info['video_id']}_frame{frame_info['frame_idx']}.png"
            save_frame_png(frame, save_path)
            saved_paths.append(str(save_path))

    return saved_paths


def generate_report(
    correlations: Dict[str, List[CorrelationResult]],
    pure_features: List[PureFeature],
    activations: List[FrameActivation],
    videos_dir: Path,
    output_dir: Path,
    top_k: int = 10,
    top_latents_per_field: int = 3,
    tracker: Optional[SpatialTopKTracker] = None,
) -> str:
    """Generate comprehensive analysis report with optional multi-granularity visualizations."""
    
    report_lines = [
        "=" * 80,
        "SEMANTIC GROUNDING ANALYSIS REPORT",
        "=" * 80,
        "",
        f"Total frames analyzed: {len(activations)}",
        f"Number of latents: {activations[0].latent_activations.shape[0]}",
        f"Visualizations: {'heatmaps + clips' if tracker else 'frames only'}",
        "",
    ]
    
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
            
            top_frames = find_top_activating_frames(activations, result.latent_idx, top_k)
            
            saved_paths = save_top_frames(
                top_frames, videos_dir, output_dir,
                result.latent_idx, field_name,
                tracker=tracker, top_k_frames=top_k,
            )
            
            report_lines.append(f"      Top {top_k} activating frames saved to: {saved_paths[0] if saved_paths else 'N/A'}")
            
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
    
    by_field: Dict[str, List[PureFeature]] = {f: [] for f in METADATA_FIELDS}
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
    
    report_path = output_dir / "semantic_grounding_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    
    logger.info(f"Saved report to {report_path}")
    
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
    
    logger.info(f"Detailed frame analysis saved to {report_path}")
    
    return report


def main(args: argparse.Namespace):
    """Main analysis function."""
    
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    exp_dir = Path(args.exp_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load SAE (always needed)
    logger.info("Loading SAE...")
    sae = load_sae(Path(args.sae_checkpoint), device)
    logger.info(f"SAE config: d_in={sae.config.d_in}, d_sae={sae.config.d_sae}, k={sae.config.k}")
    
    # Create dataset
    if args.use_val_split:
        # Val split mode: same split as SAE training (compatible with val cache)
        logger.info("Creating val dataset (same split as SAE training)...")
        val_subset, full_dataset = create_val_dataset_covla(
            videos_dir=args.test_videos_dir,
            captions_dir=args.test_captions_dir,
            input_size=tuple(args.input_size),
            num_frames=args.num_frames,
            stored_frame_rate=args.stored_frame_rate,
            frame_rate=args.frame_rate,
            num_videos=args.num_videos,
            val_split=args.val_split,
        )
        analysis_dataset: Dataset = CoVLAValWrapper(val_subset, full_dataset, args.test_captions_dir)
    else:
        # Test split mode: use separate test set via split file
        logger.info("Creating test dataset...")
        analysis_dataset = TestDataset(
            videos_dir=args.test_videos_dir,
            captions_dir=args.test_captions_dir,
            split_file=args.split_file,
            size=tuple(args.input_size),
            target_frame_rate=args.frame_rate,
            stored_frame_rate=args.stored_frame_rate,
            num_frames=args.num_frames,
            max_videos=args.max_videos,
        )
    
    test_loader = DataLoader(
        analysis_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate,
        pin_memory=True,
    )
    
    # Spatial cache for heatmap/clip visualization
    spatial_cache_path = output_dir / "spatial_topk_cache.pt" if not args.no_cache else None
    if args.rebuild_cache and spatial_cache_path and spatial_cache_path.exists():
        spatial_cache_path.unlink()
    
    # Collect activations + spatial top-K
    if args.val_cache_dir:
        # Fast path: load pre-cached Orbis activations from WDS shards
        val_cache_dir = Path(args.val_cache_dir)
        logger.info(f"Using pre-cached val activations from {val_cache_dir}")
        metadata = _collect_metadata_covla(test_loader)
        activations, tracker = collect_activations_from_val_cache(
            sae=sae,
            val_cache_dir=val_cache_dir,
            metadata=metadata,
            device=device,
            spatial_cache_path=spatial_cache_path,
            top_k_frames=args.top_k,
        )
    else:
        # Standard path: run Orbis forward passes
        logger.info("Loading Orbis model...")
        config_path = exp_dir / args.config
        ckpt_path = exp_dir / args.ckpt
        orbis_model = load_orbis_model(config_path, ckpt_path, device)
        
        cache_dir = None
        if not args.no_cache:
            cache_dir = get_test_cache_dir(
                exp_dir=exp_dir,
                data_source="covla",
                layer_idx=args.layer,
            )
            logger.info(f"Test activation cache directory: {cache_dir}")
        
        logger.info("Collecting SAE activations...")
        activations, tracker = collect_activations(
            orbis_model=orbis_model,
            sae=sae,
            dataloader=test_loader,
            layer_idx=args.layer,
            device=device,
            frame_rate=args.frame_rate,
            cache_dir=cache_dir,
            rebuild_cache=args.rebuild_cache,
            spatial_cache_path=spatial_cache_path,
            top_k_frames=args.top_k,
        )
    
    # Compute correlations
    logger.info("Computing correlations...")
    correlations = compute_correlations(activations)
    
    # Find pure features
    logger.info("Finding pure features...")
    pure_features = find_pure_features(
        correlations,
        primary_threshold=args.high_threshold,
        secondary_threshold=args.low_threshold,
    )
    logger.info(f"Found {len(pure_features)} pure features")
    
    # Save correlation data
    corr_data = {
        f: [asdict(r) for r in results]
        for f, results in correlations.items()
    }
    with open(output_dir / "correlations.json", "w") as f:
        json.dump(corr_data, f, indent=2)
    
    # Save pure features
    pure_data = [asdict(pf) for pf in pure_features]
    with open(output_dir / "pure_features.json", "w") as f:
        json.dump(pure_data, f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else x)
    
    # Generate report (with optional heatmaps/clips)
    logger.info("Generating report...")
    report = generate_report(
        correlations=correlations,
        pure_features=pure_features,
        activations=activations,
        videos_dir=Path(args.test_videos_dir),
        output_dir=output_dir,
        top_k=args.top_k,
        top_latents_per_field=args.top_latents_per_field,
        tracker=tracker,
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
    
    # Analyze top activating latents (with optional heatmaps/clips)
    logger.info("Analyzing top activating latents...")
    analyze_top_latents(
        activations=activations,
        videos_dir=Path(args.test_videos_dir),
        output_dir=output_dir,
        tracker=tracker,
        top_n_latents=args.top_n_latents,
        top_k_frames=args.top_k,
        metric=args.latent_metric,
        captions_dir=Path(args.test_captions_dir),
    )
    
    logger.info(f"Analysis complete. Results saved to {output_dir}")


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
    
    logger.info(f"Detailed frame analysis saved to {report_path}")
    
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
    parser.add_argument("--val_cache_dir", type=str, default=None,
                        help="Path to pre-cached val activations (sae_cache WDS shards). "
                        "Skips Orbis model loading when provided.")
    
    # Val split mode: use the same val set as SAE training (compatible with val cache)
    parser.add_argument("--use_val_split", action="store_true",
                        help="Use the SAE training val split (last 10%% of videos by sorted order) "
                        "instead of a test split file. Compatible with --val_cache_dir.")
    parser.add_argument("--num_videos", type=int, default=3000,
                        help="Total number of videos (must match training config for correct split)")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Val split fraction (must match training config)")
    
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
