"""
Post-Training Semantic Grounding Analysis for Orbis SAE using NuPlan data.

Analyzes trained SAE latents by computing correlations with NuPlan odometry data
(speed, acceleration, yaw rate) on the validation set. Uses the same 90/10
video-boundary split as training to ensure exact val set correspondence.

Produces multi-granularity visualizations for manual concept discovery:
- Clips (mp4): temporal driving behavior
- Frames (PNG): target frame snapshots
- Patches (heatmap): spatial activation maps showing where in the image a latent fires

Usage:
    python sae/scripts/semantic_grounding_nuplan.py \
        --exp_dir /path/to/orbis_288x512 \
        --sae_checkpoint /path/to/sae/best.pt \
        --nuplan_data_dir /path/to/nuplan_videos \
        --output_dir /path/to/analysis_output \
        --num_videos 988 --val_split 0.1
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

# Add orbis root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sae.topk_sae import TopKSAE
from sae.activation_hooks import ActivationExtractor
from sae.scripts.grounding_analysis import (
    CorrelationResult,
    PureFeature,
    compute_correlations,
    find_pure_features,
    find_top_activating_frames,
    find_top_activating_latents,
)
from sae.utils.constants import ORBIS_GRID_H, ORBIS_GRID_W
from sae.utils.logging import get_logger, setup_sae_logging
from sae.utils.model_loading import load_orbis_model, load_sae
from sae.utils.spatial_tracker import SpatialEntry, SpatialTopKTracker
from sae.utils.viz import create_clip_video, create_patch_heatmap, save_frame_png
from data.nuplan.nuplan_dataset import NuPlanOrbisMultiFrame

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


def _resolve_h5_chunk(h5_file: h5py.File, frame_idx: int) -> np.ndarray:
    """Resolve a global frame index to its chunk and local index in an H5 file.

    NuPlan H5 files store frames in multiple chunked datasets (sorted keys).
    This replicates the logic from NuPlanOrbisMultiFrame._load_frames_from_h5().
    """
    keys = sorted(h5_file.keys())
    chunk_starts = []
    cum = 0
    for k in keys:
        chunk_starts.append(cum)
        cum += h5_file[k].shape[0]
    total = cum
    frame_idx = min(frame_idx, total - 1)

    chunk_idx = 0
    for i, start in enumerate(chunk_starts):
        if i + 1 < len(chunk_starts) and frame_idx >= chunk_starts[i + 1]:
            chunk_idx = i + 1
        elif frame_idx >= start:
            chunk_idx = i
    local = frame_idx - chunk_starts[chunk_idx]
    return h5_file[keys[chunk_idx]][local]


def load_frame_from_h5(
    data_dir: str, video_id: str, frame_idx: int,
) -> Optional[np.ndarray]:
    """Load a single raw frame from NuPlan chunked H5 storage."""
    h5_path = os.path.join(data_dir, video_id, "frames.h5")
    if not os.path.exists(h5_path):
        return None
    with h5py.File(h5_path, "r") as f:
        return _resolve_h5_chunk(f, frame_idx)


def load_clip_frames_from_h5(
    data_dir: str, video_id: str, frame_indices: List[int],
) -> List[np.ndarray]:
    """Load multiple raw frames for a clip from NuPlan chunked H5 storage."""
    h5_path = os.path.join(data_dir, video_id, "frames.h5")
    if not os.path.exists(h5_path):
        return []
    frames = []
    with h5py.File(h5_path, "r") as f:
        for idx in frame_indices:
            frames.append(_resolve_h5_chunk(f, idx))
    return frames


@dataclass
class FrameActivation:
    """Activation record for a single frame."""
    video_id: str
    frame_idx: int
    clip_frame_indices: List[int]
    latent_activations: np.ndarray
    odometry: Dict[str, float]


def create_val_dataset(
    data_dir: str,
    num_videos: Optional[int],
    num_frames: int,
    val_split: float,
    size: tuple = (288, 512),
    stored_frame_rate: int = 10,
    target_frame_rate: int = 5,
) -> Dataset:
    """Create the validation dataset using the same split as training.

    Mirrors the split logic in create_datasets_nuplan() from train_sae.py:
    1. Load full dataset with include_odometry=True
    2. Split by video boundaries at val_split fraction
    3. Return the val portion as a Subset
    """
    full_dataset = NuPlanOrbisMultiFrame(
        data_dir=data_dir,
        num_frames=num_frames,
        stored_data_frame_rate=stored_frame_rate,
        target_frame_rate=target_frame_rate,
        size=size,
        num_videos=num_videos,
        include_odometry=True,
        debug=False,
    )

    total_videos = full_dataset.num_videos
    num_val_videos = max(1, int(total_videos * val_split))
    num_train_videos = total_videos - num_val_videos

    if full_dataset._video_clips is not None:
        train_clips = sum(
            nc for _, nc, _ in full_dataset._video_clips[:num_train_videos]
        )
    else:
        clips_per_video = full_dataset.clips_per_video or 1
        train_clips = num_train_videos * clips_per_video

    val_indices = range(train_clips, len(full_dataset))
    val_dataset = Subset(full_dataset, val_indices)

    logger.info(
        f"NuPlan split: {total_videos} videos total, "
        f"{num_train_videos} train / {num_val_videos} val"
    )
    logger.info(f"Val dataset: {len(val_dataset)} clips")

    return val_dataset


class NuPlanValWrapper(Dataset):
    """Wraps a NuPlan val Subset to extract odometry fields for grounding."""

    def __init__(self, val_dataset: Subset):
        self.dataset = val_dataset
        # Access the underlying NuPlanOrbisMultiFrame for clip index recomputation
        base: NuPlanOrbisMultiFrame = val_dataset.dataset  # type: ignore[assignment]
        self._num_frames: int = base.num_frames
        self._frame_interval: int = base.frame_interval

    def __len__(self) -> int:
        return len(self.dataset)

    def _recompute_clip_indices(self, sample: dict) -> List[int]:
        """Recompute full clip frame indices from the target frame index."""
        target_idx = sample["frame_idx"]
        # The target is the last frame: target_idx = start + (num_frames-1) * interval
        start = target_idx - (self._num_frames - 1) * self._frame_interval
        start = max(0, start)
        return [start + i * self._frame_interval for i in range(self._num_frames)]

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        odometry = {
            f: sample.get(f, np.nan)
            for f in ODOMETRY_FIELDS
        }

        clip_indices = self._recompute_clip_indices(sample)

        return {
            "image": sample["images"],  # (T, 3, H, W)
            "video_id": sample["video_id"],
            "frame_idx": sample["frame_idx"],
            "clip_frame_indices": clip_indices,
            "odometry": odometry,
        }


def custom_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate for NuPlan data."""
    images = torch.stack([item["image"] for item in batch])
    video_ids = [item["video_id"] for item in batch]
    frame_idxs = [item["frame_idx"] for item in batch]
    clip_frame_indices = [item["clip_frame_indices"] for item in batch]
    odometry_list = [item["odometry"] for item in batch]

    return {
        "images": images,
        "video_ids": video_ids,
        "frame_idxs": frame_idxs,
        "clip_frame_indices": clip_frame_indices,
        "odometry": odometry_list,
    }


def save_activations_cache(activations: List[FrameActivation], cache_path: Path):
    """Save collected activations to cache."""
    cache_data = {
        "version": 2,
        "num_samples": len(activations),
        "num_latents": activations[0].latent_activations.shape[0] if activations else 0,
        "samples": [
            {
                "video_id": a.video_id,
                "frame_idx": a.frame_idx,
                "clip_frame_indices": a.clip_frame_indices,
                "odometry": a.odometry,
            }
            for a in activations
        ],
        "activations": np.stack([a.latent_activations for a in activations], axis=0),
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache_data, cache_path)
    size_mb = cache_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved {len(activations)} samples to {cache_path} ({size_mb:.1f} MB)")


def load_activations_cache(cache_path: Path) -> Tuple[List[FrameActivation], int]:
    """Load activations from cache.

    Returns:
        Tuple of (activations list, cache version).
    """
    cache_data = torch.load(cache_path, weights_only=False)
    version = cache_data.get("version", 1)

    activations = []
    for i, sample in enumerate(cache_data["samples"]):
        activations.append(FrameActivation(
            video_id=sample["video_id"],
            frame_idx=sample["frame_idx"],
            clip_frame_indices=sample.get("clip_frame_indices", []),
            latent_activations=cache_data["activations"][i],
            odometry=sample["odometry"],
        ))

    logger.info(f"Loaded {len(activations)} samples from {cache_path} (version={version})")
    return activations, version


def _collect_metadata(
    dataloader: DataLoader,
) -> List[Dict[str, Any]]:
    """Collect per-sample metadata from the val dataset (CPU only, no GPU needed).

    Iterates the dataloader to extract video_id, frame_idx, clip_frame_indices,
    and odometry fields. Images are loaded but discarded.
    """
    all_metadata: List[Dict[str, Any]] = []
    for batch in tqdm(dataloader, desc="Collecting metadata"):
        for i in range(len(batch["video_ids"])):
            all_metadata.append({
                "video_id": batch["video_ids"][i],
                "frame_idx": batch["frame_idxs"][i],
                "clip_frame_indices": batch["clip_frame_indices"][i],
                "odometry": batch["odometry"][i],
            })
    return all_metadata


@torch.inference_mode()
def collect_activations_from_val_cache(
    sae: TopKSAE,
    val_cache_dir: Path,
    metadata: List[Dict[str, Any]],
    device: torch.device,
    cache_path: Optional[Path] = None,
    spatial_cache_path: Optional[Path] = None,
    top_k_frames: int = 10,
) -> Tuple[List[FrameActivation], Optional[SpatialTopKTracker]]:
    """Load pre-cached Orbis activations from WebDataset val shards and encode through SAE.

    This skips the Orbis model entirely, loading raw activations from the
    sae_cache val directory that was created during training.

    Args:
        sae: Trained SAE model.
        val_cache_dir: Path to sae_cache val directory with WDS shards.
        metadata: Pre-collected metadata from _collect_metadata().
        device: Device for SAE encoding.
        cache_path: Optional path to save/load SAE activations cache.
        spatial_cache_path: Optional path for spatial top-K cache.
        top_k_frames: Number of top frames per latent for tracker.
    """
    import io
    import webdataset as wds

    tracker = None

    # Check existing caches first
    if cache_path and cache_path.exists():
        activations, version = load_activations_cache(cache_path)
        if spatial_cache_path and spatial_cache_path.exists() and version >= 2:
            tracker = SpatialTopKTracker.load(spatial_cache_path)
            return activations, tracker
        elif version >= 2 and spatial_cache_path:
            logger.info("Activations cached (v2) but spatial cache missing, re-running SAE encoding")
        elif version < 2:
            logger.info("Old cache format (v1), re-running for clip indices + spatial data")
        else:
            return activations, None

    # Load _meta.json from val cache dir
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

    # Custom pt decoder (same as caching.py)
    def _pt_decoder(data: bytes) -> torch.Tensor:
        stream = io.BytesIO(data)
        loaded = torch.load(stream, weights_only=True, map_location="cpu")
        if isinstance(loaded, dict) and "activations" in loaded:
            return loaded["activations"]  # type: ignore[return-value]
        return loaded  # type: ignore[return-value]

    # Load shards sequentially (no shuffle, no unbatch)
    shard_pattern = sorted(val_cache_dir.glob("*.tar"))
    if not shard_pattern:
        raise FileNotFoundError(f"No .tar shards found in {val_cache_dir}")

    dataset = wds.WebDataset(  # type: ignore[attr-defined]
        [str(p) for p in shard_pattern],
        shardshuffle=0,
        empty_check=False,
    ).decode(wds.handle_extension("pt", _pt_decoder)).to_tuple("pt")  # type: ignore[attr-defined]

    # Setup SAE encoding
    d_sae = sae.config.d_sae
    num_spatial = ORBIS_GRID_H * ORBIS_GRID_W  # 576
    tracker = SpatialTopKTracker(num_latents=d_sae, top_k=top_k_frames)
    all_activations: List[FrameActivation] = []
    sample_idx = 0

    for (acts_flat,) in tqdm(dataset, desc="Encoding cached activations", total=num_cache_files):
        # acts_flat: (B * F * N, hidden_dim) flattened batch
        acts_flat = acts_flat.to(device).float()

        # Split into per-sample tensors
        tokens_per_sample = acts_flat.shape[0] // cache_batch_size
        actual_batch_size = acts_flat.shape[0] // tokens_per_sample

        acts_batched = acts_flat.view(actual_batch_size, tokens_per_sample, hidden_dim)

        # SAE encoding
        sae_acts = sae.encode(acts_batched)  # (B, tokens_per_sample, d_sae)
        sae_acts_mean = sae_acts.mean(dim=1).cpu().numpy()  # (B, d_sae)

        # Spatial tracker: target frame only (last 576 tokens)
        target_spatial = sae_acts[:, -num_spatial:, :].cpu().numpy()  # (B, 576, d_sae)

        # Build metadata for tracker
        batch_metadata = []
        for i in range(actual_batch_size):
            if sample_idx + i >= len(metadata):
                break
            meta = metadata[sample_idx + i]
            batch_metadata.append({
                "video_id": meta["video_id"],
                "frame_idx": meta["frame_idx"],
                "clip_frame_indices": meta["clip_frame_indices"],
                "metadata": meta["odometry"],
            })

        if batch_metadata:
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
                clip_frame_indices=meta["clip_frame_indices"],
                latent_activations=sae_acts_mean[i],
                odometry=meta["odometry"],
            ))
            sample_idx += 1

    logger.info(f"Encoded {len(all_activations)} samples from val cache")

    if cache_path:
        save_activations_cache(all_activations, cache_path)
    if spatial_cache_path and tracker:
        tracker.save(spatial_cache_path)

    return all_activations, tracker


@torch.inference_mode()
def collect_activations(
    orbis_model: nn.Module,
    sae: TopKSAE,
    dataloader: DataLoader,
    layer_idx: int,
    device: torch.device,
    frame_rate: int = 5,
    cache_path: Optional[Path] = None,
    spatial_cache_path: Optional[Path] = None,
    top_k_frames: int = 10,
) -> Tuple[List[FrameActivation], Optional[SpatialTopKTracker]]:
    """Collect SAE latent activations and spatial top-K for validation clips.

    Single-pass collection: computes spatially-averaged activations for
    correlation analysis AND maintains per-latent top-K spatial activation
    maps for visualization.

    Returns:
        Tuple of (activations list, spatial tracker or None if loaded from cache).
    """
    tracker = None

    # Check if both caches exist (full v2 cache + spatial)
    if cache_path and cache_path.exists():
        activations, version = load_activations_cache(cache_path)
        if spatial_cache_path and spatial_cache_path.exists() and version >= 2:
            tracker = SpatialTopKTracker.load(spatial_cache_path)
            return activations, tracker
        elif version >= 2 and spatial_cache_path:
            # Have v2 activations but no spatial cache -- need full re-run
            logger.info("Activations cached (v2) but spatial cache missing, re-running collection")
        elif version < 2:
            logger.info("Old cache format (v1), re-running collection for clip indices + spatial data")
        else:
            # No spatial_cache_path requested, return activations only
            return activations, None

    extractor = ActivationExtractor(orbis_model, layer_idx=layer_idx, flatten_spatial=True)
    all_activations = []
    d_sae = sae.config.d_sae
    tracker = SpatialTopKTracker(num_latents=d_sae, top_k=top_k_frames)

    for batch in tqdm(dataloader, desc="Collecting activations"):
        imgs = batch["images"].to(device)
        video_ids = batch["video_ids"]
        frame_idxs = batch["frame_idxs"]
        clip_frame_indices = batch["clip_frame_indices"]
        odometry_list = batch["odometry"]

        b = imgs.shape[0]

        if imgs.dim() == 4:
            imgs = imgs.unsqueeze(1)

        # Encode frames and apply sigma_min noise (matches cache creation)
        x = orbis_model.encode_frames(imgs)
        t_noise = torch.zeros(b, device=device)
        x_noised, _ = orbis_model.add_noise(x, t_noise)
        fr = torch.full((b,), frame_rate, device=device)

        # Split context and target
        if x_noised.shape[1] > 1:
            context = x_noised[:, :-1]
            target = x_noised[:, -1:]
        else:
            context = x_noised
            target = x_noised

        # Extract activations
        with extractor.capture():
            _ = orbis_model.vit(target, context, t_noise, frame_rate=fr)

        acts = extractor.get_activations()
        spatial_tokens = acts.shape[0] // b
        acts = acts.view(b, spatial_tokens, -1)

        # SAE encoding -- keep full spatial for tracker
        sae_acts = sae.encode(acts.float())  # (B, N, d_sae)
        sae_acts_mean = sae_acts.mean(dim=1).cpu().numpy()  # (B, d_sae)

        # For spatial tracker: only the target frame (last 576 tokens)
        num_spatial = ORBIS_GRID_H * ORBIS_GRID_W  # 576
        target_spatial = sae_acts[:, -num_spatial:, :].cpu().numpy()  # (B, 576, d_sae)

        # Build metadata for tracker (uses "metadata" key for data-source-agnostic tracker)
        batch_metadata = [
            {
                "video_id": video_ids[i],
                "frame_idx": frame_idxs[i],
                "clip_frame_indices": clip_frame_indices[i],
                "metadata": odometry_list[i],
            }
            for i in range(b)
        ]

        # Update spatial tracker
        tracker.update(sae_acts_mean, target_spatial, batch_metadata)

        for i in range(b):
            all_activations.append(FrameActivation(
                video_id=video_ids[i],
                frame_idx=frame_idxs[i],
                clip_frame_indices=clip_frame_indices[i],
                latent_activations=sae_acts_mean[i],
                odometry=odometry_list[i],
            ))

    if cache_path:
        save_activations_cache(all_activations, cache_path)
    if spatial_cache_path and tracker:
        tracker.save(spatial_cache_path)

    return all_activations, tracker


def _save_latent_visualizations(
    entries: List[SpatialEntry],
    data_dir: str,
    output_dir: Path,
    latent_idx: int,
) -> List[str]:
    """Save clip/frame/heatmap artifacts for a latent's top entries.

    Returns list of saved frame PNG paths (relative to output_dir parent).
    """
    saved_paths = []
    for rank, entry in enumerate(entries):
        prefix = f"rank{rank + 1}_{entry.video_id}_frame{entry.frame_idx}"

        # Target frame PNG
        frame = load_frame_from_h5(data_dir, entry.video_id, entry.frame_idx)
        if frame is None:
            continue
        frame_rgb = np.asarray(Image.fromarray(frame).convert("RGB"))

        frame_path = output_dir / f"{prefix}.png"
        save_frame_png(frame_rgb, frame_path)
        saved_paths.append(str(frame_path))

        # Patch heatmap (side-by-side)
        heatmap_path = output_dir / f"{prefix}_heatmap.png"
        title = (
            f"Latent {latent_idx} | rank {rank + 1} | "
            f"act={entry.score:.3f} | "
            f"speed={entry.metadata.get('target_speed_kmh', 0):.1f} km/h"
        )
        create_patch_heatmap(frame_rgb, entry.spatial_map, heatmap_path, title=title)

        # Clip video
        if entry.clip_frame_indices:
            clip_frames = load_clip_frames_from_h5(
                data_dir, entry.video_id, entry.clip_frame_indices,
            )
            if clip_frames:
                clip_rgb = [
                    np.asarray(Image.fromarray(f).convert("RGB")) for f in clip_frames
                ]
                clip_path = output_dir / f"{prefix}_clip.mp4"
                create_clip_video(clip_rgb, clip_path)

    return saved_paths


def analyze_top_latents(
    activations: List[FrameActivation],
    tracker: SpatialTopKTracker,
    data_dir: str,
    output_dir: Path,
    top_n_latents: int = 20,
    top_k_frames: int = 10,
) -> Tuple[List[Dict[str, Any]], str]:
    """Analyze top-activating latents and save multi-granularity visualizations.

    For each of the top N latents (by max activation), saves:
    - Target frame PNGs
    - Patch heatmap PNGs (side-by-side original | heatmap)
    - Clip mp4 videos

    Returns:
        Tuple of (top_latents_info list, markdown report string).
    """
    latent_matrix = np.stack([a.latent_activations for a in activations], axis=0)
    top_latents = find_top_activating_latents(latent_matrix, top_n=top_n_latents)
    top_latents_dir = output_dir / "top_latents"

    lines = [
        "# Top Activating Latents Analysis",
        "",
        f"**Total latents:** {latent_matrix.shape[1]}",
        f"**Total clips:** {latent_matrix.shape[0]}",
        f"**Top latents shown:** {top_n_latents}",
        "",
        "---",
        "",
    ]

    for i, latent_info in enumerate(tqdm(top_latents, desc="Saving top latent visualizations")):
        latent_idx = latent_info["latent_idx"]
        entries = tracker.get_top_k(latent_idx)[:top_k_frames]

        latent_dir = top_latents_dir / f"latent_{latent_idx}"
        latent_dir.mkdir(parents=True, exist_ok=True)

        _save_latent_visualizations(entries, data_dir, latent_dir, latent_idx)

        lines.extend([
            f"## #{i + 1}: Latent {latent_idx}",
            "",
            f"- **Max activation:** {latent_info['max']:.4f}",
            f"- **Mean:** {latent_info['mean']:.4f}",
            f"- **Std:** {latent_info['std']:.4f}",
            f"- **Sparsity:** {latent_info['sparsity']:.2%} of clips active",
            "",
            "### Top Activating Clips",
            "",
        ])

        for rank, entry in enumerate(entries):
            vid = entry.video_id
            fidx = entry.frame_idx
            prefix = f"rank{rank + 1}_{vid}_frame{fidx}"
            img_rel = f"top_latents/latent_{latent_idx}/{prefix}.png"
            heatmap_rel = f"top_latents/latent_{latent_idx}/{prefix}_heatmap.png"

            odom = entry.metadata
            lines.extend([
                f"**Rank {rank + 1}:** `{vid}` frame {fidx} (activation: {entry.score:.4f})",
                f"- Speed: {odom.get('target_speed_kmh', 0):.1f} km/h, "
                f"Accel: {odom.get('target_acceleration', 0):.2f} m/s^2, "
                f"Yaw: {odom.get('target_yaw_rate', 0):.4f} rad/s",
                "",
                f"![frame]({img_rel})",
                "",
                f"![heatmap]({heatmap_rel})",
                "",
            ])

        lines.extend(["---", ""])

    # Save top_latents.json
    with open(output_dir / "top_latents.json", "w") as f:
        json.dump(top_latents, f, indent=2)

    report = "\n".join(lines)
    report_path = output_dir / "top_latents_analysis.md"
    with open(report_path, "w") as f:
        f.write(report)

    logger.info(f"Top latents analysis saved to {report_path}")
    return top_latents, report


def save_per_field_visualizations(
    correlations: Dict[str, List[CorrelationResult]],
    tracker: Optional[SpatialTopKTracker],
    data_dir: str,
    output_dir: Path,
    top_latents_per_field: int = 3,
    top_k_frames: int = 10,
) -> None:
    """Save frame/clip/heatmap artifacts for top-correlated latents per odometry field."""
    if tracker is None:
        return

    frames_dir = output_dir / "frames"

    for field_name, results in correlations.items():
        if not results:
            continue

        for result in results[:top_latents_per_field]:
            latent_idx = result.latent_idx
            entries = tracker.get_top_k(latent_idx)[:top_k_frames]
            if not entries:
                continue

            field_dir = frames_dir / f"latent_{latent_idx}_{field_name}"
            field_dir.mkdir(parents=True, exist_ok=True)
            _save_latent_visualizations(entries, data_dir, field_dir, latent_idx)

    logger.info(f"Per-field visualizations saved to {frames_dir}")


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

    for field_name, results in correlations.items():
        report["top_correlations"][field_name] = [
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

    logger.info(f"Saved to {output_path}")


FIELD_DISPLAY_NAMES = {
    "target_speed": "Speed (m/s)",
    "target_speed_kmh": "Speed (km/h)",
    "target_acceleration": "Acceleration (m/s^2)",
    "target_yaw_rate": "Yaw Rate (rad/s)",
    "speed_mean": "Clip Average Speed",
    "acceleration_mean": "Clip Average Acceleration",
    "yaw_rate_mean": "Clip Average Yaw Rate",
    "is_stopped_rate": "Stopped Fraction",
    "is_turning_rate": "Turning Fraction",
}


def generate_markdown_report(
    correlations: Dict[str, List[CorrelationResult]],
    pure_features: List[PureFeature],
    activations: List[FrameActivation],
    tracker: Optional[SpatialTopKTracker],
    output_path: Path,
    top_latents_per_field: int = 3,
):
    """Generate human-readable markdown report with embedded image links."""
    has_viz = tracker is not None

    lines = [
        "# NuPlan SAE Semantic Grounding Analysis",
        "",
        "## Summary",
        f"- Total samples analyzed: {len(activations)}",
        f"- SAE latent dimension: {activations[0].latent_activations.shape[0]}",
        f"- Pure features found: {len(pure_features)}",
        f"- Visualizations: {'enabled' if has_viz else 'disabled'}",
        "",
        "## Top Correlations by Odometry Field",
        "",
    ]

    for field_name in ODOMETRY_FIELDS:
        if field_name not in correlations or not correlations[field_name]:
            continue

        display = FIELD_DISPLAY_NAMES.get(field_name, field_name)

        lines.append(f"### {display}")
        lines.append("")
        lines.append("| Latent | Correlation | p-value |")
        lines.append("|--------|-------------|---------|")

        for r in correlations[field_name][:10]:
            lines.append(f"| {r.latent_idx} | {r.pearson_r:+.3f} | {r.p_value:.2e} |")

        lines.append("")

        # Show top frames with images for the top correlated latents
        if has_viz:
            for result in correlations[field_name][:top_latents_per_field]:
                lat_idx = result.latent_idx
                entries = tracker.get_top_k(lat_idx)[:3]
                if not entries:
                    continue
                lines.append(f"**Latent {lat_idx}** (r={result.pearson_r:+.3f}):")
                lines.append("")
                for rank, entry in enumerate(entries):
                    prefix = f"rank{rank + 1}_{entry.video_id}_frame{entry.frame_idx}"
                    img_rel = f"frames/latent_{lat_idx}_{field_name}/{prefix}.png"
                    lines.append(f"![{prefix}]({img_rel})")
                    lines.append("")
                lines.append("")

    lines.append("## Pure Features")
    lines.append("")
    lines.append("Latents that strongly correlate with one odometry field but not others:")
    lines.append("")

    for pf in pure_features:
        display = pf.primary_field.replace("_", " ").title()
        lines.append(f"### Latent {pf.latent_idx}: {display}")
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
                lines.append(f"   - Acceleration: {odom.get('target_acceleration', 0):.2f} m/s^2")
                lines.append(f"   - Yaw rate: {odom.get('target_yaw_rate', 0):.4f} rad/s")
            lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Saved markdown to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="NuPlan SAE Semantic Grounding Analysis")

    parser.add_argument("--exp_dir", type=str, required=True, help="Orbis experiment directory")
    parser.add_argument("--sae_checkpoint", type=str, required=True, help="Path to SAE checkpoint")
    parser.add_argument("--nuplan_data_dir", type=str, required=True, help="NuPlan data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--layer", type=int, default=22, help="Transformer layer")
    parser.add_argument("--num_frames", type=int, default=6, help="Frames per clip")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_videos", type=int, default=988,
                        help="Total videos to load (must match training config for correct split)")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Val split fraction (must match training config)")
    parser.add_argument("--top_k", type=int, default=20, help="Top-K pure features to report")
    parser.add_argument("--top_n_latents", type=int, default=20,
                        help="Number of top-activating latents for concept discovery")
    parser.add_argument("--top_k_frames", type=int, default=10,
                        help="Number of top frames per latent for visualization")
    parser.add_argument("--no_cache", action="store_true", help="Don't use activation cache")
    parser.add_argument("--rebuild_cache", action="store_true", help="Force rebuild activation cache")
    parser.add_argument("--val_cache_dir", type=str, default=None,
                        help="Path to pre-cached val activations (sae_cache WDS shards). "
                        "Skips Orbis model loading when provided.")
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

    # Load SAE (always needed)
    sae = load_sae(Path(args.sae_checkpoint), device)
    logger.info(f"SAE loaded: {sae.config.d_in} -> {sae.config.d_sae} (k={sae.config.k})")

    # Create val dataset for metadata (and optionally for forward passes)
    val_dataset = create_val_dataset(
        data_dir=args.nuplan_data_dir,
        num_videos=args.num_videos,
        num_frames=args.num_frames,
        val_split=args.val_split,
    )
    wrapped_dataset = NuPlanValWrapper(val_dataset)

    val_loader = DataLoader(
        wrapped_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate,
    )

    # Collect activations + spatial top-K
    cache_path = None if args.no_cache else output_dir / "activations_cache.pt"
    spatial_cache_path = None if args.no_cache else output_dir / "spatial_topk_cache.pt"

    if args.rebuild_cache:
        if cache_path and cache_path.exists():
            cache_path.unlink()
        if spatial_cache_path and spatial_cache_path.exists():
            spatial_cache_path.unlink()

    if args.val_cache_dir:
        # Fast path: load pre-cached Orbis activations from WDS shards
        val_cache_dir = Path(args.val_cache_dir)
        logger.info(f"Using pre-cached val activations from {val_cache_dir}")
        metadata = _collect_metadata(val_loader)
        activations, tracker = collect_activations_from_val_cache(
            sae, val_cache_dir, metadata, device,
            cache_path=cache_path,
            spatial_cache_path=spatial_cache_path,
            top_k_frames=args.top_k_frames,
        )
    else:
        # Standard path: run Orbis forward passes
        orbis_model = load_orbis_model(
            exp_dir / "config.yaml",
            exp_dir / "checkpoints" / "last.ckpt",
            device,
        )
        activations, tracker = collect_activations(
            orbis_model, sae, val_loader, args.layer, device,
            cache_path=cache_path,
            spatial_cache_path=spatial_cache_path,
            top_k_frames=args.top_k_frames,
        )

    # Build matrices for shared analysis functions
    latent_matrix = np.stack([a.latent_activations for a in activations], axis=0)
    odom_matrix = np.array([
        [a.odometry.get(f, np.nan) for f in ODOMETRY_FIELDS]
        for a in activations
    ])
    sample_metadata = [
        {"video_id": a.video_id, "frame_idx": a.frame_idx, "odometry": a.odometry}
        for a in activations
    ]

    # Compute correlations
    correlations = compute_correlations(latent_matrix, odom_matrix, ODOMETRY_FIELDS)

    # Find pure features
    pure_features = find_pure_features(correlations, top_k=args.top_k)

    # Add top frames to pure features
    for pf in pure_features:
        pf.top_frames = find_top_activating_frames(
            latent_matrix, pf.latent_idx, sample_metadata, top_k=5,
        )

    # Generate JSON and markdown reports
    generate_report(correlations, pure_features, output_dir / "analysis_results.json")

    # Multi-granularity visualizations
    if tracker is not None:
        # Top latent concept discovery
        analyze_top_latents(
            activations, tracker, args.nuplan_data_dir, output_dir,
            top_n_latents=args.top_n_latents,
            top_k_frames=args.top_k_frames,
        )

        # Per-field correlated frames
        save_per_field_visualizations(
            correlations, tracker, args.nuplan_data_dir, output_dir,
            top_k_frames=args.top_k_frames,
        )

    generate_markdown_report(
        correlations, pure_features, activations, tracker,
        output_dir / "analysis_report.md",
    )

    logger.info("Analysis complete!")
    logger.info(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
