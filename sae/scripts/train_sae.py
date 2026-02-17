#!/usr/bin/env python3
"""
Training script for Top-K Sparse Autoencoder on Orbis world model activations.

This script:
1. Loads a pre-trained Orbis world model (frozen)
2. Extracts and caches ST-Transformer activations from a specified layer
3. Trains a Top-K SAE with MSE loss only (no sparsity penalty)
4. Logs metrics and saves checkpoints

Usage:
    python orbis/sae/scripts/train_sae.py \
        --exp_dir /path/to/orbis_experiment \
        --data_path /path/to/Cityscapes \
        --layer 12 \
        --k 64 \
        --expansion_factor 16

See orbis/sae/configs/default.yaml for all configuration options.
"""

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ORBIS_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ORBIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ORBIS_ROOT))

from pytorch_lightning import seed_everything

from sae.topk_sae import TopKSAE, TopKSAEConfig, TopKSAETrainer
from sae.caching import (
    CacheConfig,
    CacheResumeInfo,
    get_cache_resume_info,
    prepare_activation_cache,
    create_activation_dataloader,
    create_webdataset_dataloader,
    resolve_cache_dtype,
)
from sae.utils.constants import (
    BYTES_PER_FLOAT16,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SAE_BATCH_SIZE,
)
from sae.utils.data_utils import is_webdataset_mode, get_base_data_source
from sae.utils.model_loading import load_orbis_model
from sae.utils.logging import get_logger, format_duration, GPUMonitor, stats, EpochStats

logger = get_logger(__name__)

from semantic_stage2 import CityScapes
from data.custom_multiframe import (
    MultiHDF5DatasetMultiFrameRandomizeFrameRate,
    MultiHDF5DatasetMultiFrameFromJSONFrameRateWrapper,
)
from data.multiframe_val import MultiFrameFromPaths
from data.covla.covla_dataset import CoVLAOrbisMultiFrame
from data.nuplan.nuplan_dataset import NuPlanOrbisMultiFrame



def resolve_sae_batch_size(
    batch_size_arg: str,
    layer: int,
    expansion: int,
    k: int,
    default: int = DEFAULT_SAE_BATCH_SIZE,
) -> int:
    """
    Resolve SAE batch size from argument.
    
    Args:
        batch_size_arg: "auto" or an integer string
        layer: Transformer layer number
        expansion: SAE expansion factor
        k: Top-K sparsity value
        default: Default batch size if "auto" cannot resolve
        
    Returns:
        Resolved batch size as integer
    """
    if batch_size_arg.lower() == "auto":
        # Try to read from calibration resource
        try:
            from sae.calibrate_batch_size import (
                get_gpu_info,
                get_sae_resource_path,
                load_resource,
            )
            
            gpu_info = get_gpu_info()
            gpu_slug = gpu_info.slug
            
            resource_path = get_sae_resource_path(layer, expansion, k, gpu_slug)
            calibration = load_resource(resource_path)
            
            if calibration is None:
                logger.warning(f"No calibration resource found at {resource_path}, using default")
                return default
            
            batch_size = calibration["recommended_batch_size_tokens"]
            logger.info(f"Auto-detected batch size from calibration: {batch_size:,} tokens")
            logger.info(f"  (GPU: {calibration.get('gpu', gpu_slug)})")
            return batch_size
            
        except Exception as e:
            logger.warning(f"Failed to auto-detect batch size: {e}, using default")
            return default
    else:
        # Parse as integer
        try:
            return int(batch_size_arg)
        except ValueError:
            logger.error(f"Invalid sae_batch_size: {batch_size_arg}. Must be 'auto' or an integer.")
            raise




def create_dataloaders_cityscapes(
    data_path: str,
    input_size: tuple,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple:
    """Create train and validation dataloaders for Cityscapes."""

    dataset_train = CityScapes(
        size=input_size,
        data_path=data_path,
        split="train",
        mode="fine",
        target_type="semantic",
    )

    dataset_val = CityScapes(
        size=input_size,
        data_path=data_path,
        split="val",
        mode="fine",
        target_type="semantic",
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    train_loader = DataLoader(dataset_train, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(dataset_val, shuffle=False, **loader_kwargs)

    return train_loader, val_loader


def create_dataloaders_image_paths(
    image_paths: list,
    input_size: tuple,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    num_frames: int = 5,
) -> tuple:
    """
    Create train and validation dataloaders from a list of image paths.
    
    This is useful for quick testing with example frames.
    Uses the same data for train and val (since we only have a few frames).
    
    Args:
        image_paths: List of paths to image files
        input_size: Image size (H, W)
        batch_size: Batch size
        num_workers: Number of data loading workers
        device: Device for pin_memory
        num_frames: Number of frames per sample
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create dataset from image paths
    dataset = MultiFrameFromPaths(
        size=input_size,
        image_paths=image_paths,
        num_frames=min(num_frames, len(image_paths)),
    )
    
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    
    # Use same dataset for train and val (small example data)
    train_loader = DataLoader(dataset, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(dataset, shuffle=False, **loader_kwargs)
    
    logger.info(f"Image paths dataset: {len(dataset)} samples, {len(image_paths)} frames")
    
    return train_loader, val_loader


def create_dataloaders_hdf5(
    hdf5_paths_file: str,
    val_json: str,
    input_size: tuple,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    num_frames: int = 6,
    stored_data_frame_rate: int = 5,
    frame_rate: int = 5,
) -> tuple:
    """
    Create train and validation dataloaders for HDF5 video data.

    This uses the same data format that Orbis was trained on.

    Args:
        hdf5_paths_file: Path to text file listing HDF5 file paths (one per line)
        val_json: Path to JSON file with validation samples
        input_size: Image size (H, W)
        batch_size: Batch size
        num_workers: Number of data loading workers
        device: Device for pin_memory
        num_frames: Number of frames per sample
        stored_data_frame_rate: Frame rate of stored data
        frame_rate: Target frame rate

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Training dataset - HDF5 with randomized frame rate
    dataset_train = MultiHDF5DatasetMultiFrameRandomizeFrameRate(
        size=input_size,
        hdf5_paths_file=hdf5_paths_file,
        num_frames=num_frames,
        stored_data_frame_rate=stored_data_frame_rate,
        frame_rates_and_weights=[
            (frame_rate, 1.0)
        ],  # Single frame rate for SAE training
        aug="resize_center",  # Simple augmentation for activation consistency
    )

    # Validation dataset - from JSON
    dataset_val = MultiHDF5DatasetMultiFrameFromJSONFrameRateWrapper(
        size=input_size,
        samples_json=val_json,
        num_frames=num_frames,
        stored_data_frame_rate=stored_data_frame_rate,
        frame_rate=frame_rate,
        num_samples=500,  # Limit validation samples
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    train_loader = DataLoader(dataset_train, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(dataset_val, shuffle=False, **loader_kwargs)

    logger.info(f"HDF5 train dataset: {len(dataset_train)} samples")
    logger.info(f"HDF5 val dataset: {len(dataset_val)} samples")

    return train_loader, val_loader


def create_datasets_covla(
    videos_dir: str,
    captions_dir: Optional[str],
    input_size: tuple,
    num_frames: int = 6,
    stored_data_frame_rate: int = 20,
    frame_rate: int = 5,
    num_videos: Optional[int] = None,
    val_split: float = 0.1,
) -> Tuple[Dataset, Dataset]:
    """
    Create train and validation datasets for CoVLA video data.
    Returns Dataset objects (not DataLoaders) to enable zero-cost resume via Subset.
    
    Args:
        videos_dir: Path to CoVLA videos directory
        captions_dir: Path to CoVLA captions directory (optional)
        input_size: Target image size (H, W)
        num_frames: Number of frames per clip
        stored_data_frame_rate: Frame rate of stored videos
        frame_rate: Target frame rate for sampling
        num_videos: Limit number of videos (None for all)
        val_split: Fraction of videos for validation
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # 1. Instantiate the Full Dataset ONCE
    full_dataset = CoVLAOrbisMultiFrame(
        num_frames=num_frames,
        stored_data_frame_rate=stored_data_frame_rate,
        target_frame_rate=frame_rate,
        size=input_size,
        captions_dir=captions_dir,
        videos_dir=videos_dir,
        num_samples=num_videos,
        debug=False,
    )
    
    # 2. Calculate Split Boundaries based on VIDEO count
    total_videos = full_dataset.num_videos
    clips_per_video = full_dataset.clips_per_video
    
    num_val_videos = max(1, int(total_videos * val_split))
    num_train_videos = total_videos - num_val_videos
    
    # 3. Create Indices
    train_end_idx = num_train_videos * clips_per_video
    train_indices = range(0, train_end_idx)
    val_indices = range(train_end_idx, len(full_dataset))
    
    # 4. Create Subsets
    dataset_train = Subset(full_dataset, train_indices)
    dataset_val = Subset(full_dataset, val_indices)

    logger.info(f"CoVLA Full: {total_videos} videos, {len(full_dataset)} total clips")
    logger.info(f"Train Split: {num_train_videos} videos ({len(dataset_train)} clips)")
    logger.info(f"Val Split:   {num_val_videos} videos ({len(dataset_val)} clips)")

    return dataset_train, dataset_val


def create_dataloaders_covla(
    videos_dir: str,
    captions_dir: Optional[str],
    input_size: tuple,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    num_frames: int = 6,
    stored_data_frame_rate: int = 20,
    frame_rate: int = 5,
    num_videos: Optional[int] = None,
    val_split: float = 0.1,
) -> tuple:
    """
    Create train and validation dataloaders for CoVLA video data.
    Uses Subset to strictly split video IDs without leakage.
    
    NOTE: For zero-cost resume, use create_datasets_covla() instead and
    apply Subset before creating DataLoaders.
    """
    dataset_train, dataset_val = create_datasets_covla(
        videos_dir=videos_dir,
        captions_dir=captions_dir,
        input_size=input_size,
        num_frames=num_frames,
        stored_data_frame_rate=stored_data_frame_rate,
        frame_rate=frame_rate,
        num_videos=num_videos,
        val_split=val_split,
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # shuffle=False is correct here because we are caching sequentially
    train_loader = DataLoader(dataset_train, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(dataset_val, shuffle=False, **loader_kwargs)

    return train_loader, val_loader


def create_datasets_nuplan(
    data_dir: str,
    input_size: tuple,
    num_frames: int = 6,
    stored_data_frame_rate: int = 10,
    frame_rate: int = 5,
    num_videos: Optional[int] = None,
    val_split: float = 0.1,
) -> Tuple[Dataset, Dataset]:
    """
    Create train and validation datasets for NuPlan video data.
    Returns Dataset objects (not DataLoaders) to enable zero-cost resume via Subset.
    
    Args:
        data_dir: Path to NuPlan data directory
        input_size: Target image size (H, W)
        num_frames: Number of frames per clip
        stored_data_frame_rate: Frame rate of stored videos
        frame_rate: Target frame rate for sampling
        num_videos: Limit number of videos (None for all)
        val_split: Fraction of videos for validation
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # 1. Instantiate the Full Dataset ONCE
    full_dataset = NuPlanOrbisMultiFrame(
        data_dir=data_dir,
        num_frames=num_frames,
        stored_data_frame_rate=stored_data_frame_rate,
        target_frame_rate=frame_rate,
        size=input_size,
        num_videos=num_videos,
        include_odometry=False,  # Don't need odometry for training
        debug=False,
    )
    
    # 2. For NuPlan, clips_per_video varies, so we need to handle variable clips
    if full_dataset._video_clips is not None:
        # Variable clips mode - split by cumulative clip count
        total_videos = full_dataset.num_videos
        num_val_videos = max(1, int(total_videos * val_split))
        num_train_videos = total_videos - num_val_videos
        
        # Calculate train end index based on video boundaries
        train_clips = sum(nc for _, nc, _ in full_dataset._video_clips[:num_train_videos])
        train_indices = range(0, train_clips)
        val_indices = range(train_clips, len(full_dataset))
    else:
        # Fixed clips_per_video mode (shouldn't happen for nuplan but handle it)
        total_videos = full_dataset.num_videos
        clips_per_video = full_dataset.clips_per_video
        num_val_videos = max(1, int(total_videos * val_split))
        num_train_videos = total_videos - num_val_videos
        train_end_idx = num_train_videos * clips_per_video
        train_indices = range(0, train_end_idx)
        val_indices = range(train_end_idx, len(full_dataset))
    
    # 3. Create Subsets
    dataset_train = Subset(full_dataset, train_indices)
    dataset_val = Subset(full_dataset, val_indices)

    logger.info(f"NuPlan Full: {total_videos} videos, {len(full_dataset)} total clips")
    logger.info(f"Train Split: {num_train_videos} videos ({len(dataset_train)} clips)")
    logger.info(f"Val Split:   {num_val_videos} videos ({len(dataset_val)} clips)")

    return dataset_train, dataset_val


def create_dataloaders_nuplan(
    data_dir: str,
    input_size: tuple,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    num_frames: int = 6,
    stored_data_frame_rate: int = 10,
    frame_rate: int = 5,
    num_videos: Optional[int] = None,
    val_split: float = 0.1,
) -> tuple:
    """
    Create train and validation dataloaders for NuPlan video data.
    Uses Subset to strictly split video IDs without leakage.
    
    NOTE: For zero-cost resume, use create_datasets_nuplan() instead and
    apply Subset before creating DataLoaders.
    """
    dataset_train, dataset_val = create_datasets_nuplan(
        data_dir=data_dir,
        input_size=input_size,
        num_frames=num_frames,
        stored_data_frame_rate=stored_data_frame_rate,
        frame_rate=frame_rate,
        num_videos=num_videos,
        val_split=val_split,
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    train_loader = DataLoader(dataset_train, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(dataset_val, shuffle=False, **loader_kwargs)

    return train_loader, val_loader


def create_dataloader_with_offset(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    start_sample_idx: int = 0,
) -> DataLoader:
    """
    Create a DataLoader, optionally subsetting the dataset for zero-cost resume.
    
    If start_sample_idx > 0, creates a Subset containing only samples from
    that index onwards, enabling true zero-cost resume without iterating
    through already-processed samples.
    
    Args:
        dataset: The source dataset (may already be a Subset)
        batch_size: Batch size for the DataLoader
        num_workers: Number of data loading workers
        device: Device (used to determine pin_memory)
        start_sample_idx: Starting sample index for resume (0 = no resume)
        
    Returns:
        DataLoader configured for sequential caching (shuffle=False)
    """
    # Apply resume subset if needed
    if start_sample_idx > 0:
        if start_sample_idx >= len(dataset):
            # All samples already processed - return empty dataloader
            remaining_indices = range(0, 0)  # Empty range
        else:
            remaining_indices = range(start_sample_idx, len(dataset))
        dataset = Subset(dataset, remaining_indices)
        logger.info(f"Applied resume subset: {len(dataset)} samples remaining")
    
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        shuffle=False,  # Must be False for deterministic caching
    )
    
    return DataLoader(dataset, **loader_kwargs)


def create_dataloaders(
    data_source: str,
    data_path: str,
    input_size: tuple,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    hdf5_paths_file: Optional[str] = None,
    val_json: Optional[str] = None,
    num_frames: int = 6,
    stored_data_frame_rate: int = 5,
    frame_rate: int = 5,
    image_paths: Optional[list] = None,
    covla_videos_dir: Optional[str] = None,
    covla_captions_dir: Optional[str] = None,
    nuplan_data_dir: Optional[str] = None,
    num_videos: Optional[int] = None,
) -> tuple:
    """Create dataloaders based on data source type.

    Args:
        data_source: "cityscapes", "hdf5", "image_paths", "covla", or "nuplan"
        data_path: Path to Cityscapes root (if cityscapes)
        input_size: Image size (H, W)
        batch_size: Batch size
        num_workers: Number of workers
        device: Device
        hdf5_paths_file: Path to HDF5 paths file (required if hdf5)
        val_json: Path to validation JSON (required if hdf5)
        num_frames: Number of frames per sample (for hdf5/image_paths/covla/nuplan)
        stored_data_frame_rate: Native FPS of source data (5 for hdf5, 20 for covla, 10 for nuplan)
        frame_rate: Target frame rate
        image_paths: List of image file paths (required if image_paths)
        covla_videos_dir: Directory with CoVLA videos (required if covla)
        covla_captions_dir: Directory with CoVLA captions (optional if covla)
        nuplan_data_dir: Directory with NuPlan video folders (required if nuplan)
        num_videos: Number of videos to use (for covla/nuplan)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    if data_source == "cityscapes":
        return create_dataloaders_cityscapes(
            data_path, input_size, batch_size, num_workers, device
        )
    elif data_source == "hdf5":
        if hdf5_paths_file is None or val_json is None:
            raise ValueError(
                "hdf5_paths_file and val_json required for HDF5 data source"
            )
        return create_dataloaders_hdf5(
            hdf5_paths_file,
            val_json,
            input_size,
            batch_size,
            num_workers,
            device,
            num_frames,
            stored_data_frame_rate,
            frame_rate,
        )
    elif data_source == "image_paths":
        if image_paths is None or len(image_paths) == 0:
            raise ValueError(
                "image_paths required for image_paths data source"
            )
        return create_dataloaders_image_paths(
            image_paths,
            input_size,
            batch_size,
            num_workers,
            device,
            num_frames,
        )
    elif data_source == "covla":
        if covla_videos_dir is None:
            raise ValueError(
                "covla_videos_dir required for CoVLA data source"
            )
        # CoVLA videos are stored at 20 FPS - enforce this to avoid errors
        if stored_data_frame_rate != 20:
            logger.warning(f"CoVLA videos are 20 FPS, but stored_frame_rate={stored_data_frame_rate}. Forcing to 20.")
            stored_data_frame_rate = 20
        return create_dataloaders_covla(
            videos_dir=covla_videos_dir,
            captions_dir=covla_captions_dir,
            input_size=input_size,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            num_frames=num_frames,
            stored_data_frame_rate=stored_data_frame_rate,
            frame_rate=frame_rate,
            num_videos=num_videos,
        )
    elif data_source == "nuplan":
        if nuplan_data_dir is None:
            raise ValueError(
                "nuplan_data_dir required for NuPlan data source"
            )
        # NuPlan videos are stored at 10 FPS - enforce this to avoid errors
        if stored_data_frame_rate != 10:
            logger.warning(f"NuPlan videos are 10 FPS, but stored_frame_rate={stored_data_frame_rate}. Forcing to 10.")
            stored_data_frame_rate = 10
        return create_dataloaders_nuplan(
            data_dir=nuplan_data_dir,
            input_size=input_size,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            num_frames=num_frames,
            stored_data_frame_rate=stored_data_frame_rate,
            frame_rate=frame_rate,
            num_videos=num_videos,
        )
    else:
        raise ValueError(
            f"Unknown data source: {data_source}. Use 'cityscapes', 'hdf5', 'image_paths', 'covla', or 'nuplan'"
        )


def train_epoch(
    trainer: TopKSAETrainer,
    dataloader: DataLoader,
    epoch: int,
    total_epochs: int,
    gpu_monitor: Optional[GPUMonitor] = None,
    steps_per_epoch: Optional[int] = None,
    checkpoint_callback: Optional[Callable[[int, int], None]] = None,
    checkpoint_steps: Optional[set] = None,
) -> Tuple[Dict[str, float], EpochStats]:
    """
    Train for one epoch with full Phase 1 metric logging and IO wait tracking.
    
    Args:
        trainer: SAE trainer instance
        dataloader: DataLoader or WebLoader for training data
        epoch: Current epoch number
        total_epochs: Total number of epochs
        gpu_monitor: Optional GPU monitor for tracking utilization
        steps_per_epoch: Number of steps per epoch (required for IterableDatasets
                        like WebLoader that don't have __len__)
        checkpoint_callback: Optional callback for intra-epoch checkpoints.
            Called as callback(step, total_steps) at steps in checkpoint_steps.
        checkpoint_steps: Set of step numbers at which to call checkpoint_callback.
    
    Returns:
        Tuple of (averaged metrics dict, EpochStats with IO timing)
    """
    epoch_stats = EpochStats(epoch_num=epoch)
    metrics_sum = {
        "loss": 0.0,
        "mse_loss": 0.0,
        "ghost_loss": 0.0,
        "l0": 0.0,
        "cos_sim": 0.0,
        "rel_error": 0.0,
        "explained_variance": 0.0,
        "activation_density": 0.0,
        "l1_norm": 0.0,
        "dead_pct": 0.0,
        "lr": 0.0,
    }
    # Per-metric valid counts (some metrics can be NaN from fp16 edge cases)
    valid_counts = {k: 0 for k in metrics_sum}
    num_batches = 0

    if gpu_monitor is not None:
        gpu_monitor.set_phase(f"train_epoch_{epoch}")

    data_iter = iter(dataloader)
    # Use steps_per_epoch for IterableDatasets like WebLoader that don't have __len__
    try:
        num_steps = len(dataloader)
    except TypeError:
        if steps_per_epoch is None:
            raise ValueError(
                "steps_per_epoch is required for IterableDatasets (e.g., WebLoader) "
                "that don't have __len__"
            )
        num_steps = steps_per_epoch

    pbar = tqdm(range(num_steps), desc=f"Epoch {epoch}/{total_epochs} [train]",
                file=sys.stdout, mininterval=360.0)

    for _ in pbar:
        io_start = time.perf_counter()
        try:
            batch = next(data_iter)
        except StopIteration:
            break
        io_wait = time.perf_counter() - io_start

        compute_start = time.perf_counter()
        metrics = trainer.train_step(batch)
        compute_time = time.perf_counter() - compute_start

        batch_size = batch.shape[0] if hasattr(batch, 'shape') else len(batch)
        epoch_stats.add_batch(
            io_wait=io_wait,
            compute_time=compute_time,
            batch_size=batch_size,
            loss=metrics.get('loss', 0.0),
        )

        # NaN-safe accumulation: skip NaN/Inf values to prevent poisoning averages
        for k in metrics_sum.keys():
            if k in metrics and math.isfinite(metrics[k]):
                metrics_sum[k] += metrics[k]
                valid_counts[k] += 1
        num_batches += 1

        io_pct = (epoch_stats.io_wait_time / epoch_stats.total_time * 100) if epoch_stats.total_time > 0 else 0
        pbar.set_postfix({
            "loss": f"{metrics['loss']:.4f}",
            "l0": f"{metrics['l0']:.1f}",
            "dead%": f"{metrics['dead_pct']:.1f}",
            "io%": f"{io_pct:.1f}",
        })

        if checkpoint_callback is not None and checkpoint_steps and num_batches in checkpoint_steps:
            checkpoint_callback(num_batches, num_steps)

    if gpu_monitor is not None:
        gpu_stats = gpu_monitor.get_phase_stats(f"train_epoch_{epoch}")
        if 'avg_util_pct' in gpu_stats:
            epoch_stats.gpu_util_samples.append(gpu_stats['avg_util_pct'])
        if 'peak_memory_gb' in gpu_stats:
            epoch_stats.gpu_memory_samples.append(gpu_stats['peak_memory_gb'] * 1e9)
        gpu_monitor.clear_phase()

    # Average metrics using per-metric valid counts to handle NaN batches
    avg_metrics = {
        k: v / valid_counts[k] if valid_counts[k] > 0 else float('nan')
        for k, v in metrics_sum.items()
    }

    return avg_metrics, epoch_stats


@torch.no_grad()
def eval_epoch(
    trainer: TopKSAETrainer,
    dataloader: DataLoader,
) -> Dict[str, float]:
    """Evaluate on validation set with full Phase 1 metric logging."""
    
    metrics_sum = {
        "loss": 0.0,
        "mse_loss": 0.0,
        "ghost_loss": 0.0,
        "l0": 0.0,
        "cos_sim": 0.0,
        "rel_error": 0.0,
        "explained_variance": 0.0,
        "activation_density": 0.0,
        "l1_norm": 0.0,
        "dead_pct": 0.0,
        "lr": 0.0,
    }
    valid_counts = {k: 0 for k in metrics_sum}
    num_batches = 0
    
    eval_start = time.perf_counter()
    for batch in tqdm(dataloader, desc="Evaluating", file=sys.stdout, mininterval=360.0):
        metrics = trainer.eval_step(batch)
        
        for k in metrics_sum.keys():
            if k in metrics and math.isfinite(metrics[k]):
                metrics_sum[k] += metrics[k]
                valid_counts[k] += 1
        num_batches += 1
    
    eval_time = time.perf_counter() - eval_start
    avg_metrics = {
        k: v / valid_counts[k] if valid_counts[k] > 0 else float('nan')
        for k, v in metrics_sum.items()
    }
    avg_metrics['eval_time_s'] = eval_time
    
    return avg_metrics



@dataclass
class ExperimentPaths:
    """Bundles path variables that flow through the training pipeline."""
    output_dir: Path
    cache_dir: Path
    train_cache_dir: Path
    val_cache_dir: Path
    config_path: Path
    exp_dir: Path
    orbis_root: Path
    barcode: str
    timestamp: str

    def create_directories(self) -> None:
        """Create all experiment directories."""
        for d in [self.output_dir, self.train_cache_dir, self.val_cache_dir]:
            d.mkdir(parents=True, exist_ok=True)



def _setup_experiment(
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple["ExperimentPaths", Optional[nn.Module], int, bool, str, torch.dtype]:
    """Seed, device, path resolution, directory creation, model loading.

    Returns:
        Tuple of (paths, model, hidden_size, use_webdataset, base_data_source, cache_dtype)
    """
    if args.seed > 0:
        seed_everything(args.seed)

    exp_dir = Path(args.exp_dir)
    config_path = exp_dir / args.config
    orbis_root = Path(__file__).resolve().parents[2]

    cache_base = orbis_root / "logs_sae" / "sae_cache"
    cache_dir = cache_base / args.data_source / exp_dir.name / f"layer_{args.layer}"

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.run_name:
        barcode = args.run_name
    else:
        raise ValueError("run_name is required")

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        runs_base = orbis_root / "logs_sae" / "runs"
        output_dir = runs_base / args.data_source / exp_dir.name / f"layer_{args.layer}" / barcode

    exp = ExperimentPaths(
        output_dir=output_dir,
        cache_dir=cache_dir,
        train_cache_dir=cache_dir / "train",
        val_cache_dir=cache_dir / "val",
        config_path=config_path,
        exp_dir=exp_dir,
        orbis_root=orbis_root,
        barcode=barcode,
        timestamp=timestamp,
    )
    exp.create_directories()

    logger.info(f"[barcode] {barcode}")
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Cache directory: {cache_dir}")
    logger.info(f"Output directory: {output_dir}")

    base_data_source = get_base_data_source(args.data_source)
    use_webdataset = is_webdataset_mode(args.data_source)

    skip_model_load = use_webdataset or args.train_only

    if skip_model_load:
        model = None
        hidden_size = 768  # Standard hidden size for Orbis/ViT-Base
        reason = "WebDataset mode" if use_webdataset else "--train_only flag"
        logger.info(f"Skipping model load ({reason}) - using hidden_size={hidden_size}")
    else:
        ckpt_path = exp_dir / args.ckpt
        model = load_orbis_model(str(config_path), str(ckpt_path), device)
        hidden_size = model.vit.blocks[0].norm1.normalized_shape[0]
        logger.info(f"ST-Transformer hidden size: {hidden_size}")

    cache_dtype = resolve_cache_dtype(args.cache_dtype)
    return exp, model, hidden_size, use_webdataset, base_data_source, cache_dtype


def _create_sae_and_trainer(
    args: argparse.Namespace,
    hidden_size: int,
    device: torch.device,
    total_training_steps: int = 0,
) -> Tuple[TopKSAE, TopKSAETrainer, TopKSAEConfig]:
    """Instantiate SAE model and trainer."""
    decoder_init_norm = args.decoder_init_norm if args.decoder_init_norm > 0 else None
    sae_config = TopKSAEConfig(
        d_in=hidden_size,
        expansion_factor=args.expansion_factor,
        k=args.k,
        dead_feature_window=args.dead_feature_window,
        aux_loss_coefficient=args.aux_loss_coefficient,
        decoder_init_norm=decoder_init_norm,
        normalize_activations=args.normalize_activations,
        b_dec_init_method=args.b_dec_init,
    )
    sae = TopKSAE(sae_config)
    logger.info(f"Created SAE: {sae}")

    trainer = TopKSAETrainer(
        model=sae,
        lr=args.lr,
        device=device,
        compile_model=args.compile,
        max_grad_norm=args.max_grad_norm,
        total_training_steps=total_training_steps,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_end_factor=args.lr_end_factor,
    )
    logger.info("fp32 training with Adam optimizer")
    logger.info(
        f"LR schedule: cosine warmup ({args.lr_warmup_steps} warmup steps, "
        f"end_factor={args.lr_end_factor})"
    )
    logger.info(
        f"Aux TopK loss coefficient: {args.aux_loss_coefficient}, "
        f"dead_feature_window: {args.dead_feature_window}"
    )
    if args.compile:
        logger.info("torch.compile enabled")

    return sae, trainer, sae_config


def _save_experiment_config(
    args: argparse.Namespace,
    exp: ExperimentPaths,
    sae_config: TopKSAEConfig,
    train_meta: Dict[str, Any],
    val_meta: Dict[str, Any],
) -> None:
    """Serialize all config to JSON for reproducibility."""
    git_commit = None
    try:
        import subprocess
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=exp.orbis_root,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
    except Exception:
        pass

    config_save = {
        "sae_config": {
            "d_in": sae_config.d_in,
            "d_sae": sae_config.d_sae,
            "expansion_factor": sae_config.expansion_factor,
            "k": sae_config.k,
            "aux_loss_coefficient": sae_config.aux_loss_coefficient,
            "dead_feature_window": sae_config.dead_feature_window,
            "decoder_init_norm": sae_config.decoder_init_norm,
            "rescale_acts_by_decoder_norm": sae_config.rescale_acts_by_decoder_norm,
            "normalize_activations": sae_config.normalize_activations,
            "b_dec_init_method": sae_config.b_dec_init_method,
        },
        "training": {
            "lr": args.lr,
            "max_grad_norm": args.max_grad_norm,
            "lr_warmup_steps": args.lr_warmup_steps,
            "lr_end_factor": args.lr_end_factor,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "eval_every": args.eval_every,
            "save_every": args.save_every,
            "seed": args.seed,
        },
        "activation": {
            "layer_idx": args.layer,
            "t_noise": args.t_noise,
            "frame_rate": args.frame_rate,
        },
        "data": {
            "data_source": args.data_source,
            "input_size": args.input_size,
            "num_frames": args.num_frames,
            "stored_frame_rate": args.stored_frame_rate,
            "num_videos": args.num_videos,
            "num_workers": args.num_workers,
            "data_path": args.data_path,
            "hdf5_paths_file": args.hdf5_paths_file,
            "val_json": args.val_json,
            "covla_videos_dir": args.covla_videos_dir,
            "covla_captions_dir": args.covla_captions_dir,
            "nuplan_data_dir": args.nuplan_data_dir,
            "image_paths": args.image_paths,
        },
        "cache": {
            "cache_dir": str(exp.cache_dir),
            "cache_dtype": args.cache_dtype,
            "streaming": args.streaming,
        },
        "orbis": {
            "exp_dir": str(exp.exp_dir),
            "ckpt": args.ckpt,
            "config": args.config,
        },
        "output": {
            "output_dir": str(exp.output_dir),
            "run_name": exp.barcode,
        },
        "metadata": {
            "git_commit": git_commit,
            "device": args.device,
            "timestamp": exp.timestamp,
            "train_tokens": train_meta["total_tokens"],
            "val_tokens": val_meta["total_tokens"],
        },
    }

    with open(exp.output_dir / "config.json", "w") as f:
        json.dump(config_save, f, indent=2)
    logger.info(f"Saved full config to {exp.output_dir / 'config.json'}")


def _run_training_loop(
    args: argparse.Namespace,
    sae: TopKSAE,
    trainer: TopKSAETrainer,
    train_act_loader,
    val_act_loader: Optional[DataLoader],
    exp: "ExperimentPaths",
    use_webdataset: bool,
    steps_per_epoch: int,
    gpu_monitor: GPUMonitor,
) -> Tuple[list, float, float]:
    """Epoch loop with checkpointing, validation, and intra-epoch saves."""
    logger.info(f"Starting training for {args.num_epochs} epochs...")
    training_start = time.perf_counter()
    best_val_loss = float("inf")
    history: list = []

    # Automatic intra-epoch checkpointing when epochs are few and long
    intra_ckpt_steps: Optional[set] = None
    intra_ckpt_callback: Optional[Callable[[int, int], None]] = None
    if args.num_epochs < 3 and steps_per_epoch > 0:
        num_fractions = 4
        intra_ckpt_steps = {
            int(steps_per_epoch * (i + 1) / num_fractions)
            for i in range(num_fractions - 1)
        }
        logger.info(
            f"Intra-epoch checkpointing enabled (num_epochs={args.num_epochs} < 3): "
            f"checkpoints at steps {sorted(intra_ckpt_steps)} / {steps_per_epoch}"
        )

        def _make_checkpoint_callback(
            _sae: TopKSAE,
            _trainer: TopKSAETrainer,
            _val_loader: Optional[DataLoader],
            _output_dir: Path,
            _logger: logging.Logger,
        ) -> Callable[[int, int], None]:
            """Create checkpoint callback with proper closure over mutable best_val_loss."""
            def callback(step: int, total_steps: int) -> None:
                nonlocal best_val_loss
                pct = step / total_steps * 100
                _logger.info(f"  Intra-epoch checkpoint at step {step}/{total_steps} ({pct:.0f}%)")
                _sae.save(str(_output_dir / f"sae_step_{step:06d}.pt"))

                if _val_loader is not None:
                    val_metrics = eval_epoch(_trainer, _val_loader)
                    _logger.info(
                        f"    val_loss={val_metrics['loss']:.4f}, "
                        f"R2={val_metrics['explained_variance']:.4f}"
                    )
                    if val_metrics['loss'] < best_val_loss:
                        best_val_loss = val_metrics['loss']
                        _sae.save(str(_output_dir / "best_sae.pt"))
                        _logger.info(f"    -> New best model (val_loss={best_val_loss:.4f})!")
            return callback

        intra_ckpt_callback = _make_checkpoint_callback(
            sae, trainer, val_act_loader, exp.output_dir, logger,
        )

    for epoch in range(1, args.num_epochs + 1):
        train_metrics, epoch_stats_rec = train_epoch(
            trainer, train_act_loader, epoch, args.num_epochs, gpu_monitor,
            steps_per_epoch=steps_per_epoch if use_webdataset else None,
            checkpoint_callback=intra_ckpt_callback,
            checkpoint_steps=intra_ckpt_steps,
        )
        stats.record_epoch(epoch_stats_rec)

        if epoch % args.eval_every == 0 or epoch == args.num_epochs:
            if val_act_loader is not None:
                val_metrics = eval_epoch(trainer, val_act_loader)
            else:
                val_metrics = None

            logger.info(
                f"Epoch {epoch}/{args.num_epochs} Train: "
                f"loss={train_metrics['loss']:.4f} "
                f"mse={train_metrics.get('mse_loss', 0):.4f} "
                f"aux={train_metrics.get('ghost_loss', 0):.4f} "
                f"l0={train_metrics['l0']:.1f} "
                f"R\u00b2={train_metrics['explained_variance']:.4f} "
                f"cos={train_metrics['cos_sim']:.4f} "
                f"dead%={train_metrics['dead_pct']:.1f} "
                f"lr={train_metrics.get('lr', 0):.2e} "
                f"io%={epoch_stats_rec.io_wait_pct:.1f}"
            )
            if val_metrics is not None:
                logger.info(
                    f"  Val: loss={val_metrics['loss']:.4f} l0={val_metrics['l0']:.1f} "
                    f"R\u00b2={val_metrics['explained_variance']:.4f} cos={val_metrics['cos_sim']:.4f}"
                )

            compare_loss = val_metrics['loss'] if val_metrics else train_metrics['loss']
            if compare_loss < best_val_loss:
                best_val_loss = compare_loss
                sae.save(str(exp.output_dir / "best_sae.pt"))
                logger.info("  -> New best model saved!")
        else:
            val_metrics = None

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        if epoch % args.save_every == 0:
            sae.save(str(exp.output_dir / f"sae_epoch_{epoch:03d}.pt"))

    return history, best_val_loss, training_start


def _finalize_training(
    args: argparse.Namespace,
    exp: "ExperimentPaths",
    sae: TopKSAE,
    history: list,
    best_val_loss: float,
    training_start: float,
    train_meta: Dict[str, Any],
    val_meta: Dict[str, Any],
    gpu_monitor: GPUMonitor,
    steps_per_epoch: int,
) -> None:
    """Save final model, collect stats, print summary."""
    sae.save(str(exp.output_dir / "final_sae.pt"))

    with open(exp.output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    if torch.cuda.is_available():
        gpu_monitor.stop()
        monitor_stats = gpu_monitor.get_stats()
        stats.set_gpu_stats(monitor_stats)
        ram_keys = ("job_ram_peak_gb", "main_process_ram_peak_gb", "system_ram_peak_gb", "system_ram_total_gb")
        stats.set_ram_stats({k: monitor_stats[k] for k in ram_keys if k in monitor_stats})

    training_time = time.perf_counter() - training_start
    stats.set_training_stats(
        num_epochs=args.num_epochs,
        batches_per_epoch=steps_per_epoch,
        final_loss=best_val_loss,
    )

    stats.set_cache_stats(
        train_batches=len(train_meta.get('files', [])) if isinstance(train_meta.get('files'), list) else train_meta.get('num_files', 0),
        train_tokens=train_meta['tokens_used'],
        train_bytes=train_meta.get('total_tokens', 0) * train_meta.get('hidden_dim', 0) * BYTES_PER_FLOAT16,
        val_batches=len(val_meta.get('files', [])) if isinstance(val_meta.get('files'), list) else val_meta.get('num_files', 0),
        val_tokens=val_meta['tokens_used'],
        val_bytes=val_meta.get('total_tokens', 0) * val_meta.get('hidden_dim', 0) * BYTES_PER_FLOAT16,
        dtype=args.cache_dtype,
    )

    stats.print_summary()
    stats.save_json(exp.output_dir / "timing_stats.json")

    logger.info("Training complete!")
    logger.info(f"  Best val loss: {best_val_loss:.4f}")
    logger.info(f"  Training time: {format_duration(training_time)}")
    logger.info(f"  Outputs saved to: {exp.output_dir}")


def _cache_with_resume(
    args: argparse.Namespace,
    model: nn.Module,
    exp: "ExperimentPaths",
    base_data_source: str,
    cache_dtype: torch.dtype,
    device: torch.device,
) -> Tuple[List[Path], List[Path]]:
    """Cache activations for covla/nuplan with zero-cost resume via Subset."""
    if base_data_source == "covla":
        if args.covla_videos_dir is None:
            raise ValueError("covla_videos_dir required for CoVLA data source")
        train_dataset, val_dataset = create_datasets_covla(
            videos_dir=args.covla_videos_dir,
            captions_dir=args.covla_captions_dir,
            input_size=tuple(args.input_size),
            num_frames=args.num_frames,
            stored_data_frame_rate=args.stored_frame_rate,
            frame_rate=args.frame_rate,
            num_videos=args.num_videos,
        )
    else:
        if args.nuplan_data_dir is None:
            raise ValueError("nuplan_data_dir required for NuPlan data source")
        train_dataset, val_dataset = create_datasets_nuplan(
            data_dir=args.nuplan_data_dir,
            input_size=tuple(args.input_size),
            num_frames=args.num_frames,
            stored_data_frame_rate=args.stored_frame_rate,
            frame_rate=args.frame_rate,
            num_videos=args.num_videos,
        )

    train_resume = get_cache_resume_info(
        cache_dir=exp.train_cache_dir, batch_size=args.batch_size,
        total_samples=len(train_dataset), rebuild=args.rebuild_cache,
    )
    val_resume = get_cache_resume_info(
        cache_dir=exp.val_cache_dir, batch_size=args.batch_size,
        total_samples=len(val_dataset), rebuild=args.rebuild_cache,
    )

    train_loader = create_dataloader_with_offset(
        dataset=train_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, device=device,
        start_sample_idx=train_resume.start_sample_idx,
    )
    val_loader = create_dataloader_with_offset(
        dataset=val_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, device=device,
        start_sample_idx=val_resume.start_sample_idx,
    )

    if args.train_only:
        if not train_resume.is_complete or not val_resume.is_complete:
            raise RuntimeError(
                "--train_only specified but cache is incomplete. "
                "Run caching first or remove --train_only flag."
            )
        return train_resume.valid_files, val_resume.valid_files

    data_dir = args.covla_videos_dir if base_data_source == "covla" else args.nuplan_data_dir
    cache_config = CacheConfig(
        layer_idx=args.layer, dtype=cache_dtype, dtype_name=args.cache_dtype,
        t_noise=args.t_noise, frame_rate=args.frame_rate,
        seed=args.cache_seed, orbis_exp_dir=str(exp.exp_dir),
        data_source=args.data_source, data_dir=data_dir,
        num_videos=args.num_videos, num_frames=args.num_frames,
        val_split=0.1, input_size=tuple(args.input_size),
        stored_frame_rate=args.stored_frame_rate,
    )

    try:
        if not train_resume.is_complete:
            logger.info("Caching train activations...")
            train_files = prepare_activation_cache(
                model=model, dataloader=train_loader, cache_dir=exp.train_cache_dir,
                config=cache_config, resume=train_resume, device=device,
            )
        else:
            train_files = train_resume.valid_files

        if not val_resume.is_complete:
            logger.info("Caching validation activations...")
            val_files = prepare_activation_cache(
                model=model, dataloader=val_loader, cache_dir=exp.val_cache_dir,
                config=cache_config, resume=val_resume, device=device,
            )
        else:
            val_files = val_resume.valid_files

        if args.cache_only:
            logger.info("--cache_only: Caching complete, exiting")
            sys.exit(0)
    except Exception as e:
        logger.error(f"Caching FAILED: {e}")
        sys.exit(1)

    return train_files, val_files


def _cache_simple(
    args: argparse.Namespace,
    model: nn.Module,
    exp: "ExperimentPaths",
    cache_dtype: torch.dtype,
    device: torch.device,
) -> Tuple[List[Path], List[Path]]:
    """Cache activations for legacy sources (cityscapes, hdf5, image_paths)."""
    train_loader, val_loader = create_dataloaders(
        data_source=args.data_source, data_path=args.data_path,
        input_size=tuple(args.input_size), batch_size=args.batch_size,
        num_workers=args.num_workers, device=device,
        hdf5_paths_file=args.hdf5_paths_file, val_json=args.val_json,
        num_frames=args.num_frames, stored_data_frame_rate=args.stored_frame_rate,
        frame_rate=args.frame_rate, image_paths=args.image_paths,
        covla_videos_dir=args.covla_videos_dir,
        covla_captions_dir=args.covla_captions_dir,
        nuplan_data_dir=args.nuplan_data_dir,
        num_videos=args.num_videos,
    )

    train_resume = get_cache_resume_info(
        cache_dir=exp.train_cache_dir, batch_size=args.batch_size,
        total_samples=len(train_loader.dataset), rebuild=args.rebuild_cache,
    )
    val_resume = get_cache_resume_info(
        cache_dir=exp.val_cache_dir, batch_size=args.batch_size,
        total_samples=len(val_loader.dataset), rebuild=args.rebuild_cache,
    )

    if args.train_only:
        if not train_resume.is_complete or not val_resume.is_complete:
            raise RuntimeError(
                "--train_only specified but cache is incomplete. "
                "Run caching first or remove --train_only flag."
            )
        return train_resume.valid_files, val_resume.valid_files

    cache_config = CacheConfig(
        layer_idx=args.layer, dtype=cache_dtype, dtype_name=args.cache_dtype,
        t_noise=args.t_noise, frame_rate=args.frame_rate,
        seed=args.cache_seed, orbis_exp_dir=str(exp.exp_dir),
        data_source=args.data_source, data_dir=args.data_path,
        num_frames=args.num_frames, input_size=tuple(args.input_size),
        stored_frame_rate=args.stored_frame_rate,
    )

    try:
        if not train_resume.is_complete:
            logger.info("Caching train activations...")
            train_files = prepare_activation_cache(
                model=model, dataloader=train_loader, cache_dir=exp.train_cache_dir,
                config=cache_config, resume=CacheResumeInfo.fresh(), device=device,
            )
        else:
            train_files = train_resume.valid_files

        if not val_resume.is_complete:
            logger.info("Caching validation activations...")
            val_files = prepare_activation_cache(
                model=model, dataloader=val_loader, cache_dir=exp.val_cache_dir,
                config=cache_config, resume=CacheResumeInfo.fresh(), device=device,
            )
        else:
            val_files = val_resume.valid_files

        if args.cache_only:
            logger.info("--cache_only: Caching complete, exiting")
            sys.exit(0)
    except Exception as e:
        logger.error(f"Caching FAILED: {e}")
        sys.exit(1)

    return train_files, val_files


def _prepare_and_cache_activations(
    args: argparse.Namespace,
    model: Optional[nn.Module],
    exp: "ExperimentPaths",
    base_data_source: str,
    use_webdataset: bool,
    cache_dtype: torch.dtype,
    device: torch.device,
) -> Tuple[Optional[List[Path]], Optional[List[Path]]]:
    """Dispatch data loading and activation caching to the appropriate flow.

    Returns (train_cache_files, val_cache_files), or (None, None) for WebDataset.
    """
    logger.info(f"Loading data from {args.data_source} source")

    if use_webdataset:
        logger.info(f"  Base source: {base_data_source}, format: WebDataset shards")
        return None, None

    if base_data_source in ("covla", "nuplan"):
        return _cache_with_resume(args, model, exp, base_data_source, cache_dtype, device)

    return _cache_simple(args, model, exp, cache_dtype, device)


def _create_webdataset_loaders(
    args: argparse.Namespace,
    exp: "ExperimentPaths",
    hidden_size: int,
    sae_batch_size: int,
) -> Tuple[DataLoader, Optional[DataLoader], Dict[str, Any], Dict[str, Any]]:
    """Create WebDataset dataloaders for SAE training from sharded tar archives."""
    wds_dir = exp.train_cache_dir
    if not wds_dir.exists():
        raise FileNotFoundError(
            f"WebDataset directory not found: {wds_dir}\n"
            f"Run the conversion script first: python sae/scripts/convert_to_webdataset.py"
        )

    available_shards = sorted(wds_dir.glob(f"layer_{args.layer}-train-*.tar"))
    total_shards = len(available_shards)
    if total_shards == 0:
        raise FileNotFoundError(f"No shards found in {wds_dir}")

    num_shards = args.num_shards or total_shards
    if num_shards > total_shards:
        logger.warning(f"Requested {num_shards} shards but only {total_shards} available. Using all.")
        num_shards = total_shards

    train_shards = [str(p) for p in available_shards[:num_shards]]
    logger.info(f"WebDataset: {wds_dir}, {num_shards}/{total_shards} shards")

    # 4 clips x 6 frames x 576 spatial = 13824 correlated tokens per shard file
    CACHED_BATCH_SIZE = 13824
    logger.info(f"Shuffle buffer: {args.shuffle_buffer:,} tokens")

    train_act_loader, train_meta = create_webdataset_dataloader(
        shard_pattern=train_shards,
        batch_size=sae_batch_size,
        shuffle_buffer=args.shuffle_buffer,
        num_workers=DEFAULT_NUM_WORKERS,
        seed=args.seed,
        cached_batch_size=CACHED_BATCH_SIZE,
    )

    val_act_loader = None
    val_meta: Dict[str, Any] = {"tokens_used": 0, "total_tokens": 0}
    if exp.val_cache_dir.exists():
        val_shards = sorted(exp.val_cache_dir.glob(f"layer_{args.layer}-val-*.tar"))
        if len(val_shards) > 0:
            logger.info(f"Using {len(val_shards)} WebDataset shards for validation")
            val_act_loader, val_meta = create_webdataset_dataloader(
                shard_pattern=[str(p) for p in val_shards],
                batch_size=sae_batch_size,
                shuffle_buffer=0,
                num_workers=0,
                seed=args.seed,
                cached_batch_size=CACHED_BATCH_SIZE,
            )
            meta_path = exp.val_cache_dir / "_meta.json"
            if meta_path.exists():
                try:
                    with open(meta_path, "r") as f:
                        meta_json = json.load(f)
                    if isinstance(meta_json, dict) and "total_tokens" in meta_json:
                        val_meta["tokens_used"] = int(meta_json["total_tokens"])
                        val_meta["total_tokens"] = int(meta_json["total_tokens"])
                except Exception:
                    pass
        elif list(exp.val_cache_dir.glob("batch_*.pt")):
            logger.info("Using individual .pt files for validation")
            val_act_loader, val_meta = create_activation_dataloader(
                exp.val_cache_dir, batch_size=sae_batch_size,
                shuffle=False, num_workers=0, in_memory=True,
                max_tokens=args.max_tokens,
            )

    if val_act_loader is None:
        logger.warning("No validation cache found. Validation will be limited.")

    val_meta["hidden_dim"] = hidden_size

    estimated_batches_per_shard = 471
    estimated_tokens = num_shards * estimated_batches_per_shard * CACHED_BATCH_SIZE
    train_meta["tokens_used"] = estimated_tokens
    train_meta["total_tokens"] = estimated_tokens
    train_meta["estimated"] = True
    train_meta["hidden_dim"] = hidden_size
    train_meta["num_files"] = len(train_shards)

    return train_act_loader, val_act_loader, train_meta, val_meta


def _create_sae_dataloaders(
    args: argparse.Namespace,
    exp: "ExperimentPaths",
    hidden_size: int,
    use_webdataset: bool,
    train_cache_files: Optional[List[Path]],
    val_cache_files: Optional[List[Path]],
) -> Tuple[DataLoader, Optional[DataLoader], Dict[str, Any], Dict[str, Any], int]:
    """Create activation dataloaders for SAE training, dispatching by storage mode."""
    sae_batch_size = resolve_sae_batch_size(
        batch_size_arg=args.sae_batch_size, layer=args.layer,
        expansion=args.expansion_factor, k=args.k,
    )
    logger.info(f"SAE batch size: {sae_batch_size:,} tokens")

    if use_webdataset:
        train_act_loader, val_act_loader, train_meta, val_meta = (
            _create_webdataset_loaders(args, exp, hidden_size, sae_batch_size)
        )
    else:
        in_memory = not args.streaming
        logger.info(f"Creating activation dataloaders (streaming={args.streaming})...")
        train_act_loader, train_meta = create_activation_dataloader(
            exp.train_cache_dir, batch_size=sae_batch_size, shuffle=True,
            num_workers=DEFAULT_NUM_WORKERS if args.streaming else 0,
            in_memory=in_memory, max_tokens=args.max_tokens,
        )
        val_act_loader, val_meta = create_activation_dataloader(
            exp.val_cache_dir, batch_size=sae_batch_size, shuffle=False,
            num_workers=DEFAULT_NUM_WORKERS if args.streaming else 0,
            in_memory=in_memory, max_tokens=args.max_tokens,
        )

    logger.info(f"Train tokens: {train_meta['tokens_used']:,}" + (" (estimated)" if train_meta.get("estimated") else ""))
    logger.info(f"Val tokens: {val_meta['tokens_used']:,}")

    steps_per_epoch = train_meta['tokens_used'] // sae_batch_size
    logger.info(f"Steps per epoch: {steps_per_epoch:,}")

    return train_act_loader, val_act_loader, train_meta, val_meta, steps_per_epoch


def _start_monitoring(
    args: argparse.Namespace,
    exp: "ExperimentPaths",
) -> GPUMonitor:
    """Initialize timing stats and start GPU monitoring."""
    stats.set_run_info(run_name=exp.barcode, layer=args.layer, data_source=args.data_source)

    cache_path_str = str(exp.cache_dir)
    if "/tmp" in cache_path_str or "/scratch" in cache_path_str:
        data_source_type = "tmpdir"
    elif "/data/lmb" in cache_path_str:
        data_source_type = "NVMe_SSD"
    else:
        data_source_type = "NFS"
    stats.set_data_source(source_type=data_source_type, path=cache_path_str)

    gpu_monitor = GPUMonitor(sample_interval_s=2.0, device_id=0 if torch.cuda.is_available() else -1)
    if torch.cuda.is_available():
        gpu_monitor.start()
    return gpu_monitor


def main(args: argparse.Namespace, unknown_args: Sequence[str] = ()):
    """Main training function."""
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    exp, model, hidden_size, use_webdataset, base_data_source, cache_dtype = (
        _setup_experiment(args, device)
    )

    train_cache_files, val_cache_files = _prepare_and_cache_activations(
        args, model, exp, base_data_source, use_webdataset, cache_dtype, device,
    )

    train_act_loader, val_act_loader, train_meta, val_meta, steps_per_epoch = (
        _create_sae_dataloaders(
            args, exp, hidden_size, use_webdataset, train_cache_files, val_cache_files,
        )
    )

    total_training_steps = steps_per_epoch * args.num_epochs
    sae, trainer, sae_config = _create_sae_and_trainer(
        args, hidden_size, device, total_training_steps=total_training_steps
    )
    _save_experiment_config(args, exp, sae_config, train_meta, val_meta)
    gpu_monitor = _start_monitoring(args, exp)

    # Initialize b_dec from training data before the training loop
    trainer.initialize_b_dec(train_act_loader)

    history, best_val_loss, training_start = _run_training_loop(
        args, sae, trainer, train_act_loader, val_act_loader,
        exp, use_webdataset, steps_per_epoch, gpu_monitor,
    )

    _finalize_training(
        args, exp, sae, history, best_val_loss, training_start,
        train_meta, val_meta, gpu_monitor, steps_per_epoch,
    )


def parse_args(argv=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Top-K SAE on Orbis world model activations"
    )

    # Required paths
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Path to Orbis experiment directory")

    # Data source selection
    parser.add_argument(
        "--data_source",
        type=str,
        default="hdf5",
        choices=["cityscapes", "hdf5", "image_paths", "covla", "nuplan", "covla-webdataset", "nuplan-webdataset"],
        help="Data source: 'hdf5' (native Orbis data), 'cityscapes', 'image_paths', 'covla', or 'nuplan'. "
             "Use '-webdataset' suffix (e.g., 'nuplan-webdataset') for WebDataset shards (faster on NFS).",
    )

    # Cityscapes options (used when data_source=cityscapes)
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to Cityscapes dataset (required if data_source=cityscapes)",
    )

    # HDF5 options (used when data_source=hdf5) - native Orbis training data
    parser.add_argument(
        "--hdf5_paths_file",
        type=str,
        default=None,
        help="Path to text file listing HDF5 file paths (required if data_source=hdf5)",
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
        help="Frame rate of stored data (5 for HDF5, 20 for CoVLA)",
    )

    # Image paths options (used when data_source=image_paths) - for example frames
    parser.add_argument(
        "--image_paths",
        type=str,
        nargs="+",
        default=None,
        help="List of image file paths (required if data_source=image_paths)",
    )

    # CoVLA options (used when data_source=covla)
    parser.add_argument(
        "--covla_videos_dir",
        type=str,
        default=None,
        help="Directory containing CoVLA .mp4 videos (required if data_source=covla)",
    )
    parser.add_argument(
        "--covla_captions_dir",
        type=str,
        default=None,
        help="Directory containing CoVLA caption JSONL files (optional if data_source=covla)",
    )

    # NuPlan options (used when data_source=nuplan)
    parser.add_argument(
        "--nuplan_data_dir",
        type=str,
        default=None,
        help="Directory containing NuPlan video folders with frames.h5 (required if data_source=nuplan)",
    )

    # Shared video dataset options
    parser.add_argument(
        "--num_videos",
        type=int,
        default=None,
        help="Number of videos to use from dataset (default: all)",
    )

    # Orbis model paths (relative to exp_dir)
    parser.add_argument("--ckpt", type=str, default="checkpoints/last.ckpt",
                        help="Checkpoint path relative to exp_dir")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Config path relative to exp_dir")

    # SAE architecture
    parser.add_argument("--k", type=int, default=64,
                        help="Number of active features (top-k)")
    parser.add_argument("--expansion_factor", type=int, default=16,
                        help="SAE expansion factor")

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
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for data loading")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")

    # Caching (cache path is auto-computed: logs_sae/sae_cache/{dataset}/{model}/layer_{layer}/)
    parser.add_argument("--cache_dtype", type=str, default="float16",
                        help="Storage dtype for cached activations")
    parser.add_argument("--rebuild_cache", action="store_true",
                        help="Force rebuild cache")
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming mode (load from disk) instead of loading all activations into RAM. "
                             "Required for large datasets (>50M tokens) that don't fit in memory.")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Maximum tokens to use from cache (allows using subset without rebuilding). "
                             "If None, uses all cached tokens.")
    parser.add_argument("--cache_only", action="store_true",
                        help="Run caching phase only, then exit. Used by orchestrator for separate cache jobs.")
    parser.add_argument("--train_only", action="store_true",
                        help="Skip caching, assume cache exists. Used by orchestrator for separate train jobs.")
    parser.add_argument("--cache_seed", type=int, default=42,
                        help="Random seed for caching noise generation (for reproducibility)")
    
    # WebDataset options (used when data_source ends with '-webdataset')
    parser.add_argument("--num_shards", type=int, default=None,
                        help="Number of shards to use for WebDataset (for partial training). "
                             "Example: --num_shards 59 for half the dataset.")
    parser.add_argument("--shuffle_buffer", type=int, default=500000,
                        help="Shuffle buffer size for WebDataset (default: 500000 tokens). "
                             "Larger = better IID mixing but more memory. 500k uses ~1.5GB.")

    # Training
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping (0 to disable)")
    parser.add_argument("--lr_warmup_steps", type=int, default=500,
                        help="Number of linear LR warmup steps")
    parser.add_argument("--lr_end_factor", type=float, default=0.1,
                        help="End LR as fraction of base LR for cosine schedule (0.1 = lr/10)")

    # SAE-specific training options
    parser.add_argument("--aux_loss_coefficient", type=float, default=1.0,
                        help="Coefficient for auxiliary TopK loss (dead-feature revival). "
                             "Set to 0 to disable. Default: 1.0 (from Gao et al.)")
    parser.add_argument("--dead_feature_window", type=int, default=1000,
                        help="Steps without firing before a feature is considered dead")
    parser.add_argument("--decoder_init_norm", type=float, default=0.1,
                        help="Normalize decoder rows to this norm at init (0.1 from Anthropic). "
                             "Set to 0 to disable.")
    parser.add_argument("--b_dec_init", type=str, default="geometric_median",
                        choices=["geometric_median", "mean", "zeros"],
                        help="Method for initializing decoder bias")
    parser.add_argument("--normalize_activations", type=str, default="none",
                        choices=["none", "layer_norm"],
                        help="Input activation normalization before encoding")

    parser.add_argument("--sae_batch_size", type=str, default="4096",
                        help="SAE batch size in tokens (default: 4096). "
                             "Use 'auto' to read from calibration resource file.")
    parser.add_argument("--eval_every", type=int, default=5,
                        help="Evaluate every N epochs")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    

    # Output
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: logs_sae/runs/{dataset}/{model}/layer_{N}/{barcode}/)")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Run name/barcode (default: auto-generated with timestamp)")
    parser.add_argument("--exp_name", type=str, default="topk_sae",
                        help="Experiment name prefix (unused if run_name provided)")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for faster execution (default mode for Turing stability)")

    return parser.parse_known_args(argv)


if __name__ == "__main__":
    args, unknown = parse_args()
    main(args, unknown)
