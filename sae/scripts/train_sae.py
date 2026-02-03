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
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from omegaconf import OmegaConf
from tqdm import tqdm

# Add project root to path (scripts are in sae/scripts/, so go up 3 levels for project, 2 for orbis)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ORBIS_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ORBIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ORBIS_ROOT))

from util import instantiate_from_config
from pytorch_lightning import seed_everything

# Local imports (use absolute imports for script execution)
from sae.topk_sae import TopKSAE, TopKSAEConfig, TopKSAETrainer
from sae.caching import (
    CacheResumeInfo,
    get_cache_resume_info,
    prepare_activation_cache,
    create_activation_dataloader,
    resolve_cache_dtype,
)
from sae.metrics import (
    compute_normalized_loss_recovered,
    compute_dictionary_coverage,
)

# Import datasets
from semantic_stage2 import CityScapes
from data.custom_multiframe import (
    MultiHDF5DatasetMultiFrameRandomizeFrameRate,
    MultiHDF5DatasetMultiFrameFromJSONFrameRateWrapper,
)
from data.multiframe_val import MultiFrameFromPaths
from data.covla.covla_dataset import CoVLAOrbisMultiFrame
from data.nuplan.nuplan_dataset import NuPlanOrbisMultiFrame


def load_orbis_model(
    config_path: str,
    ckpt_path: str,
    device: torch.device,
) -> nn.Module:
    """
    Load pre-trained Orbis world model.
    
    Args:
        config_path: Path to model config.yaml
        ckpt_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode with frozen parameters
    """
    print(f"[model] Loading config from {config_path}")
    cfg_model = OmegaConf.load(config_path)

    print(f"[model] Instantiating model...")
    model = instantiate_from_config(cfg_model.model)

    print(f"[model] Loading checkpoint from {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=False)

    # Freeze model and move to device
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print(f"[model] Model loaded and frozen")
    return model


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
    
    print(f"[data] Image paths dataset: {len(dataset)} samples, {len(image_paths)} frames")
    
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

    print(f"[data] HDF5 train dataset: {len(dataset_train)} samples")
    print(f"[data] HDF5 val dataset: {len(dataset_val)} samples")

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

    print(f"[data] CoVLA Full: {total_videos} videos, {len(full_dataset)} total clips")
    print(f"[data] Train Split: {num_train_videos} videos ({len(dataset_train)} clips)")
    print(f"[data] Val Split:   {num_val_videos} videos ({len(dataset_val)} clips)")

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

    print(f"[data] NuPlan Full: {total_videos} videos, {len(full_dataset)} total clips")
    print(f"[data] Train Split: {num_train_videos} videos ({len(dataset_train)} clips)")
    print(f"[data] Val Split:   {num_val_videos} videos ({len(dataset_val)} clips)")

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
        print(f"[data] Applied resume subset: {len(dataset)} samples remaining")
    
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
    """
    Create dataloaders based on data source type.

    Args:
        data_source: "cityscapes", "hdf5", "image_paths", "covla", or "nuplan"
        data_path: Path to Cityscapes root (if cityscapes) or ignored otherwise
        input_size: Image size (H, W)
        batch_size: Batch size
        num_workers: Number of workers
        device: Device
        hdf5_paths_file: Path to HDF5 paths file (required if hdf5)
        val_json: Path to validation JSON (required if hdf5)
        num_frames: Number of frames per sample (for hdf5/image_paths/covla/nuplan)
        stored_data_frame_rate: Stored frame rate (5 for hdf5, 20 for covla, 10 for nuplan)
        frame_rate: Target frame rate (for hdf5/covla/nuplan)
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
            print(f"[warning] CoVLA videos are 20 FPS, but stored_frame_rate={stored_data_frame_rate}. Forcing to 20.")
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
            print(f"[warning] NuPlan videos are 10 FPS, but stored_frame_rate={stored_data_frame_rate}. Forcing to 10.")
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
) -> Dict[str, float]:
    """Train for one epoch with full Phase 1 metric logging."""
    
    # All Phase 1 metrics to track
    metrics_sum = {
        "loss": 0.0,
        "l0": 0.0,
        "cos_sim": 0.0,
        "rel_error": 0.0,
        "explained_variance": 0.0,
        "activation_density": 0.0,
        "l1_norm": 0.0,
        "dead_pct": 0.0,
    }
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [train]", 
                 file=sys.stdout, mininterval=360.0)  # Write to stdout, update every 30s
    for batch in pbar:
        metrics = trainer.train_step(batch)
        
        for k in metrics_sum.keys():
            if k in metrics:
                metrics_sum[k] += metrics[k]
        num_batches += 1
        
        # Display key metrics in progress bar: loss, l0, dead%
        pbar.set_postfix({
            "loss": f"{metrics['loss']:.4f}",
            "l0": f"{metrics['l0']:.1f}",
            "dead%": f"{metrics['dead_pct']:.1f}",
        })
    
    # Average metrics
    return {k: v / num_batches for k, v in metrics_sum.items()}


@torch.no_grad()
def eval_epoch(
    trainer: TopKSAETrainer,
    dataloader: DataLoader,
) -> Dict[str, float]:
    """Evaluate on validation set with full Phase 1 metric logging."""
    
    # All Phase 1 metrics to track
    metrics_sum = {
        "loss": 0.0,
        "l0": 0.0,
        "cos_sim": 0.0,
        "rel_error": 0.0,
        "explained_variance": 0.0,
        "activation_density": 0.0,
        "l1_norm": 0.0,
        "dead_pct": 0.0,
    }
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Evaluating", file=sys.stdout, mininterval=360.0):
        metrics = trainer.eval_step(batch)
        
        for k in metrics_sum.keys():
            if k in metrics:
                metrics_sum[k] += metrics[k]
        num_batches += 1
    
    return {k: v / num_batches for k, v in metrics_sum.items()}


@torch.no_grad()
def run_phase2_evaluation(
    orbis_model: nn.Module,
    sae: TopKSAE,
    image_dataloader: DataLoader,
    activation_dataloader: DataLoader,
    layer_idx: int,
    device: torch.device,
    t_noise: float = 0.0,
    frame_rate: int = 5,
    max_batches: int = 50,
) -> Dict[str, float]:
    """
    Run Phase 2 evaluation metrics: Normalized Loss Recovered and Dictionary Coverage.
    
    This is expensive as it requires running the full Orbis model.
    
    Args:
        orbis_model: Pre-trained Orbis world model
        sae: Trained SAE model
        image_dataloader: DataLoader for images (for NLR)
        activation_dataloader: DataLoader for cached activations (for coverage)
        layer_idx: Layer to intervene on
        device: Device to run on
        t_noise: Noise timestep
        frame_rate: Frame rate conditioning
        max_batches: Max batches to limit compute
        
    Returns:
        Dictionary with Phase 2 metrics
    """
    print("\n" + "="*60)
    print("[Phase 2] Running evaluation metrics...")
    print("="*60)
    
    results = {}
    
    # 1. Normalized Loss Recovered
    print("\n[Phase 2] Computing Normalized Loss Recovered...")
    try:
        nlr_results = compute_normalized_loss_recovered(
            model=orbis_model,
            sae=sae,
            dataloader=image_dataloader,
            layer_idx=layer_idx,
            device=device,
            t_noise=t_noise,
            frame_rate=frame_rate,
            max_batches=max_batches,
        )
        results.update({f"phase2_{k}": v for k, v in nlr_results.items()})
        print(f"  L_base: {nlr_results['L_base']:.6f}")
        print(f"  L_sae:  {nlr_results['L_sae']:.6f}")
        print(f"  L_zero: {nlr_results['L_zero']:.6f}")
        print(f"  Normalized Loss Recovered: {nlr_results['normalized_loss_recovered']:.4f}")
    except Exception as e:
        print(f"  [WARNING] NLR computation failed: {e}")
        results["phase2_nlr_error"] = str(e)
    
    # 2. Dictionary Coverage
    print("\n[Phase 2] Computing Dictionary Coverage...")
    try:
        coverage_results = compute_dictionary_coverage(
            sae=sae,
            dataloader=activation_dataloader,
            device=device,
            max_batches=max_batches * 2,  # Can process more batches for coverage
        )
        results.update({f"phase2_{k}": v for k, v in coverage_results.items()})
        print(f"  Dictionary Coverage: {coverage_results['dictionary_coverage_pct']:.2f}%")
        print(f"  Active Features: {coverage_results['num_active_features']} / {coverage_results['total_features']}")
    except Exception as e:
        print(f"  [WARNING] Coverage computation failed: {e}")
        results["phase2_coverage_error"] = str(e)
    
    print("="*60 + "\n")
    
    return results


def main(args: argparse.Namespace, unknown_args: Sequence[str] = ()):
    """Main training function."""

    # Set seed for reproducibility
    if args.seed > 0:
        seed_everything(args.seed)

    device = torch.device(args.device)
    print(f"[setup] Using device: {device}")

    # Setup paths
    exp_dir = Path(args.exp_dir)
    config_path = exp_dir / args.config
    ckpt_path = exp_dir / args.ckpt
    
    # Orbis root is 2 levels up from scripts/ (sae/scripts/ -> orbis/)
    orbis_root = Path(__file__).resolve().parents[2]
    
    # Cache path: logs_sae/sae_cache/{dataset}/{model}/{layer}/
    cache_base = orbis_root / "logs_sae" / "sae_cache"
    cache_dir = cache_base / args.data_source / exp_dir.name / f"layer_{args.layer}"
    
    # Setup output directory with timestamp for unique runs
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use provided run_name or generate barcode
    if args.run_name:
        barcode = args.run_name
    else:
        raise ValueError("run_name is required")
    
    if args.output_dir is not None:
        # User specified custom output directory
        output_dir = Path(args.output_dir)
    else:
        # Default: logs_sae/runs/{dataset}/{model}/{layer}/{barcode}/
        runs_base = orbis_root / "logs_sae" / "runs"
        output_dir = runs_base / args.data_source / exp_dir.name / f"layer_{args.layer}" / barcode
    
    # Print barcode for SLURM script to capture
    print(f"[barcode] {barcode}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] Experiment directory: {exp_dir}")
    print(f"[setup] Cache directory: {cache_dir}")
    print(f"[setup] Output directory: {output_dir}")

    # Load Orbis model
    model = load_orbis_model(str(config_path), str(ckpt_path), device)

    # Get hidden dimension from model
    hidden_size = model.vit.blocks[0].norm1.normalized_shape[0]
    print(f"[setup] ST-Transformer hidden size: {hidden_size}")

    # Cache directories
    train_cache_dir = cache_dir / "train"
    val_cache_dir = cache_dir / "val"
    cache_dtype = resolve_cache_dtype(args.cache_dtype)
    
    # ========================================================================
    # ZERO-COST RESUME FLOW for CoVLA and NuPlan
    # For these sources, we:
    # 1. Create datasets first (not dataloaders)
    # 2. Check cache status to determine resume point
    # 3. Create dataloaders with Subset for remaining samples
    # 4. Cache only the remaining activations
    # ========================================================================
    
    print(f"[data] Loading data from {args.data_source} source")
    
    if args.data_source in ("covla", "nuplan"):
        # === Step 1: Create datasets (not dataloaders) ===
        if args.data_source == "covla":
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
        else:  # nuplan
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
        
        # === Step 2: Check cache status for zero-cost resume ===
        print(f"\n[cache] Checking train cache status...")
        train_resume = get_cache_resume_info(
            cache_dir=train_cache_dir,
            batch_size=args.batch_size,
            total_samples=len(train_dataset),
            rebuild=args.rebuild_cache,
        )
        
        print(f"\n[cache] Checking validation cache status...")
        val_resume = get_cache_resume_info(
            cache_dir=val_cache_dir,
            batch_size=args.batch_size,
            total_samples=len(val_dataset),
            rebuild=args.rebuild_cache,
        )
        
        # === Step 3: Create dataloaders with Subset for remaining samples ===
        train_loader = create_dataloader_with_offset(
            dataset=train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            start_sample_idx=train_resume.start_sample_idx,
        )
        
        val_loader = create_dataloader_with_offset(
            dataset=val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            start_sample_idx=val_resume.start_sample_idx,
        )
        
        # === Step 4: Cache activations (zero-cost resume) ===
        # Skip caching if --train_only is set (assumes cache exists)
        if args.train_only:
            print(f"\n[cache] --train_only: Skipping caching, assuming cache exists")
            train_cache_files = train_resume.valid_files
            val_cache_files = val_resume.valid_files
            if not train_resume.is_complete or not val_resume.is_complete:
                raise RuntimeError(
                    "--train_only specified but cache is incomplete. "
                    "Run caching first or remove --train_only flag."
                )
        else:
            # Determine data_dir for metadata
            data_dir = args.covla_videos_dir if args.data_source == "covla" else args.nuplan_data_dir
            
            try:
                if not train_resume.is_complete:
                    print(f"\n[cache] Caching train activations...")
                    train_cache_files = prepare_activation_cache(
                        model=model,
                        dataloader=train_loader,
                        cache_dir=train_cache_dir,
                        layer_idx=args.layer,
                        device=device,
                        dtype=cache_dtype,
                        dtype_name=args.cache_dtype,
                        t_noise=args.t_noise,
                        frame_rate=args.frame_rate,
                        start_batch_idx=train_resume.start_batch_idx,
                        existing_tokens=train_resume.total_tokens,
                        existing_files=train_resume.valid_files,
                        existing_hidden_dim=train_resume.hidden_dim,
                        # Reproducibility metadata
                        seed=args.cache_seed,
                        orbis_exp_dir=str(exp_dir),
                        data_source=args.data_source,
                        data_dir=data_dir,
                        num_videos=args.num_videos,
                        num_frames=args.num_frames,
                        val_split=0.1,  # Hardcoded in create_datasets_*
                        input_size=tuple(args.input_size),
                        stored_frame_rate=args.stored_frame_rate,
                    )
                else:
                    print(f"\n[cache] Train cache complete, skipping...")
                    train_cache_files = train_resume.valid_files
                
                if not val_resume.is_complete:
                    print(f"\n[cache] Caching validation activations...")
                    val_cache_files = prepare_activation_cache(
                        model=model,
                        dataloader=val_loader,
                        cache_dir=val_cache_dir,
                        layer_idx=args.layer,
                        device=device,
                        dtype=cache_dtype,
                        dtype_name=args.cache_dtype,
                        t_noise=args.t_noise,
                        frame_rate=args.frame_rate,
                        start_batch_idx=val_resume.start_batch_idx,
                        existing_tokens=val_resume.total_tokens,
                        existing_files=val_resume.valid_files,
                        existing_hidden_dim=val_resume.hidden_dim,
                        # Reproducibility metadata
                        seed=args.cache_seed,
                        orbis_exp_dir=str(exp_dir),
                        data_source=args.data_source,
                        data_dir=data_dir,
                        num_videos=args.num_videos,
                        num_frames=args.num_frames,
                        val_split=0.1,
                        input_size=tuple(args.input_size),
                        stored_frame_rate=args.stored_frame_rate,
                    )
                else:
                    print(f"\n[cache] Validation cache complete, skipping...")
                    val_cache_files = val_resume.valid_files
                
                # Exit after caching if --cache_only is set
                if args.cache_only:
                    print(f"\n[cache] --cache_only: Caching complete, exiting")
                    sys.exit(0)
                    
            except Exception as e:
                print(f"\n[cache] FAILED: {e}")
                sys.exit(1)  # Non-zero exit for SLURM afterok dependency
    
    else:
        # === LEGACY FLOW for other data sources (cityscapes, hdf5, image_paths) ===
        # These don't benefit from zero-cost resume as much, use simple flow
        train_loader, val_loader = create_dataloaders(
            data_source=args.data_source,
            data_path=args.data_path,
            input_size=tuple(args.input_size),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            hdf5_paths_file=args.hdf5_paths_file,
            val_json=args.val_json,
            num_frames=args.num_frames,
            stored_data_frame_rate=args.stored_frame_rate,
            frame_rate=args.frame_rate,
            image_paths=args.image_paths,
            covla_videos_dir=args.covla_videos_dir,
            covla_captions_dir=args.covla_captions_dir,
            nuplan_data_dir=args.nuplan_data_dir,
            num_videos=args.num_videos,
        )
        
        # Check cache for these sources (simpler flow without zero-cost resume)
        train_resume = get_cache_resume_info(
            cache_dir=train_cache_dir,
            batch_size=args.batch_size,
            total_samples=len(train_loader.dataset),
            rebuild=args.rebuild_cache,
        )
        val_resume = get_cache_resume_info(
            cache_dir=val_cache_dir,
            batch_size=args.batch_size,
            total_samples=len(val_loader.dataset),
            rebuild=args.rebuild_cache,
        )
        
        # Skip caching if --train_only is set (assumes cache exists)
        if args.train_only:
            print(f"\n[cache] --train_only: Skipping caching, assuming cache exists")
            train_cache_files = train_resume.valid_files
            val_cache_files = val_resume.valid_files
            if not train_resume.is_complete or not val_resume.is_complete:
                raise RuntimeError(
                    "--train_only specified but cache is incomplete. "
                    "Run caching first or remove --train_only flag."
                )
        else:
            try:
                # Cache activations (will skip batches via continue if resuming)
                if not train_resume.is_complete:
                    print(f"\n[cache] Caching train activations...")
                    train_cache_files = prepare_activation_cache(
                        model=model,
                        dataloader=train_loader,
                        cache_dir=train_cache_dir,
                        layer_idx=args.layer,
                        device=device,
                        dtype=cache_dtype,
                        dtype_name=args.cache_dtype,
                        t_noise=args.t_noise,
                        frame_rate=args.frame_rate,
                        start_batch_idx=0,  # Full dataloader, start from 0
                        existing_tokens=0,
                        existing_files=[],
                        # Reproducibility metadata
                        seed=args.cache_seed,
                        orbis_exp_dir=str(exp_dir),
                        data_source=args.data_source,
                        data_dir=args.data_path,
                        num_frames=args.num_frames,
                        input_size=tuple(args.input_size),
                        stored_frame_rate=args.stored_frame_rate,
                    )
                else:
                    train_cache_files = train_resume.valid_files
                
                if not val_resume.is_complete:
                    print(f"\n[cache] Caching validation activations...")
                    val_cache_files = prepare_activation_cache(
                        model=model,
                        dataloader=val_loader,
                        cache_dir=val_cache_dir,
                        layer_idx=args.layer,
                        device=device,
                        dtype=cache_dtype,
                        dtype_name=args.cache_dtype,
                        t_noise=args.t_noise,
                        frame_rate=args.frame_rate,
                        start_batch_idx=0,
                        existing_tokens=0,
                        existing_files=[],
                        # Reproducibility metadata
                        seed=args.cache_seed,
                        orbis_exp_dir=str(exp_dir),
                        data_source=args.data_source,
                        data_dir=args.data_path,
                        num_frames=args.num_frames,
                        input_size=tuple(args.input_size),
                        stored_frame_rate=args.stored_frame_rate,
                    )
                else:
                    val_cache_files = val_resume.valid_files
                
                # Exit after caching if --cache_only is set
                if args.cache_only:
                    print(f"\n[cache] --cache_only: Caching complete, exiting")
                    sys.exit(0)
                    
            except Exception as e:
                print(f"\n[cache] FAILED: {e}")
                sys.exit(1)  # Non-zero exit for SLURM afterok dependency

    # Create activation dataloaders for SAE training
    # Use streaming mode for large datasets that don't fit in RAM
    in_memory = not args.streaming
    print(f"\n[data] Creating activation dataloaders (streaming={args.streaming})...")
    
    sae_batch_size = args.batch_size * args.sae_batch_multiplier
    train_act_loader, train_meta = create_activation_dataloader(
        train_cache_dir,
        batch_size=sae_batch_size,
        shuffle=True,
        num_workers=4 if args.streaming else 0,  # Workers help with streaming, hurt with in-memory
        in_memory=in_memory,
        max_tokens=args.max_tokens,
    )

    val_act_loader, val_meta = create_activation_dataloader(
        val_cache_dir,
        batch_size=sae_batch_size,
        shuffle=False,
        num_workers=4 if args.streaming else 0,
        in_memory=in_memory,
        max_tokens=args.max_tokens,  # Use same limit for val
    )

    print(f"[data] SAE batch size: {sae_batch_size:,} ({args.batch_size} × {args.sae_batch_multiplier})")
    print(f"[data] Train tokens: {train_meta['tokens_used']:,}")
    print(f"[data] Val tokens: {val_meta['tokens_used']:,}")

    # Create SAE
    sae_config = TopKSAEConfig(
        d_in=hidden_size,
        expansion_factor=args.expansion_factor,
        k=args.k,
    )

    sae = TopKSAE(sae_config)
    print(f"\n[model] Created SAE: {sae}")

    # Create trainer with optional optimizations
    trainer = TopKSAETrainer(
        model=sae,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        compile_model=args.compile,
    )
    
    print(f"[trainer] Accelerate fp16 mixed precision enabled")
    if args.compile:
        print(f"[trainer] torch.compile enabled")

    # Save comprehensive config for reproducibility
    # Get git commit hash if available
    git_commit = None
    try:
        import subprocess
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=orbis_root,
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        pass
    
    config_save = {
        # SAE architecture
        "sae_config": {
            "d_in": sae_config.d_in,
            "d_sae": sae_config.d_sae,
            "expansion_factor": sae_config.expansion_factor,
            "k": sae_config.k,
        },
        # Training hyperparameters
        "training": {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "eval_every": args.eval_every,
            "save_every": args.save_every,
            "seed": args.seed,
        },
        # Activation extraction settings
        "activation": {
            "layer_idx": args.layer,
            "t_noise": args.t_noise,
            "frame_rate": args.frame_rate,
        },
        # Data configuration
        "data": {
            "data_source": args.data_source,
            "input_size": args.input_size,
            "num_frames": args.num_frames,
            "stored_frame_rate": args.stored_frame_rate,
            "num_videos": args.num_videos,
            "num_workers": args.num_workers,
            # Source-specific paths (only include relevant ones)
            "data_path": args.data_path,
            "hdf5_paths_file": args.hdf5_paths_file,
            "val_json": args.val_json,
            "covla_videos_dir": args.covla_videos_dir,
            "covla_captions_dir": args.covla_captions_dir,
            "nuplan_data_dir": args.nuplan_data_dir,
            "image_paths": args.image_paths,
        },
        # Cache settings
        "cache": {
            "cache_dir": str(cache_dir),
            "cache_dtype": args.cache_dtype,
            "streaming": args.streaming,
        },
        # Orbis model
        "orbis": {
            "exp_dir": str(exp_dir),
            "ckpt": args.ckpt,
            "config": args.config,
        },
        # Output paths
        "output": {
            "output_dir": str(output_dir),
            "run_name": barcode,
        },
        # Metadata for reproducibility
        "metadata": {
            "git_commit": git_commit,
            "device": args.device,
            "timestamp": timestamp,
            "train_tokens": train_meta["total_tokens"],
            "val_tokens": val_meta["total_tokens"],
        },
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(config_save, f, indent=2)
    
    print(f"[setup] Saved full config to {output_dir / 'config.json'}")

    # Training loop
    print(f"\n[train] Starting training for {args.num_epochs} epochs...")
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.num_epochs + 1):
        # Train
        train_metrics = train_epoch(trainer, train_act_loader, epoch, args.num_epochs)

        # Evaluate
        if epoch % args.eval_every == 0 or epoch == args.num_epochs:
            val_metrics = eval_epoch(trainer, val_act_loader)

            print(f"\nEpoch {epoch}/{args.num_epochs}:")
            print(f"  Train - loss: {train_metrics['loss']:.4f}, l0: {train_metrics['l0']:.1f}, "
                  f"R²: {train_metrics['explained_variance']:.4f}, cos_sim: {train_metrics['cos_sim']:.4f}")
            print(f"          rel_err: {train_metrics['rel_error']:.4f}, L1: {train_metrics['l1_norm']:.2f}, "
                  f"dead%: {train_metrics['dead_pct']:.1f}")
            print(f"  Val   - loss: {val_metrics['loss']:.4f}, l0: {val_metrics['l0']:.1f}, "
                  f"R²: {val_metrics['explained_variance']:.4f}, cos_sim: {val_metrics['cos_sim']:.4f}")
            print(f"          rel_err: {val_metrics['rel_error']:.4f}, L1: {val_metrics['l1_norm']:.2f}")

            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                sae.save(str(output_dir / "best_sae.pt"))
                print(f"  -> New best model saved!")
        else:
            val_metrics = None
            print(f"Epoch {epoch}/{args.num_epochs}: train_loss={train_metrics['loss']:.4f}, "
                  f"l0={train_metrics['l0']:.1f}, dead%={train_metrics['dead_pct']:.1f}")

        # Phase 2 evaluation at specified epochs
        phase2_metrics = None
        phase2_epochs = args.phase2_eval_epochs or [args.num_epochs]  # Default: only final epoch
        if epoch in phase2_epochs:
            phase2_metrics = run_phase2_evaluation(
                orbis_model=model,
                sae=sae,
                image_dataloader=val_loader,  # Use image dataloader for NLR
                activation_dataloader=val_act_loader,  # Use activation dataloader for coverage
                layer_idx=args.layer,
                device=device,
                t_noise=args.t_noise,
                frame_rate=args.frame_rate,
                max_batches=args.phase2_max_batches,
            )

        # Record history
        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "phase2": phase2_metrics,
        })

        # Save checkpoint periodically
        if epoch % args.save_every == 0:
            sae.save(str(output_dir / f"sae_epoch_{epoch:03d}.pt"))

    # Save final model and history
    sae.save(str(output_dir / "final_sae.pt"))

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[done] Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Outputs saved to: {output_dir}")


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
        choices=["cityscapes", "hdf5", "image_paths", "covla", "nuplan"],
        help="Data source: 'hdf5' (native Orbis data), 'cityscapes', 'image_paths', 'covla', or 'nuplan'",
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

    # Training
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay")
    parser.add_argument("--sae_batch_multiplier", type=int, default=1024,
                        help="SAE batch size = batch_size * sae_batch_multiplier")
    parser.add_argument("--eval_every", type=int, default=5,
                        help="Evaluate every N epochs")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    
    # Phase 2 evaluation (expensive - requires Orbis model inference)
    parser.add_argument("--phase2_eval_epochs", type=int, nargs="*", default=None,
                        help="Epochs at which to run Phase 2 evaluation (NLR, coverage). "
                             "Example: --phase2_eval_epochs 10 25 50. Default: only final epoch")
    parser.add_argument("--phase2_max_batches", type=int, default=50,
                        help="Max batches for Phase 2 evaluation (to limit compute)")

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
