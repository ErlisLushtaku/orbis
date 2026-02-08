"""
Metrics for evaluating SAE quality on Orbis world model.

Implements:
- Loss Recovered: How well SAE reconstructions preserve model behavior
- Dead Features: Count of features that never activate
- Activation Density: Per-feature activation frequency histograms
"""

import time
from itertools import islice
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from .activation_hooks import ActivationIntervenor, ActivationExtractor, ZeroAblationIntervenor
from .topk_sae import TopKSAE
from .utils.logging_utils import get_logger, format_duration

logger = get_logger(__name__)


@torch.no_grad()
def compute_loss_recovered(
    model: nn.Module,
    sae: TopKSAE,
    dataloader: DataLoader,
    layer_idx: int,
    device: torch.device,
    t_noise: float = 0.0,
    frame_rate: int = 5,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute the "Loss Recovered" metric.
    
    This measures how well SAE reconstructions preserve the model's behavior
    by comparing prediction errors with and without SAE intervention.
    
    Loss Recovered = 1 - (error_with_sae / baseline_error)
    
    A value of 1.0 means perfect reconstruction (no increase in error).
    A value of 0.0 means the SAE reconstruction doubles the error.
    
    Args:
        model: Orbis world model
        sae: Trained SAE model
        dataloader: DataLoader yielding image batches
        layer_idx: Which layer to intervene on
        device: Device to run on
        t_noise: Noise timestep
        frame_rate: Frame rate conditioning
        max_batches: Maximum batches to evaluate (None = all)
        
    Returns:
        Dictionary with baseline_error, sae_error, and loss_recovered
    """
    model.eval()
    sae.eval()
    
    interventor = ActivationIntervenor(model, sae, layer_idx=layer_idx)
    
    baseline_errors = []
    sae_errors = []
    
    iterator = tqdm(islice(dataloader, max_batches), total=max_batches, desc="Computing Loss Recovered")
    
    for batch in iterator:
        # Handle batch format
        if isinstance(batch, (list, tuple)):
            imgs = batch[0]
        elif isinstance(batch, dict):
            imgs = batch.get('images', batch.get('image'))
        else:
            imgs = batch
        
        imgs = imgs.to(device)
        b = imgs.shape[0]
        
        # Encode frames
        x = model.encode_frames(imgs)
        
        # Setup for forward pass
        t = torch.full((b,), t_noise, device=device)
        
        if t_noise > 0:
            target_t, noise = model.add_noise(x, t)
        else:
            target_t = x
            noise = torch.zeros_like(x)
        
        if target_t.dim() == 4:
            target_t = target_t.unsqueeze(1)
        
        fr = torch.full((b,), frame_rate, device=device)
        
        # Prepare context and target
        if target_t.shape[1] > 1:
            context = target_t[:, :-1]
            target = target_t[:, -1:]
        else:
            context = None
            target = target_t
        
        # Ground truth (velocity field target)
        gt_target = model.A(t).view(-1, 1, 1, 1) * x[:, -1] + model.B(t).view(-1, 1, 1, 1) * noise[:, -1] \
                    if x.dim() == 5 else model.A(t).view(-1, 1, 1, 1) * x + model.B(t).view(-1, 1, 1, 1) * noise
        
        # Baseline prediction (no intervention)
        baseline_pred = model.vit(target, context, t, frame_rate=fr)
        baseline_error = F.mse_loss(baseline_pred.float(), gt_target.float())
        baseline_errors.append(baseline_error.item())
        
        # SAE intervention prediction
        with interventor.intervene():
            sae_pred = model.vit(target, context, t, frame_rate=fr)
        sae_error = F.mse_loss(sae_pred.float(), gt_target.float())
        sae_errors.append(sae_error.item())
    
    # Compute averages
    baseline_avg = np.mean(baseline_errors)
    sae_avg = np.mean(sae_errors)
    
    # Loss recovered (1.0 = perfect, 0.0 = doubled error)
    loss_recovered = 1.0 - (sae_avg / (baseline_avg + 1e-8))
    
    return {
        "baseline_error": baseline_avg,
        "sae_error": sae_avg,
        "loss_recovered": loss_recovered,
        "error_ratio": sae_avg / (baseline_avg + 1e-8),
    }


@torch.no_grad()
def compute_normalized_loss_recovered(
    model: nn.Module,
    sae: TopKSAE,
    dataloader: DataLoader,
    layer_idx: int,
    device: torch.device,
    t_noise: float = 0.0,
    frame_rate: int = 5,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute the Normalized Loss Recovered (NLR) metric.
    
    This is the scientifically rigorous version of loss recovered that measures
    the fraction of model performance preserved using the formula:
    
        NLR = (L_zero - L_sae) / (L_zero - L_base)
    
    Where:
        - L_base: Baseline model loss (no intervention)
        - L_sae: Model loss when target activations are replaced by SAE reconstructions
        - L_zero: Model loss when target activations are zero-ablated (worst case)
    
    Interpretation:
        - NLR = 1.0: Perfect reconstruction (L_sae = L_base)
        - NLR = 0.0: SAE provides no benefit over zero ablation (L_sae = L_zero)
        - NLR > 1.0: SAE somehow improves on baseline (rare, could indicate overfitting)
        - NLR < 0.0: SAE is worse than zero ablation (broken SAE)
    
    Args:
        model: Orbis world model
        sae: Trained SAE model
        dataloader: DataLoader yielding image batches
        layer_idx: Which layer to intervene on
        device: Device to run on
        t_noise: Noise timestep
        frame_rate: Frame rate conditioning
        max_batches: Maximum batches to evaluate (None = all)
        
    Returns:
        Dictionary with L_base, L_sae, L_zero, and normalized_loss_recovered
    """
    model.eval()
    sae.eval()
    
    sae_interventor = ActivationIntervenor(model, sae, layer_idx=layer_idx)
    zero_interventor = ZeroAblationIntervenor(model, layer_idx=layer_idx)
    
    baseline_errors = []
    sae_errors = []
    zero_errors = []
    
    iterator = tqdm(islice(dataloader, max_batches), total=max_batches, desc="Computing Normalized Loss Recovered")
    
    for batch in iterator:
        # Handle batch format
        if isinstance(batch, (list, tuple)):
            imgs = batch[0]
        elif isinstance(batch, dict):
            imgs = batch.get('images', batch.get('image'))
        else:
            imgs = batch
        
        imgs = imgs.to(device)
        b = imgs.shape[0]
        
        # Encode frames
        x = model.encode_frames(imgs)
        
        # Setup for forward pass
        t = torch.full((b,), t_noise, device=device)
        
        if t_noise > 0:
            target_t, noise = model.add_noise(x, t)
        else:
            target_t = x
            noise = torch.zeros_like(x)
        
        if target_t.dim() == 4:
            target_t = target_t.unsqueeze(1)
        
        fr = torch.full((b,), frame_rate, device=device)
        
        # Prepare context and target
        if target_t.shape[1] > 1:
            context = target_t[:, :-1]
            target = target_t[:, -1:]
        else:
            context = None
            target = target_t
        
        # Ground truth (velocity field target)
        gt_target = model.A(t).view(-1, 1, 1, 1) * x[:, -1] + model.B(t).view(-1, 1, 1, 1) * noise[:, -1] \
                    if x.dim() == 5 else model.A(t).view(-1, 1, 1, 1) * x + model.B(t).view(-1, 1, 1, 1) * noise
        
        # L_base: Baseline prediction (no intervention)
        baseline_pred = model.vit(target, context, t, frame_rate=fr)
        baseline_error = F.mse_loss(baseline_pred.float(), gt_target.float())
        baseline_errors.append(baseline_error.item())
        
        # L_sae: SAE intervention prediction
        with sae_interventor.intervene():
            sae_pred = model.vit(target, context, t, frame_rate=fr)
        sae_error = F.mse_loss(sae_pred.float(), gt_target.float())
        sae_errors.append(sae_error.item())
        
        # L_zero: Zero ablation prediction (worst case)
        with zero_interventor.intervene():
            zero_pred = model.vit(target, context, t, frame_rate=fr)
        zero_error = F.mse_loss(zero_pred.float(), gt_target.float())
        zero_errors.append(zero_error.item())
    
    # Compute averages
    L_base = np.mean(baseline_errors)
    L_sae = np.mean(sae_errors)
    L_zero = np.mean(zero_errors)
    
    # Normalized Loss Recovered: (L_zero - L_sae) / (L_zero - L_base)
    # Measures what fraction of the gap between worst-case and baseline is recovered
    denominator = L_zero - L_base
    if abs(denominator) < 1e-8:
        # Edge case: baseline equals zero ablation (layer has no effect)
        normalized_loss_recovered = 1.0 if abs(L_sae - L_base) < 1e-8 else 0.0
    else:
        normalized_loss_recovered = (L_zero - L_sae) / denominator
    
    return {
        "L_base": L_base,
        "L_sae": L_sae,
        "L_zero": L_zero,
        "normalized_loss_recovered": normalized_loss_recovered,
        # Also include the simple loss recovered for comparison
        "loss_recovered_simple": 1.0 - (L_sae / (L_base + 1e-8)),
    }


@torch.no_grad()
def compute_dead_features(
    sae: TopKSAE,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.0,
    max_batches: Optional[int] = None,
) -> Dict[str, any]:
    """
    Count features that never activate across the dataset.
    
    A "dead" feature is one that has activation <= threshold
    across all samples in the dataset.
    
    Args:
        sae: Trained SAE model
        dataloader: DataLoader yielding activation batches
        device: Device to run on
        threshold: Activation threshold for counting as "active"
        max_batches: Maximum batches to process
        
    Returns:
        Dictionary with dead count, alive count, and dead feature indices
    """
    sae.eval()
    sae = sae.to(device)
    
    d_sae = sae.d_sae
    feature_ever_active = torch.zeros(d_sae, dtype=torch.bool, device=device)
    
    iterator = tqdm(islice(dataloader, max_batches), total=max_batches, desc="Checking dead features")
    
    for batch in iterator:
        batch = batch.to(device)
        
        # Get sparse activations
        sparse_acts = sae.encode(batch)
        
        # Mark features that activated at least once
        batch_active = (sparse_acts > threshold).any(dim=0)
        feature_ever_active = feature_ever_active | batch_active
    
    num_alive = feature_ever_active.sum().item()
    num_dead = d_sae - num_alive
    dead_indices = torch.where(~feature_ever_active)[0].cpu().tolist()
    
    return {
        "num_dead": num_dead,
        "num_alive": num_alive,
        "dead_fraction": num_dead / d_sae,
        "dead_indices": dead_indices,
        "total_features": d_sae,
    }


@torch.no_grad()
def compute_dictionary_coverage(
    sae: TopKSAE,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute dictionary coverage: percentage of features that activate at least once.
    
    This is a Phase 2 evaluation metric that measures how much of the learned
    dictionary is actually being used on the validation set. Low coverage may
    indicate:
        - Many dead/unused features
        - Over-parameterized dictionary
        - Training data distribution mismatch
    
    Args:
        sae: Trained SAE model
        dataloader: DataLoader yielding activation batches (validation set)
        device: Device to run on
        max_batches: Maximum batches to process (None = all)
        
    Returns:
        Dictionary with coverage percentage and feature statistics
    """
    sae.eval()
    sae = sae.to(device)
    
    d_sae = sae.d_sae
    feature_ever_active = torch.zeros(d_sae, dtype=torch.bool, device=device)
    total_samples = 0
    
    iterator = tqdm(islice(dataloader, max_batches), total=max_batches, desc="Computing dictionary coverage")
    
    for batch in iterator:
        batch = batch.to(device)
        batch_size = batch.shape[0]
        
        # Get sparse activations
        sparse_acts = sae.encode(batch)
        
        # Mark features that activated at least once
        batch_active = (sparse_acts > 0).any(dim=0)
        feature_ever_active = feature_ever_active | batch_active
        total_samples += batch_size
    
    num_active_features = feature_ever_active.sum().item()
    coverage_pct = (num_active_features / d_sae) * 100.0
    
    return {
        "dictionary_coverage_pct": coverage_pct,
        "num_active_features": num_active_features,
        "num_dead_features": d_sae - num_active_features,
        "total_features": d_sae,
        "total_samples_evaluated": total_samples,
    }


@torch.no_grad()
def compute_activation_density(
    sae: TopKSAE,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, any]:
    """
    Compute per-feature activation density (frequency of activation).
    
    Args:
        sae: Trained SAE model
        dataloader: DataLoader yielding activation batches
        device: Device to run on
        max_batches: Maximum batches to process
        
    Returns:
        Dictionary with density statistics and histogram data
    """
    sae.eval()
    sae = sae.to(device)
    
    d_sae = sae.d_sae
    feature_activation_counts = torch.zeros(d_sae, dtype=torch.long, device=device)
    total_samples = 0
    
    iterator = tqdm(islice(dataloader, max_batches), total=max_batches, desc="Computing activation density")
    
    for batch in iterator:
        batch = batch.to(device)
        batch_size = batch.shape[0]
        
        # Get sparse activations
        sparse_acts = sae.encode(batch)
        
        # Count activations per feature
        activations = (sparse_acts > 0).sum(dim=0)
        feature_activation_counts += activations
        total_samples += batch_size
    
    # Compute density (fraction of samples where each feature activates)
    density = (feature_activation_counts.float() / total_samples).cpu().numpy()
    
    # Compute statistics
    stats = {
        "mean_density": float(np.mean(density)),
        "median_density": float(np.median(density)),
        "std_density": float(np.std(density)),
        "min_density": float(np.min(density)),
        "max_density": float(np.max(density)),
        "total_samples": total_samples,
    }
    
    # Histogram bins
    hist_counts, hist_edges = np.histogram(density, bins=50)
    stats["histogram_counts"] = hist_counts.tolist()
    stats["histogram_edges"] = hist_edges.tolist()
    stats["density_per_feature"] = density.tolist()
    
    return stats


@torch.no_grad()
def compute_temporal_stability(
    sae: TopKSAE,
    temporal_dataloader: DataLoader,
    device: torch.device,
    num_frames: int = 6,
    num_spatial_tokens: int = None,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute temporal stability of feature activations using lag-1 autocorrelation.
    
    This Phase 3 metric measures how stable/persistent features are across time,
    which indicates whether features represent temporally coherent objects rather
    than frame-specific noise.
    
    For each feature, we compute the Pearson correlation between its activation
    pattern at time t and time t+1, then average across all features.
    
    Interpretation:
        - High autocorrelation (>0.5): Features represent persistent temporal objects
        - Low autocorrelation (<0.2): Features are noisy or frame-specific
        - Negative autocorrelation: Features oscillate (unusual, may indicate issues)
    
    Args:
        sae: Trained SAE model
        temporal_dataloader: DataLoader yielding activation batches with temporal structure.
            Expected shape: (batch_size, d_in) where consecutive samples come from
            the same video sequence. The dataloader should preserve temporal order
            (shuffle=False) and batch_size should be a multiple of (num_frames * num_spatial_tokens).
        device: Device to run on
        num_frames: Number of frames per video clip in the cached activations
        num_spatial_tokens: Number of spatial tokens per frame. If None, will be inferred
            from batch size and num_frames.
        max_batches: Maximum batches to process (None = all)
        
    Returns:
        Dictionary with mean/std/median autocorrelation and per-feature values
    """
    sae.eval()
    sae = sae.to(device)
    
    d_sae = sae.d_sae
    
    # Accumulators for computing correlation across batches
    # We'll collect all per-feature autocorrelations and average at the end
    all_feature_autocorrs = []
    
    iterator = tqdm(islice(temporal_dataloader, max_batches), total=max_batches, desc="Computing temporal stability")
    
    for batch in iterator:
        batch = batch.to(device)
        batch_size = batch.shape[0]
        
        # Get sparse activations for all tokens in batch
        sparse_acts = sae.encode(batch)  # (batch_size, d_sae)
        
        # Infer num_spatial_tokens if not provided
        if num_spatial_tokens is None:
            # Assume batch is flattened (B * F * N, d_in) where B=clips, F=frames, N=spatial
            # Try to infer N from total tokens and num_frames
            # This requires batch_size to be divisible by num_frames
            if batch_size % num_frames != 0:
                # Can't determine temporal structure, skip this batch
                continue
            num_spatial_tokens = batch_size // num_frames
        
        total_tokens_per_clip = num_frames * num_spatial_tokens
        
        if batch_size < total_tokens_per_clip:
            # Not enough tokens for temporal analysis
            continue
        
        # Reshape to (num_clips, num_frames, num_spatial_tokens, d_sae)
        num_clips = batch_size // total_tokens_per_clip
        if num_clips == 0:
            continue
            
        # Truncate to full clips
        truncated_size = num_clips * total_tokens_per_clip
        acts_clipped = sparse_acts[:truncated_size]
        
        # Reshape: (num_clips, num_frames, num_spatial_tokens, d_sae)
        acts_reshaped = acts_clipped.view(num_clips, num_frames, num_spatial_tokens, d_sae)
        
        # Average over spatial tokens to get per-frame feature activations
        # Shape: (num_clips, num_frames, d_sae)
        acts_per_frame = acts_reshaped.mean(dim=2)
        
        # Compute lag-1 autocorrelation for each feature
        # acts_t: frames 0 to F-2, acts_t1: frames 1 to F-1
        acts_t = acts_per_frame[:, :-1, :]   # (num_clips, num_frames-1, d_sae)
        acts_t1 = acts_per_frame[:, 1:, :]   # (num_clips, num_frames-1, d_sae)
        
        # Flatten clips and time for correlation computation
        # Shape: (num_clips * (num_frames-1), d_sae)
        acts_t_flat = acts_t.reshape(-1, d_sae)
        acts_t1_flat = acts_t1.reshape(-1, d_sae)
        
        # Compute Pearson correlation for each feature
        # corr(X, Y) = cov(X,Y) / (std(X) * std(Y))
        mean_t = acts_t_flat.mean(dim=0, keepdim=True)
        mean_t1 = acts_t1_flat.mean(dim=0, keepdim=True)
        
        acts_t_centered = acts_t_flat - mean_t
        acts_t1_centered = acts_t1_flat - mean_t1
        
        # Covariance (sum of products / n)
        cov = (acts_t_centered * acts_t1_centered).mean(dim=0)
        
        # Standard deviations
        std_t = acts_t_centered.std(dim=0)
        std_t1 = acts_t1_centered.std(dim=0)
        
        # Pearson correlation per feature
        # Avoid division by zero for dead features
        denom = std_t * std_t1 + 1e-8
        autocorr = cov / denom  # (d_sae,)
        
        all_feature_autocorrs.append(autocorr.cpu())
    
    if len(all_feature_autocorrs) == 0:
        return {
            "mean_autocorrelation": 0.0,
            "std_autocorrelation": 0.0,
            "median_autocorrelation": 0.0,
            "error": "No valid batches for temporal analysis",
        }
    
    # Average autocorrelations across all batches
    stacked = torch.stack(all_feature_autocorrs, dim=0)  # (num_batches, d_sae)
    mean_per_feature = stacked.mean(dim=0).numpy()  # (d_sae,)
    
    # Filter out NaN values (from dead features)
    valid_autocorrs = mean_per_feature[~np.isnan(mean_per_feature)]
    
    if len(valid_autocorrs) == 0:
        return {
            "mean_autocorrelation": 0.0,
            "std_autocorrelation": 0.0,
            "median_autocorrelation": 0.0,
            "error": "All features have NaN autocorrelation",
        }
    
    return {
        "mean_autocorrelation": float(np.mean(valid_autocorrs)),
        "std_autocorrelation": float(np.std(valid_autocorrs)),
        "median_autocorrelation": float(np.median(valid_autocorrs)),
        "min_autocorrelation": float(np.min(valid_autocorrs)),
        "max_autocorrelation": float(np.max(valid_autocorrs)),
        "num_valid_features": len(valid_autocorrs),
        "num_dead_features": d_sae - len(valid_autocorrs),
        "autocorrelation_per_feature": mean_per_feature.tolist(),
    }


def plot_activation_density_histogram(
    density_stats: Dict,
    save_path: Optional[str] = None,
    title: str = "Activation Density Distribution",
) -> plt.Figure:
    """
    Plot histogram of activation densities.
    
    Args:
        density_stats: Output from compute_activation_density()
        save_path: Optional path to save the figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    density = np.array(density_stats["density_per_feature"])
    
    ax.hist(density, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(density_stats["mean_density"], color='red', linestyle='--', 
               label=f'Mean: {density_stats["mean_density"]:.4f}')
    ax.axvline(density_stats["median_density"], color='green', linestyle='--',
               label=f'Median: {density_stats["median_density"]:.4f}')
    
    ax.set_xlabel("Activation Density (fraction of samples)")
    ax.set_ylabel("Number of Features")
    ax.set_title(title)
    ax.legend()
    
    # Add text box with statistics
    textstr = f'Dead: {(density == 0).sum()}\nTotal: {len(density)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved density histogram to {save_path}")
    
    return fig


def run_full_evaluation(
    model: nn.Module,
    sae: TopKSAE,
    image_dataloader: DataLoader,
    activation_dataloader: DataLoader,
    layer_idx: int,
    device: torch.device,
    output_dir: Path,
    t_noise: float = 0.0,
    frame_rate: int = 5,
    max_batches_loss: int = 50,
    max_batches_density: int = 100,
) -> Dict[str, any]:
    """
    Run full SAE evaluation suite.
    
    Args:
        model: Orbis world model
        sae: Trained SAE
        image_dataloader: DataLoader for images (for loss recovered)
        activation_dataloader: DataLoader for cached activations (for density/dead)
        layer_idx: Layer to intervene on
        device: Device to run on
        output_dir: Directory to save results
        t_noise: Noise timestep
        frame_rate: Frame rate conditioning
        max_batches_loss: Max batches for loss recovered
        max_batches_density: Max batches for density computation
        
    Returns:
        Dictionary with all metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    timing = {}
    
    # 1. Loss Recovered
    logger.info("=" * 60)
    logger.info("Computing Loss Recovered...")
    logger.info("=" * 60)
    start_time = time.perf_counter()
    loss_results = compute_loss_recovered(
        model, sae, image_dataloader, layer_idx, device,
        t_noise=t_noise, frame_rate=frame_rate, max_batches=max_batches_loss
    )
    timing["loss_recovered"] = time.perf_counter() - start_time
    results["loss_recovered"] = loss_results
    logger.info(f"  Baseline Error: {loss_results['baseline_error']:.6f}")
    logger.info(f"  SAE Error:      {loss_results['sae_error']:.6f}")
    logger.info(f"  Loss Recovered: {loss_results['loss_recovered']:.4f}")
    logger.info(f"  Time: {format_duration(timing['loss_recovered'])}")
    
    # 2. Dead Features
    logger.info("=" * 60)
    logger.info("Computing Dead Features...")
    logger.info("=" * 60)
    start_time = time.perf_counter()
    dead_results = compute_dead_features(
        sae, activation_dataloader, device, max_batches=max_batches_density
    )
    timing["dead_features"] = time.perf_counter() - start_time
    results["dead_features"] = {
        k: v for k, v in dead_results.items() if k != "dead_indices"
    }
    results["dead_features"]["dead_indices_sample"] = dead_results["dead_indices"][:100]  # Save first 100
    logger.info(f"  Dead Features:  {dead_results['num_dead']} / {dead_results['total_features']} ({dead_results['dead_fraction']*100:.1f}%)")
    logger.info(f"  Alive Features: {dead_results['num_alive']}")
    logger.info(f"  Time: {format_duration(timing['dead_features'])}")
    
    # 3. Activation Density
    logger.info("=" * 60)
    logger.info("Computing Activation Density...")
    logger.info("=" * 60)
    start_time = time.perf_counter()
    density_results = compute_activation_density(
        sae, activation_dataloader, device, max_batches=max_batches_density
    )
    timing["activation_density"] = time.perf_counter() - start_time
    results["activation_density"] = {
        k: v for k, v in density_results.items() if k != "density_per_feature"
    }
    logger.info(f"  Mean Density:   {density_results['mean_density']:.4f}")
    logger.info(f"  Median Density: {density_results['median_density']:.4f}")
    logger.info(f"  Std Density:    {density_results['std_density']:.4f}")
    logger.info(f"  Time: {format_duration(timing['activation_density'])}")
    
    # Plot and save density histogram
    fig = plot_activation_density_histogram(
        density_results,
        save_path=str(output_dir / "activation_density_histogram.png"),
        title="SAE Feature Activation Density"
    )
    plt.close(fig)
    
    # Save full density array separately (can be large)
    np.save(output_dir / "density_per_feature.npy", 
            np.array(density_results["density_per_feature"]))
    
    # Add timing to results
    results["timing"] = timing
    total_time = sum(timing.values())
    results["timing"]["total"] = total_time
    
    # Save results
    import json
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {output_dir}")
    logger.info(f"Total evaluation time: {format_duration(total_time)}")
    
    return results
