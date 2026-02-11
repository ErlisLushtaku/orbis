"""
Sparse Autoencoder (SAE) training and analysis module for Orbis world model.

This module provides tools to:
- Train Top-K SAEs on ST-Transformer activations
- Extract activations using forward hooks
- Evaluate reconstruction quality and interpretability
- Visualize learned features with the Feature Dashboard
"""

from .topk_sae import TopKSAE, TopKSAEConfig, TopKSAETrainer
from .activation_hooks import ActivationExtractor, ActivationIntervenor, ZeroAblationIntervenor
from .caching import (
    CacheConfig,
    CacheResumeInfo,
    prepare_activation_cache,
    load_activation_cache,
    create_activation_dataloader,
    CachedActivationDataset,
    StreamingActivationDataset,
)
from .metrics import (
    compute_loss_recovered,
    compute_normalized_loss_recovered,
    compute_dead_features,
    compute_dictionary_coverage,
    compute_activation_density,
    compute_temporal_stability,
    run_full_evaluation,
)
from .utils.logging import (
    setup_sae_logging,
    get_logger,
    format_duration,
    format_bytes,
    format_throughput,
    GPUMonitor,
    stats,
    PhaseTimer,
    EpochStats,
    TimingStats,
)

__all__ = [
    "TopKSAE",
    "TopKSAEConfig",
    "TopKSAETrainer",
    "ActivationExtractor",
    "ActivationIntervenor",
    "ZeroAblationIntervenor",
    "CacheConfig",
    "CacheResumeInfo",
    "prepare_activation_cache",
    "load_activation_cache",
    "create_activation_dataloader",
    "CachedActivationDataset",
    "StreamingActivationDataset",
    "compute_loss_recovered",
    "compute_dead_features",
    "compute_activation_density",
    "compute_normalized_loss_recovered",
    "compute_dictionary_coverage",
    "compute_temporal_stability",
    "run_full_evaluation",
    "setup_sae_logging",
    "get_logger",
    "stats",
    "PhaseTimer",
    "EpochStats",
    "GPUMonitor",
    "TimingStats",
    "format_duration",
    "format_bytes",
    "format_throughput",
]
