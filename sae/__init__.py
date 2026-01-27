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

__all__ = [
    # SAE Model
    "TopKSAE",
    "TopKSAEConfig",
    "TopKSAETrainer",
    # Activation Extraction & Intervention
    "ActivationExtractor",
    "ActivationIntervenor",
    "ZeroAblationIntervenor",
    # Caching
    "prepare_activation_cache",
    "load_activation_cache",
    "create_activation_dataloader",
    "CachedActivationDataset",
    "StreamingActivationDataset",
    # Metrics - Phase 1 (Training)
    "compute_loss_recovered",
    "compute_dead_features",
    "compute_activation_density",
    # Metrics - Phase 2 (Evaluation)
    "compute_normalized_loss_recovered",
    "compute_dictionary_coverage",
    # Metrics - Phase 3 (Analysis)
    "compute_temporal_stability",
    # Full Evaluation Suite
    "run_full_evaluation",
]
