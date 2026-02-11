"""
Shared analysis functions for semantic grounding scripts.

Contains data-source-agnostic correlation, pure feature detection,
and latent ranking logic used by both NuPlan and CoVLA grounding.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from sae.utils.constants import EPSILON, MIN_CORRELATION_SAMPLES

logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """Correlation between a latent and a metadata field."""

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


def compute_correlations(
    latent_matrix: np.ndarray,
    field_matrix: np.ndarray,
    field_names: List[str],
    min_samples: int = MIN_CORRELATION_SAMPLES,
) -> Dict[str, List[CorrelationResult]]:
    """Compute Pearson correlation between each latent and each metadata field.

    Args:
        latent_matrix: (num_samples, num_latents) array.
        field_matrix: (num_samples, num_fields) array (may contain NaN).
        field_names: Names of each column in field_matrix.
        min_samples: Minimum valid samples to compute correlation.

    Returns:
        Dict mapping field_name to sorted list of CorrelationResults.
    """
    num_latents = latent_matrix.shape[1]
    num_samples = latent_matrix.shape[0]

    logger.info(f"Computing correlations for {num_latents} latents, {num_samples} samples")

    results: Dict[str, List[CorrelationResult]] = {f: [] for f in field_names}

    for field_idx, field_name in enumerate(field_names):
        field_values = field_matrix[:, field_idx]
        valid_mask = ~np.isnan(field_values)

        if valid_mask.sum() < min_samples:
            logger.info(f"  Skipping {field_name}: only {valid_mask.sum()} valid samples")
            continue

        valid_values = field_values[valid_mask]
        valid_latents = latent_matrix[valid_mask]

        for latent_idx in range(num_latents):
            latent_values = valid_latents[:, latent_idx]

            if np.std(latent_values) < EPSILON:
                continue

            r, p = stats.pearsonr(latent_values, valid_values)

            results[field_name].append(CorrelationResult(
                latent_idx=latent_idx,
                field_name=field_name,
                pearson_r=float(r),
                p_value=float(p),
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
    latent_correlations: Dict[int, Dict[str, float]] = {}
    for field_name, results in correlations.items():
        for result in results:
            if result.latent_idx not in latent_correlations:
                latent_correlations[result.latent_idx] = {}
            latent_correlations[result.latent_idx][field_name] = result.pearson_r

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


def find_top_activating_latents(
    latent_matrix: np.ndarray,
    top_n: int = 20,
    metric: str = "max",
) -> List[Dict[str, Any]]:
    """Find latents with the highest overall activation across all frames.

    Args:
        latent_matrix: (num_samples, num_latents) array.
        top_n: Number of top latents to return.
        metric: Ranking metric -- "max", "mean", or "std".
    """
    if metric == "max":
        scores = np.max(latent_matrix, axis=0)
    elif metric == "mean":
        scores = np.mean(latent_matrix, axis=0)
    elif metric == "std":
        scores = np.std(latent_matrix, axis=0)
    else:
        raise ValueError(f"Unknown metric: {metric}")

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
            "sparsity": float(np.mean(latent_acts > 0)),
        })

    return results


def find_top_activating_frames(
    latent_activations: np.ndarray,
    latent_idx: int,
    sample_metadata: List[dict],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Find top-K frames that most strongly activate a given latent.

    Args:
        latent_activations: (num_samples, num_latents) array.
        latent_idx: Which latent to rank by.
        sample_metadata: List of per-sample metadata dicts (must have video_id, frame_idx).
        top_k: Number of top frames.
    """
    scores = latent_activations[:, latent_idx]
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        meta = sample_metadata[idx]
        results.append({
            "video_id": meta["video_id"],
            "frame_idx": meta["frame_idx"],
            "activation": float(scores[idx]),
            **{k: v for k, v in meta.items() if k not in ("video_id", "frame_idx")},
        })

    return results
