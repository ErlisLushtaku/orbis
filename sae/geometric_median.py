"""
Geometric median computation via Weiszfeld iterations.

Ported from ViT-Prisma (vit_prisma/sae/training/geometric_median.py).
Used for initializing the decoder bias (b_dec) of sparse autoencoders.
"""

from types import SimpleNamespace
from typing import Optional

import torch
from tqdm import tqdm

from .utils.logging import get_logger

logger = get_logger(__name__)


def weighted_average(points: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights / weights.sum()
    return (points * weights.view(-1, 1)).sum(dim=0)


@torch.no_grad()
def geometric_median_objective(
    median: torch.Tensor, points: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    norms = torch.linalg.norm(points - median.view(1, -1), dim=1)
    return (norms * weights).sum()


def compute_geometric_median(
    points: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    maxiter: int = 100,
    ftol: float = 1e-20,
) -> SimpleNamespace:
    """
    Compute the geometric median of a set of points using Weiszfeld iterations.

    Args:
        points: Tensor of shape (n, d).
        weights: Optional tensor of shape (n,). Uniform if None.
        eps: Minimum denominator to avoid division by zero.
        maxiter: Maximum Weiszfeld iterations.
        ftol: Fractional tolerance for early stopping.

    Returns:
        SimpleNamespace with fields:
            median: Tensor of shape (d,), the geometric median estimate.
            termination: String describing how the algorithm stopped.
    """
    with torch.no_grad():
        if weights is None:
            weights = torch.ones(points.shape[0], device=points.device)

        new_weights = weights
        median = weighted_average(points, weights)
        objective_value = geometric_median_objective(median, points, weights)

        early_termination = False
        for _ in tqdm(range(maxiter), desc="Geometric median"):
            prev_obj_value = objective_value

            norms = torch.linalg.norm(points - median.view(1, -1), dim=1)
            new_weights = weights / torch.clamp(norms, min=eps)
            median = weighted_average(points, new_weights)
            objective_value = geometric_median_objective(median, points, weights)

            if abs(prev_obj_value - objective_value) <= ftol * objective_value:
                early_termination = True
                break

    median = weighted_average(points, new_weights)
    return SimpleNamespace(
        median=median,
        termination=(
            "converged within tolerance"
            if early_termination
            else "maximum iterations reached"
        ),
    )
