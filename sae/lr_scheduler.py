"""
Learning rate schedulers for SAE training.

Ported from ViT-Prisma (vit_prisma/sae/training/get_scheduler.py).
"""

import math
from typing import Any, Optional

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


def get_scheduler(
    scheduler_name: Optional[str], optimizer: optim.Optimizer, **kwargs: Any
) -> lr_scheduler.LRScheduler:
    """
    Create a learning rate scheduler.

    Args:
        scheduler_name: One of "constant", "constantwithwarmup",
            "cosineannealingwarmup", or None (constant).
        optimizer: The optimizer to schedule.
        **kwargs: Additional arguments:
            warm_up_steps: Number of warmup steps.
            training_steps: Total training steps.
            lr_end: Minimum LR (absolute, or fraction depending on scheduler).
    """

    def _warmup_cosine_lambda(
        warm_up_steps: int, training_steps: int, lr_end: float
    ):
        def lr_lambda(steps: int) -> float:
            if steps < warm_up_steps:
                return (steps + 1) / warm_up_steps
            progress = (steps - warm_up_steps) / max(training_steps - warm_up_steps, 1)
            return lr_end + 0.5 * (1.0 - lr_end) * (1.0 + math.cos(math.pi * progress))

        return lr_lambda

    if scheduler_name is None or scheduler_name.lower() == "constant":
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda steps: 1.0)

    elif scheduler_name.lower() == "constantwithwarmup":
        warm_up_steps = kwargs.get("warm_up_steps", 0)
        return lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda steps: min(1.0, (steps + 1) / max(warm_up_steps, 1)),
        )

    elif scheduler_name.lower() == "cosineannealingwarmup":
        warm_up_steps = kwargs.get("warm_up_steps", 0)
        training_steps = kwargs.get("training_steps")
        if training_steps is None:
            raise ValueError("training_steps must be provided for cosineannealingwarmup")
        lr_end = kwargs.get("lr_end", 0.0)
        lr_lambda = _warmup_cosine_lambda(warm_up_steps, training_steps, lr_end)
        return lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif scheduler_name.lower() == "cosineannealing":
        training_steps = kwargs.get("training_steps")
        if training_steps is None:
            raise ValueError("training_steps must be provided for cosineannealing")
        lr_end = kwargs.get("lr_end", 0.0)
        return lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=training_steps, eta_min=lr_end
        )

    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
