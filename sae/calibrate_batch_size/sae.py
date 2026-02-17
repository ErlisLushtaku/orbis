#!/usr/bin/env python3
"""
SAE-specific batch size calibration.

Provides batch size calibration for SAE training, which requires full
forward + backward + optimizer steps to accurately measure memory usage.

Usage as CLI:
    python -m sae.calibrate_batch_size.sae --layer 22

Output:
    Saves to: logs_sae/resources/{model}/layer_{layer}/topk_x{exp}_k{k}/{gpu_slug}.json
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

# Add project root to path for standalone execution
ORBIS_ROOT = Path(__file__).resolve().parents[2]
if str(ORBIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ORBIS_ROOT))

from sae.calibrate_batch_size.core import (
    ORBIS_ROOT,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_MODEL_NAME,
    GPUInfo,
    calibrate,
    cleanup_gpu,
)
from sae.topk_sae import TopKSAE, TopKSAEConfig

logger = logging.getLogger(__name__)


def sae_causes_oom(
    sae_config: TopKSAEConfig,
    batch_size: int,
    warmup_steps: int = 5,
) -> bool:
    """
    Test if a batch size causes OOM for SAE training.
    
    Runs full training cycles (forward + backward + optimizer step) to
    accurately measure peak memory usage. This is critical because:
    - Gradient buffers are allocated during backward pass
    - Optimizer state (momentum + variance) is allocated during first step
    - Memory fragmentation occurs over multiple steps
    
    Args:
        sae_config: SAE configuration (includes d_in, d_sae, k)
        batch_size: Number of tokens per batch
        warmup_steps: Number of training cycles to run
        
    Returns:
        True if batch size causes OOM, False otherwise
    """
    model = None
    optimizer = None
    
    try:
        # Create fresh model and optimizer for this test (fp32, matching real training)
        model = TopKSAE(sae_config).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Run multiple full training cycles to stress memory
        for _ in range(warmup_steps):
            # Generate random batch (simulates activation data, fp32)
            batch = torch.randn(
                batch_size, sae_config.d_in,
                device="cuda", dtype=torch.float32
            )
            
            optimizer.zero_grad()
            
            # Forward pass (fp32, matching real training)
            reconstruction, sparse_acts, loss, _, _ = model(batch)
            
            # Backward pass (allocates gradient buffers)
            loss.backward()
            
            # Optimizer step (allocates optimizer state: momentum, variance)
            optimizer.step()
            
            # Clean up iteration tensors
            del batch, reconstruction, sparse_acts, loss
        
        return False  # No OOM
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return True  # OOM detected
        # Preserve original error context for debugging
        logger.error(f"Unexpected RuntimeError during OOM test: {e}")
        raise RuntimeError(f"OOM test failed with unexpected error") from e
        
    finally:
        # CRITICAL: Nested try-finally ensures cleanup_gpu always runs
        # even if deletion fails
        try:
            if optimizer is not None:
                del optimizer
            if model is not None:
                del model
        finally:
            cleanup_gpu()


def calibrate_sae_batch_size(
    sae_config: TopKSAEConfig,
    baseline: int = 512,
    alignment: int = 256,
    warmup_steps: int = 5,
    safety_margin: float = 0.10,
    log: Optional[logging.Logger] = None,
) -> Tuple[int, int]:
    """
    Calibrate SAE training batch size (in tokens).
    
    Design choices:
        - baseline=512: Conservative start that works on most GPUs
        - alignment=256: Matches tensor core alignment for efficiency
        - warmup_steps=5: Enough to capture memory fragmentation
        - safety_margin=0.10: 10% buffer for memory fluctuations
    
    Args:
        sae_config: SAE configuration
        baseline: Starting batch size for power search (default: 512 tokens)
        alignment: Round results to multiple of this (default: 256)
        warmup_steps: Training cycles per OOM test (default: 5)
        safety_margin: Fraction to subtract for safety (default: 0.10)
        log: Optional logger for progress messages
        
    Returns:
        Tuple of (max_tokens, recommended_tokens) per batch
    """
    if log:
        log.info(f"SAE config: d_in={sae_config.d_in}, d_sae={sae_config.d_sae}, k={sae_config.k}")
    
    def oom_test(batch_size: int) -> bool:
        return sae_causes_oom(sae_config, batch_size, warmup_steps)
    
    return calibrate(
        oom_test_fn=oom_test,
        baseline=baseline,
        alignment=alignment,
        safety_margin=safety_margin,
        log=log,
    )


def get_sae_resource_dir(
    layer: int,
    expansion_factor: int,
    k: int,
    model_name: str = DEFAULT_MODEL_NAME,
) -> Path:
    """
    Get the resource directory for SAE calibration results.
    
    Path pattern: logs_sae/resources/{model}/layer_{layer}/topk_x{exp}_k{k}/
    
    Args:
        layer: Transformer layer number
        expansion_factor: SAE expansion factor
        k: Top-K sparsity value
        model_name: Model name (default: orbis_288x512)
        
    Returns:
        Path to resource directory
    """
    sae_type = f"topk_x{expansion_factor}_k{k}"
    return ORBIS_ROOT / "logs_sae" / "resources" / model_name / f"layer_{layer}" / sae_type


def get_sae_resource_path(
    layer: int,
    expansion_factor: int,
    k: int,
    gpu_slug: str,
    model_name: str = DEFAULT_MODEL_NAME,
) -> Path:
    """
    Get the full resource file path for SAE calibration.
    
    Path pattern: logs_sae/resources/{model}/layer_{layer}/topk_x{exp}_k{k}/{gpu_slug}.json
    
    Args:
        layer: Transformer layer number
        expansion_factor: SAE expansion factor
        k: Top-K sparsity value
        gpu_slug: GPU slug (e.g., "nvidia_geforce_rtx_3090_24gb")
        model_name: Model name (default: orbis_288x512)
        
    Returns:
        Path to resource JSON file
    """
    resource_dir = get_sae_resource_dir(layer, expansion_factor, k, model_name)
    return resource_dir / f"{gpu_slug}.json"


def create_sae_resource_data(
    gpu_info: GPUInfo,
    max_batch: int,
    recommended_batch: int,
    expansion_factor: int,
    k: int,
) -> Dict[str, Any]:
    """
    Create the resource data dict for SAE calibration.
    
    Args:
        gpu_info: GPU information
        max_batch: Maximum batch size found
        recommended_batch: Recommended batch size with safety margin
        expansion_factor: SAE expansion factor
        k: Top-K sparsity value
        
    Returns:
        Dict ready for JSON serialization
    """
    sae_config_str = f"topk_x{expansion_factor}_k{k}"
    
    return {
        **gpu_info.to_dict(),
        "max_batch_size_tokens": max_batch,
        "recommended_batch_size_tokens": recommended_batch,
        "precision": "float16",
        "sae_config": sae_config_str,
        "calibrated_at": date.today().isoformat(),
    }


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments for SAE calibration."""
    parser = argparse.ArgumentParser(
        description="Calibrate SAE batch size for current GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # SAE configuration
    parser.add_argument(
        "--layer", type=int, required=True,
        help="Layer number (for resource file organization)"
    )
    parser.add_argument(
        "--expansion-factor", type=int, default=16,
        help="SAE expansion factor (default: 16)"
    )
    parser.add_argument(
        "--k", type=int, default=64,
        help="Top-K sparsity (default: 64)"
    )
    parser.add_argument(
        "--d-in", type=int, default=DEFAULT_HIDDEN_SIZE,
        help=f"Input dimension / hidden size (default: {DEFAULT_HIDDEN_SIZE})"
    )
    
    # Calibration parameters
    parser.add_argument(
        "--baseline", type=int, default=512,
        help="Starting batch size for power search (default: 512)"
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=5,
        help="Training cycles per OOM test (default: 5)"
    )
    parser.add_argument(
        "--safety-margin", type=float, default=0.10,
        help="Safety margin fraction (default: 0.10 = 10%%)"
    )
    
    # Options
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing resource file"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run calibration but don't save resource file"
    )
    
    return parser.parse_args()


def main() -> int:
    """CLI entry point for SAE batch size calibration."""
    from sae.calibrate_batch_size.runner import run_calibration
    from sae.utils.logging import get_logger, setup_sae_logging
    
    args = _parse_args()
    setup_sae_logging(level="INFO")
    log = get_logger(__name__)
    
    # Create SAE config
    sae_config = TopKSAEConfig(
        d_in=args.d_in,
        expansion_factor=args.expansion_factor,
        k=args.k,
    )
    
    log.info(f"Layer: {args.layer}")
    log.info(f"SAE: d_in={sae_config.d_in}, d_sae={sae_config.d_sae}, k={sae_config.k}")
    
    # Define callbacks for the runner
    def get_resource_path(gpu_slug: str) -> Path:
        return get_sae_resource_path(
            layer=args.layer,
            expansion_factor=args.expansion_factor,
            k=args.k,
            gpu_slug=gpu_slug,
        )
    
    def run_calibrate():
        return calibrate_sae_batch_size(
            sae_config=sae_config,
            baseline=args.baseline,
            alignment=256,
            warmup_steps=args.warmup_steps,
            safety_margin=args.safety_margin,
            log=log,
        )
    
    def create_resource_data(gpu_info, max_batch, recommended_batch):
        return create_sae_resource_data(
            gpu_info=gpu_info,
            max_batch=max_batch,
            recommended_batch=recommended_batch,
            expansion_factor=args.expansion_factor,
            k=args.k,
        )
    
    # Run calibration
    result = run_calibration(
        name="SAE",
        unit="tokens",
        get_resource_path=get_resource_path,
        run_calibrate=run_calibrate,
        create_resource_data=create_resource_data,
        force=args.force,
        dry_run=args.dry_run,
        log=log,
    )
    
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
