#!/usr/bin/env python3
"""
Orbis-specific batch size calibration.

Provides batch size calibration for Orbis activation caching, which only
requires forward passes (no backward/optimizer) since the Orbis model is frozen.

IMPORTANT: batch_size here refers to the number of CLIPS (video sequences),
NOT individual frames. Each clip contains num_frames sequential frames that
are processed together with temporal attention.

Usage as CLI:
    python -m sae.calibrate_batch_size.orbis --orbis-exp-dir /path/to/exp --layer 22

Output:
    Saves to: resources/{model}/layer_{layer}/{gpu_slug}.json
    Partition map: resources/partition_map.json
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

# Add project root to path for standalone execution
ORBIS_ROOT = Path(__file__).resolve().parents[2]
if str(ORBIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ORBIS_ROOT))

from sae.calibrate_batch_size.core import (
    ORBIS_ROOT,
    ORBIS_PARTITION_MAP_PATH,
    DEFAULT_MODEL_NAME,
    GPUInfo,
    calibrate,
    cleanup_gpu,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Orbis OOM Testing
# =============================================================================

def orbis_causes_oom(
    model: nn.Module,
    input_size: Tuple[int, int],
    num_frames: int,
    batch_size: int,
    warmup_steps: int = 5,
    t_noise: float = 0.0,
    frame_rate: int = 5,
) -> bool:
    """
    Test if a batch size causes OOM for Orbis activation caching.
    
    Runs forward passes only (no backward/optimizer) since the Orbis model
    is frozen during caching. This simulates the memory usage during
    prepare_activation_cache().
    
    CRITICAL: batch_size is the number of CLIPS, not frames!
    Each clip contains num_frames sequential frames processed together.
    
    Memory usage per batch:
        - Input clips: batch_size × num_frames × 3 × H × W
        - Tokenizer output: batch_size × num_frames × spatial_tokens × D
        - ViT activations at each layer for context + target
    
    Args:
        model: Orbis world model (must be in eval mode, on CUDA)
        input_size: (height, width) of input frames
        num_frames: Number of frames per clip (default 6)
        batch_size: Number of clips per batch (NOT frames!)
        warmup_steps: Number of forward passes to run
        t_noise: Noise timestep for denoising (default 0.0)
        frame_rate: Frame rate conditioning value (default 5)
        
    Returns:
        True if batch size causes OOM, False otherwise
    """
    H, W = input_size
    device = next(model.parameters()).device
    
    try:
        for _ in range(warmup_steps):
            # Generate synthetic clip batch
            # Shape: (batch_size, num_frames, 3, H, W) -> flattened to (batch * frames, 3, H, W)
            # This matches how the dataloader stacks frames
            imgs = torch.randn(
                batch_size * num_frames, 3, H, W,
                device=device, dtype=torch.float32
            )
            
            # Encode frames through tokenizer (same as caching.py)
            with torch.no_grad():
                x = model.encode_frames(imgs)
                
                # Prepare for denoising step
                b = x.shape[0]  # This is batch_size * num_frames after tokenization
                t = torch.full((b,), t_noise, device=device)
                
                # Add noise to match training distribution
                target_t, _ = model.add_noise(x, t)
                
                # Validate and reshape for multi-frame processing
                # After encode_frames: (batch*frames, spatial, hidden)
                # Need to reshape to (batch, frames, spatial, hidden)
                if target_t.dim() == 3:
                    # Expected: (batch*frames, spatial, hidden)
                    total_frames = target_t.shape[0]
                    spatial_tokens = target_t.shape[1]
                    hidden_dim = target_t.shape[2]
                    
                    expected_frames = batch_size * num_frames
                    if total_frames != expected_frames:
                        raise ValueError(
                            f"Unexpected tensor shape after encode_frames: got {total_frames} tokens, "
                            f"expected {expected_frames} (batch_size={batch_size} × num_frames={num_frames})"
                        )
                    
                    target_t = target_t.view(batch_size, num_frames, spatial_tokens, hidden_dim)
                    
                elif target_t.dim() == 4:
                    # Already (batch, frames, spatial, hidden)
                    if target_t.shape[0] != batch_size or target_t.shape[1] != num_frames:
                        raise ValueError(
                            f"Unexpected 4D tensor shape: {target_t.shape}, "
                            f"expected batch_size={batch_size}, num_frames={num_frames}"
                        )
                    target_t = target_t.unsqueeze(2)  # Add sequence dim if needed
                else:
                    raise ValueError(
                        f"Unexpected tensor dimension: {target_t.dim()}, shape: {target_t.shape}. "
                        f"Expected 3D (batch*frames, spatial, hidden) or 4D (batch, frames, spatial, hidden)"
                    )
                
                # Run through transformer (same as caching.py)
                fr = torch.full((batch_size,), frame_rate, device=device)
                
                # Context is previous frames (if multi-frame)
                if target_t.shape[1] > 1:
                    context = target_t[:, :-1]
                    target = target_t[:, -1:]
                else:
                    context = None
                    target = target_t
                
                t_batch = torch.full((batch_size,), t_noise, device=device)
                _ = model.vit(target, context, t_batch, frame_rate=fr)
            
            # Clean up iteration tensors
            del imgs, x, target_t, t, fr, target
            if context is not None:
                del context
        
        return False  # No OOM
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return True  # OOM detected
        # Preserve original error context for debugging
        logger.error(f"Unexpected RuntimeError during OOM test: {e}")
        raise RuntimeError(f"OOM test failed with unexpected error") from e
        
    finally:
        cleanup_gpu()


def calibrate_orbis_batch_size(
    model: nn.Module,
    input_size: Tuple[int, int],
    num_frames: int = 6,
    baseline: int = 4,
    alignment: int = 2,
    warmup_steps: int = 5,
    safety_margin: float = 0.10,
    t_noise: float = 0.0,
    frame_rate: int = 5,
    log: Optional[logging.Logger] = None,
) -> Tuple[int, int]:
    """
    Calibrate Orbis caching batch size (in clips, NOT frames).
    
    Design choices:
        - baseline=4: Conservative start since Orbis uses more memory per clip
        - alignment=2: Clips are typically processed in pairs for context/target
        - warmup_steps=5: Enough to capture memory patterns
        - safety_margin=0.10: 10% buffer for memory fluctuations
    
    Args:
        model: Orbis world model (must be in eval mode, on CUDA)
        input_size: (height, width) of input frames
        num_frames: Number of frames per clip (default: 6)
        baseline: Starting batch size for power search (default: 4 clips)
        alignment: Round results to multiple of this (default: 2)
        warmup_steps: Forward passes per OOM test (default: 5)
        safety_margin: Fraction to subtract for safety (default: 0.10)
        t_noise: Noise timestep for denoising (default: 0.0)
        frame_rate: Frame rate conditioning value (default: 5)
        log: Optional logger for progress messages
        
    Returns:
        Tuple of (max_clips, recommended_clips) per batch
    """
    if log:
        log.info(f"Orbis config: input_size={input_size}, num_frames={num_frames}")
    
    def oom_test(batch_size: int) -> bool:
        return orbis_causes_oom(
            model=model,
            input_size=input_size,
            num_frames=num_frames,
            batch_size=batch_size,
            warmup_steps=warmup_steps,
            t_noise=t_noise,
            frame_rate=frame_rate,
        )
    
    return calibrate(
        oom_test_fn=oom_test,
        baseline=baseline,
        alignment=alignment,
        safety_margin=safety_margin,
        log=log,
    )


# =============================================================================
# Resource Path Management
# =============================================================================

def get_orbis_resource_dir(
    layer: int,
    model_name: str = DEFAULT_MODEL_NAME,
) -> Path:
    """
    Get the resource directory for Orbis caching calibration results.
    
    Path pattern: resources/{model}/layer_{layer}/
    
    This mirrors the SAE pattern: logs_sae/resources/{model}/layer_{layer}/...
    
    Args:
        layer: Transformer layer number to extract activations from
        model_name: Model name (e.g., "orbis_288x512")
        
    Returns:
        Path to resource directory
    """
    return ORBIS_ROOT / "resources" / model_name / f"layer_{layer}"


def get_orbis_resource_path(
    layer: int,
    gpu_slug: str,
    model_name: str = DEFAULT_MODEL_NAME,
) -> Path:
    """
    Get the full resource file path for Orbis caching calibration.
    
    Path pattern: resources/{model}/layer_{layer}/{gpu_slug}.json
    
    Args:
        layer: Transformer layer number to extract activations from
        gpu_slug: GPU slug (e.g., "nvidia_geforce_rtx_3090_24gb")
        model_name: Model name (e.g., "orbis_288x512")
        
    Returns:
        Path to resource JSON file
    """
    resource_dir = get_orbis_resource_dir(layer, model_name)
    return resource_dir / f"{gpu_slug}.json"


def create_orbis_resource_data(
    gpu_info: GPUInfo,
    max_batch: int,
    recommended_batch: int,
    layer: int,
    model_name: str,
    input_size: Tuple[int, int],
    num_frames: int,
) -> Dict[str, Any]:
    """
    Create the resource data dict for Orbis caching calibration.
    
    Args:
        gpu_info: GPU information
        max_batch: Maximum batch size (clips) found
        recommended_batch: Recommended batch size with safety margin
        layer: Transformer layer number
        model_name: Model name
        input_size: (height, width) of input frames
        num_frames: Number of frames per clip
        
    Returns:
        Dict ready for JSON serialization
    """
    H, W = input_size
    
    return {
        **gpu_info.to_dict(),
        "max_batch_size_clips": max_batch,
        "recommended_batch_size_clips": recommended_batch,
        "layer": layer,
        "model": model_name,
        "input_size": f"{H}x{W}",
        "num_frames": num_frames,
        "precision": "float32",  # Orbis uses fp32 for forward pass
        "calibrated_at": date.today().isoformat(),
    }


# =============================================================================
# CLI Entry Point
# =============================================================================

def _load_orbis_model(
    config_path: str,
    ckpt_path: str,
    device: torch.device,
) -> nn.Module:
    """Load pre-trained Orbis world model."""
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    
    logger.info(f"Loading config from {config_path}")
    cfg_model = OmegaConf.load(config_path)

    logger.info("Instantiating model...")
    model = instantiate_from_config(cfg_model.model)

    logger.info(f"Loading checkpoint from {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=False)

    # Freeze model and move to device
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    logger.info("Model loaded and frozen")
    return model


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments for Orbis calibration."""
    parser = argparse.ArgumentParser(
        description="Calibrate Orbis caching batch size for current GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Required arguments
    parser.add_argument(
        "--orbis-exp-dir", type=str, required=True,
        help="Path to Orbis experiment directory"
    )
    parser.add_argument(
        "--layer", type=int, required=True,
        help="Layer number to extract activations from (for resource file organization)"
    )
    
    # Input configuration
    parser.add_argument(
        "--input-size", type=int, nargs=2, default=[288, 512],
        metavar=("H", "W"),
        help="Input frame size (height, width) (default: 288 512)"
    )
    parser.add_argument(
        "--num-frames", type=int, default=6,
        help="Number of frames per clip (default: 6)"
    )
    parser.add_argument(
        "--model-name", type=str, default=DEFAULT_MODEL_NAME,
        help=f"Model name for resource organization (default: {DEFAULT_MODEL_NAME})"
    )
    
    # Calibration parameters
    parser.add_argument(
        "--baseline", type=int, default=4,
        help="Starting batch size (clips) for power search (default: 4)"
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=5,
        help="Forward passes per OOM test (default: 5)"
    )
    parser.add_argument(
        "--safety-margin", type=float, default=0.10,
        help="Safety margin fraction (default: 0.10 = 10%%)"
    )
    parser.add_argument(
        "--t-noise", type=float, default=0.0,
        help="Noise timestep for denoising (default: 0.0)"
    )
    parser.add_argument(
        "--frame-rate", type=int, default=5,
        help="Frame rate conditioning value (default: 5)"
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
    """CLI entry point for Orbis caching batch size calibration."""
    from sae.calibrate_batch_size.runner import run_calibration
    from sae.logging_utils import get_logger, setup_sae_logging
    
    args = _parse_args()
    setup_sae_logging(level="INFO")
    log = get_logger(__name__)
    
    input_size = tuple(args.input_size)
    
    # Validate Orbis experiment directory
    orbis_exp_dir = Path(args.orbis_exp_dir).resolve()
    if not orbis_exp_dir.exists():
        log.error(f"Orbis experiment directory not found: {orbis_exp_dir}")
        return 1
    
    config_path = orbis_exp_dir / "config.yaml"
    ckpt_dir = orbis_exp_dir / "checkpoints"
    
    # Find latest checkpoint
    ckpt_files = list(ckpt_dir.glob("*.ckpt"))
    if not ckpt_files:
        log.error(f"No checkpoint files found in {ckpt_dir}")
        return 1
    ckpt_path = max(ckpt_files, key=lambda p: p.stat().st_mtime)
    
    log.info(f"Config: {config_path}")
    log.info(f"Checkpoint: {ckpt_path}")
    log.info(f"Layer: {args.layer}")
    log.info(f"Input size: {input_size[0]}x{input_size[1]}")
    log.info(f"Num frames: {args.num_frames}")
    
    # Load model
    device = torch.device("cuda")
    model = _load_orbis_model(str(config_path), str(ckpt_path), device)
    
    # Define callbacks for the runner
    def get_resource_path(gpu_slug: str) -> Path:
        return get_orbis_resource_path(
            layer=args.layer,
            gpu_slug=gpu_slug,
            model_name=args.model_name,
        )
    
    def run_calibrate():
        return calibrate_orbis_batch_size(
            model=model,
            input_size=input_size,
            num_frames=args.num_frames,
            baseline=args.baseline,
            alignment=2,
            warmup_steps=args.warmup_steps,
            safety_margin=args.safety_margin,
            t_noise=args.t_noise,
            frame_rate=args.frame_rate,
            log=log,
        )
    
    def create_resource_data(gpu_info, max_batch, recommended_batch):
        return create_orbis_resource_data(
            gpu_info=gpu_info,
            max_batch=max_batch,
            recommended_batch=recommended_batch,
            layer=args.layer,
            model_name=args.model_name,
            input_size=input_size,
            num_frames=args.num_frames,
        )
    
    # Run calibration
    result = run_calibration(
        name="Orbis Caching",
        unit="clips",
        get_resource_path=get_resource_path,
        run_calibrate=run_calibrate,
        create_resource_data=create_resource_data,
        force=args.force,
        dry_run=args.dry_run,
        partition_map_path=ORBIS_PARTITION_MAP_PATH,
        log=log,
    )
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
