"""
Core batch size calibration utilities shared between SAE and Orbis calibration.

This module provides:
- GPU information detection and slug generation
- Partition map management (self-discovering GPU mapping)
- Generic calibration algorithm (power + binary search)
- Resource file management with atomic operations
- GPU memory cleanup utilities
"""

import gc
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from sae.utils.file_lock import atomic_json_file

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

ORBIS_ROOT = Path(__file__).resolve().parents[2]

# Partition maps for each resource directory
# Both map SLURM partition names to GPU slugs
SAE_PARTITION_MAP_PATH = ORBIS_ROOT / "logs_sae" / "resources" / "partition_map.json"
ORBIS_PARTITION_MAP_PATH = ORBIS_ROOT / "resources" / "partition_map.json"

# Legacy alias for backwards compatibility
PARTITION_MAP_PATH = SAE_PARTITION_MAP_PATH

# Default model configuration
DEFAULT_MODEL_NAME = "orbis_288x512"
DEFAULT_HIDDEN_SIZE = 768  # Hidden size for orbis_288x512


# =============================================================================
# GPU Information
# =============================================================================

@dataclass
class GPUInfo:
    """
    GPU information for resource file naming and calibration.
    
    Attributes:
        name: Full GPU name (e.g., "NVIDIA GeForce RTX 3090")
        vram_gb: Total VRAM in GB
        slug: Filename-safe slug (e.g., "nvidia_geforce_rtx_3090_24gb")
    """
    name: str
    vram_gb: float
    slug: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "gpu": self.name,
            "vram_total_gb": round(self.vram_gb, 1),
        }


def get_gpu_slug(name: str, vram_gb: float) -> str:
    """
    Generate a consistent, filename-safe GPU slug.
    
    Format: {sanitized_name}_{vram}gb
    Example: "nvidia_geforce_rtx_3090_24gb"
    
    Args:
        name: GPU name string
        vram_gb: VRAM in GB
        
    Returns:
        Lowercase slug with underscores, including VRAM
    """
    # Sanitize: lowercase, replace non-alphanumeric with underscore, strip trailing
    base = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
    return f"{base}_{int(vram_gb)}gb"


def get_gpu_info() -> GPUInfo:
    """
    Detect current GPU and return GPUInfo.
    
    Returns:
        GPUInfo with name, VRAM, and slug
        
    Raises:
        RuntimeError: If CUDA is not available
    """
    import torch
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Cannot detect GPU.")
    
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    slug = get_gpu_slug(name, vram_gb)
    
    return GPUInfo(name=name, vram_gb=vram_gb, slug=slug)


# =============================================================================
# Partition Map (Self-Discovery)
# =============================================================================

def get_current_partition() -> Optional[str]:
    """
    Get SLURM partition from environment.
    
    Only available on compute nodes when running under SLURM.
    Returns None on login nodes or non-SLURM environments.
    """
    return os.environ.get("SLURM_JOB_PARTITION")


def load_partition_map(path: Optional[Path] = None) -> Dict[str, str]:
    """
    Load the partition -> gpu_slug mapping from disk.
    
    Args:
        path: Path to partition map JSON file (default: SAE_PARTITION_MAP_PATH)
    
    Returns:
        Dict mapping partition names to GPU slugs, empty if file doesn't exist
    """
    if path is None:
        path = SAE_PARTITION_MAP_PATH
    
    if not path.exists():
        return {}
    
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load partition map from {path}: {e}")
        return {}


def update_partition_map(
    partition: str,
    gpu_slug: str,
    path: Optional[Path] = None,
) -> None:
    """
    Atomically update the partition map with a new mapping.
    
    Uses file locking to prevent race conditions when multiple
    calibration jobs finish simultaneously.
    
    Args:
        partition: SLURM partition name
        gpu_slug: GPU slug (e.g., "nvidia_geforce_rtx_3090_24gb")
        path: Path to partition map JSON file (default: SAE_PARTITION_MAP_PATH)
    """
    if path is None:
        path = SAE_PARTITION_MAP_PATH
    
    with atomic_json_file(path) as (data, save):
        if data.get(partition) != gpu_slug:
            data[partition] = gpu_slug
            save(data)
            logger.info(f"Updated partition map: {partition} -> {gpu_slug}")
        else:
            logger.debug(f"Partition map already up-to-date: {partition} -> {gpu_slug}")


def resolve_partition_to_gpu_slug(
    partition: str,
    path: Optional[Path] = None,
) -> Optional[str]:
    """
    Look up GPU slug for a partition.
    
    Used by launch.py on the login node to determine GPU type
    for a partition before submitting jobs.
    
    Args:
        partition: SLURM partition name
        path: Path to partition map JSON file (default: SAE_PARTITION_MAP_PATH)
        
    Returns:
        GPU slug if partition is in map, None otherwise
    """
    mapping = load_partition_map(path)
    return mapping.get(partition)


# =============================================================================
# Generic Calibration Algorithm
# =============================================================================

def calibrate(
    oom_test_fn: Callable[[int], bool],
    baseline: int = 512,
    alignment: int = 256,
    safety_margin: float = 0.10,
    log: Optional[logging.Logger] = None,
) -> Tuple[int, int]:
    """
    Generic power + binary search calibration algorithm.
    
    Finds the maximum batch size that doesn't cause OOM, then applies
    a safety margin and alignment.
    
    Algorithm:
        1. Power search: Double from baseline until OOM
        2. Binary search: Refine between last success and first failure
        3. Apply safety margin and round to alignment
    
    Design choices:
        - baseline=512: Safe starting point for most GPUs with SAE
        - alignment=256: Matches typical tensor core alignment for efficiency
        - safety_margin=0.10: 10% buffer accounts for memory fluctuations
    
    Args:
        oom_test_fn: Function(batch_size) -> True if OOM occurs
        baseline: Starting batch size for power search
        alignment: Round results to multiple of this value
        safety_margin: Fraction to subtract for safety (0.10 = 10%)
        log: Optional logger for progress messages
        
    Returns:
        Tuple of (max_batch_size, recommended_batch_size)
    """
    if log:
        log.info(f"Starting calibration: baseline={baseline}, alignment={alignment}")
    
    # Phase 1: Power search - find upper bound
    batch_size = baseline
    while not oom_test_fn(batch_size):
        if log:
            log.info(f"  Batch size {batch_size:,} OK")
        batch_size *= 2
    
    if log:
        log.info(f"  Batch size {batch_size:,} OOM - starting binary search")
    
    upper, lower = batch_size, batch_size // 2
    
    # Phase 2: Binary search - refine within alignment
    while upper - lower > alignment:
        mid = ((upper + lower) // 2 // alignment) * alignment  # Keep aligned
        if oom_test_fn(mid):
            if log:
                log.info(f"  Binary search: {mid:,} OOM")
            upper = mid
        else:
            if log:
                log.info(f"  Binary search: {mid:,} OK")
            lower = mid
    
    max_batch = lower
    
    # Apply safety margin and align
    recommended = (int(max_batch * (1 - safety_margin)) // alignment) * alignment
    
    # Ensure recommended is at least baseline
    recommended = max(baseline, recommended)
    
    if log:
        log.info(f"Calibration complete: max={max_batch:,}, recommended={recommended:,}")
    
    return max_batch, recommended


# =============================================================================
# GPU Cleanup
# =============================================================================

def cleanup_gpu() -> None:
    """
    Force GPU memory cleanup between OOM tests.
    
    Critical sequence:
        1. gc.collect() - Release Python references
        2. torch.cuda.synchronize() - Wait for GPU operations to complete
        3. torch.cuda.empty_cache() - Release cached memory
    
    The synchronize() call is essential: without it, empty_cache() may
    fail to release memory from in-flight operations.
    """
    import torch
    
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


# =============================================================================
# Resource File Management
# =============================================================================

def save_resource_atomic(path: Path, data: Dict[str, Any]) -> None:
    """
    Atomically save a resource JSON file with file locking.
    
    Uses atomic_json_file to prevent corruption from concurrent writes.
    Creates parent directories if needed.
    
    Args:
        path: Path to resource file
        data: Dict to save as JSON
    """
    with atomic_json_file(path) as (_, save):
        save(data)
    logger.info(f"Saved resource file: {path}")


def load_resource(path: Path) -> Optional[Dict[str, Any]]:
    """
    Load a resource JSON file if it exists.
    
    Args:
        path: Path to resource file
        
    Returns:
        Parsed JSON dict, or None if file doesn't exist
    """
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load resource {path}: {e}")
            return None
    return None
