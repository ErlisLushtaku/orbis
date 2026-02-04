"""
Shared CLI runner for batch size calibration scripts.

Provides a common execution flow to reduce duplication between
SAE and Orbis calibration CLIs.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from sae.calibrate_batch_size.core import (
    GPUInfo,
    get_current_partition,
    get_gpu_info,
    load_resource,
    save_resource_atomic,
    update_partition_map,
)

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of a calibration run."""
    max_batch: int
    recommended_batch: int
    gpu_info: GPUInfo
    skipped: bool = False  # True if existing resource was used


def run_calibration(
    name: str,
    unit: str,
    get_resource_path: Callable[[str], Path],
    run_calibrate: Callable[[], Tuple[int, int]],
    create_resource_data: Callable[[GPUInfo, int, int], Dict[str, Any]],
    force: bool = False,
    dry_run: bool = False,
    partition_map_path: Optional[Path] = None,
    log: Optional[logging.Logger] = None,
) -> Optional[CalibrationResult]:
    """
    Common calibration execution flow.
    
    Handles:
        1. GPU detection and logging
        2. Resource existence check (skip vs force)
        3. Running calibration (via callback)
        4. Saving resource file
        5. Updating partition map
    
    Args:
        name: Calibration name for logging (e.g., "SAE", "Orbis Caching")
        unit: Batch size unit for logging (e.g., "tokens", "clips")
        get_resource_path: Function(gpu_slug) -> Path to resource file
        run_calibrate: Function() -> (max_batch, recommended_batch)
        create_resource_data: Function(gpu_info, max, rec) -> dict
        force: Overwrite existing resource file
        dry_run: Run calibration but don't save
        partition_map_path: Path to partition map JSON (default: SAE_PARTITION_MAP_PATH)
        log: Logger for progress messages
        
    Returns:
        CalibrationResult or None on error
    """
    if log is None:
        log = logger
    
    log.info("=" * 60)
    log.info(f"{name} Batch Size Calibration (Self-Discovery)")
    log.info("=" * 60)
    
    # Step 1: Detect GPU hardware
    try:
        gpu_info = get_gpu_info()
    except RuntimeError as e:
        log.error(str(e))
        return None
    
    log.info(f"GPU: {gpu_info.name}")
    log.info(f"VRAM: {gpu_info.vram_gb:.1f} GB")
    log.info(f"Slug: {gpu_info.slug}")
    
    # Step 2: Check if hardware file already exists
    resource_path = get_resource_path(gpu_info.slug)
    existing = load_resource(resource_path)
    
    if existing and not force:
        # Hardware already calibrated - skip OOM search
        max_key = f"max_batch_size_{unit}"
        rec_key = f"recommended_batch_size_{unit}"
        
        log.info(f"Hardware file exists: {resource_path}")
        log.info(f"  Max batch size:         {existing[max_key]:,} {unit}")
        log.info(f"  Recommended batch size: {existing[rec_key]:,} {unit}")
        log.info("Skipping OOM search (use --force to recalibrate)")
        
        # But still update partition map!
        _update_partition(gpu_info.slug, partition_map_path, log)
        
        return CalibrationResult(
            max_batch=existing[max_key],
            recommended_batch=existing[rec_key],
            gpu_info=gpu_info,
            skipped=True,
        )
    
    if existing and force:
        log.info(f"Force flag set - recalibrating (overwriting {resource_path})")
    
    # Step 3: Run calibration
    log.info("-" * 60)
    max_batch, recommended_batch = run_calibrate()
    log.info("-" * 60)
    
    # Step 4: Report results
    log.info("Results:")
    log.info(f"  Max batch size:         {max_batch:,} {unit}")
    log.info(f"  Recommended batch size: {recommended_batch:,} {unit}")
    
    # Step 5: Save resource file
    if dry_run:
        log.info("Dry run - not saving resource file")
    else:
        resource_data = create_resource_data(gpu_info, max_batch, recommended_batch)
        save_resource_atomic(resource_path, resource_data)
    
    # Step 6: Update partition map
    _update_partition(gpu_info.slug, partition_map_path, log)
    
    log.info("=" * 60)
    log.info("Calibration complete")
    log.info("=" * 60)
    
    return CalibrationResult(
        max_batch=max_batch,
        recommended_batch=recommended_batch,
        gpu_info=gpu_info,
        skipped=False,
    )


def _update_partition(
    gpu_slug: str,
    partition_map_path: Optional[Path],
    log: logging.Logger,
) -> None:
    """Update partition map with current GPU slug."""
    partition = get_current_partition()
    if partition:
        update_partition_map(partition, gpu_slug, partition_map_path)
        log.info(f"Updated partition map: {partition} -> {gpu_slug}")
    else:
        log.info("No SLURM partition detected (not running under SLURM?)")
