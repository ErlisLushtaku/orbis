"""
Batch size calibration module for SAE training and Orbis activation caching.

This module automatically calibrates batch sizes based on GPU memory capacity
using a power + binary search algorithm with safety margins.

Two calibration types:
- SAE calibration: batch_size = tokens (requires backward pass + optimizer)
- Orbis calibration: batch_size = clips (forward-only, no backward)

Usage:
    # As library
    from sae.calibrate_batch_size import calibrate_sae_batch_size, get_gpu_info
    
    # As CLI
    python -m sae.calibrate_batch_size.sae --layer 22
    python -m sae.calibrate_batch_size.orbis --orbis-exp-dir /path/to/exp
"""

from sae.calibrate_batch_size.core import (
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_MODEL_NAME,
    ORBIS_ROOT,
    PARTITION_MAP_PATH,
    SAE_PARTITION_MAP_PATH,
    ORBIS_PARTITION_MAP_PATH,
    GPUInfo,
    calibrate,
    cleanup_gpu,
    get_current_partition,
    get_gpu_info,
    get_gpu_slug,
    load_partition_map,
    load_resource,
    resolve_partition_to_gpu_slug,
    save_resource_atomic,
    update_partition_map,
)
from sae.calibrate_batch_size.sae import (
    calibrate_sae_batch_size,
    create_sae_resource_data,
    get_sae_resource_dir,
    get_sae_resource_path,
    sae_causes_oom,
)
from sae.calibrate_batch_size.orbis import (
    calibrate_orbis_batch_size,
    create_orbis_resource_data,
    get_orbis_resource_dir,
    get_orbis_resource_path,
    orbis_causes_oom,
)
from sae.calibrate_batch_size.runner import run_calibration, CalibrationResult

__all__ = [
    # Constants
    "ORBIS_ROOT",
    "PARTITION_MAP_PATH",
    "SAE_PARTITION_MAP_PATH",
    "ORBIS_PARTITION_MAP_PATH",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_HIDDEN_SIZE",
    # GPU Info
    "GPUInfo",
    "get_gpu_slug",
    "get_gpu_info",
    # Partition Map
    "get_current_partition",
    "load_partition_map",
    "update_partition_map",
    "resolve_partition_to_gpu_slug",
    # Generic Calibration
    "calibrate",
    "cleanup_gpu",
    # Resource Management
    "save_resource_atomic",
    "load_resource",
    # SAE Calibration
    "sae_causes_oom",
    "calibrate_sae_batch_size",
    "get_sae_resource_dir",
    "get_sae_resource_path",
    "create_sae_resource_data",
    # Orbis Calibration
    "orbis_causes_oom",
    "calibrate_orbis_batch_size",
    "get_orbis_resource_dir",
    "get_orbis_resource_path",
    "create_orbis_resource_data",
    # CLI Runner
    "run_calibration",
    "CalibrationResult",
]
