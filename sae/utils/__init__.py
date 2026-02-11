"""Utilities for the SAE module."""

from .file_lock import atomic_json_file
from .constants import (
    EPSILON,
    HISTOGRAM_BINS,
    PLOT_DPI,
    CACHED_BATCH_SIZE,
    CACHE_SAVE_INTERVAL,
    SHARD_SHUFFLE_SIZE,
    DEFAULT_SAE_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    BYTES_PER_FLOAT16,
)
from .data_utils import (
    is_webdataset_mode,
    get_base_data_source,
    extract_batch_images,
    compute_ground_truth,
)
from .model_loading import load_orbis_model, load_sae
from .logging import (
    setup_sae_logging,
    get_logger,
    format_duration,
    format_throughput,
    format_bytes,
    GPUMonitor,
    PhaseRecord,
    PhaseTimer,
    EpochStats,
    TimingStats,
    stats,
)

__all__ = [
    "atomic_json_file",
    "EPSILON",
    "HISTOGRAM_BINS",
    "PLOT_DPI",
    "CACHED_BATCH_SIZE",
    "CACHE_SAVE_INTERVAL",
    "SHARD_SHUFFLE_SIZE",
    "DEFAULT_SAE_BATCH_SIZE",
    "DEFAULT_NUM_WORKERS",
    "BYTES_PER_FLOAT16",
    "is_webdataset_mode",
    "get_base_data_source",
    "extract_batch_images",
    "compute_ground_truth",
    "load_orbis_model",
    "load_sae",
    "setup_sae_logging",
    "get_logger",
    "format_duration",
    "format_throughput",
    "format_bytes",
    "GPUMonitor",
    "PhaseRecord",
    "PhaseTimer",
    "EpochStats",
    "TimingStats",
    "stats",
]
