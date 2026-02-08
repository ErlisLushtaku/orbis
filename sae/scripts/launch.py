#!/usr/bin/env python3
"""
SAE Training Orchestrator: Manages calibration, caching, and training as SLURM jobs.

This script checks cache and calibration status and submits appropriate SLURM jobs:
1. If calibration resource missing: Submit calibration job first (finds optimal batch size)
2. If cache is incomplete: Submit cache job (depends on calibration if needed)
3. Submit train job (depends on cache and uses calibrated batch size)

Self-Discovery System:
    - GPU hardware is automatically detected by calibration jobs on compute nodes
    - Partition-to-GPU mappings are stored in logs_sae/resources/partition_map.json
    - Hardware-specific calibration results are shared across partitions with same GPU

Dependency chain: calibrate_batch_size_sae.sh -> cache.sh -> train.sh

Usage:
    # Full pipeline (calibrate if needed + cache + train)
    python sae/scripts/launch.py --data_source nuplan --layer 22 --k 64 --expansion 16
    
    # Force cache rebuild (with confirmation)
    python sae/scripts/launch.py --data_source covla --layer 12 --rebuild_cache
    
    # Cache only (no training)
    python sae/scripts/launch.py --data_source nuplan --layer 22 --cache_only
    
    # Train only (assume cache exists)
    python sae/scripts/launch.py --data_source covla --layer 12 --train_only
    
    # Skip calibration (use manual --sae_batch_size)
    python sae/scripts/launch.py --data_source nuplan --layer 22 --skip_calibration --sae_batch_size 8192
    
    # Dry run (print commands without executing)
    python sae/scripts/launch.py --data_source nuplan --layer 22 --dry_run
    
    # Skip confirmation prompts (for automation)
    python sae/scripts/launch.py --data_source nuplan --layer 22 --yes
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

ORBIS_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "orbis_288x512"

# Cache and log base directories
CACHE_BASE = ORBIS_ROOT / "logs_sae" / "sae_cache"
RESOURCE_BASE = ORBIS_ROOT / "logs_sae" / "resources"
LOG_BASE = ORBIS_ROOT / "sae" / "slurm" / "logs"

# Unified script paths (parametric by --data_source)
CACHE_SCRIPT = ORBIS_ROOT / "sae" / "slurm" / "cache.sh"
TRAIN_SCRIPT = ORBIS_ROOT / "sae" / "slurm" / "train.sh"
CALIBRATE_BATCH_SIZE_SAE_SCRIPT = ORBIS_ROOT / "sae" / "slurm" / "calibrate_batch_size_sae.sh"

# Default partition (used when none specified)
DEFAULT_PARTITION = "tflmb_gpu-rtx4090"


# =============================================================================
# Helper Functions
# =============================================================================

def confirm(message: str, default: bool = False) -> bool:
    """
    Ask user for confirmation.
    
    Args:
        message: Question to ask
        default: Default answer if user just presses Enter
        
    Returns:
        True if user confirms, False otherwise
    """
    suffix = " [Y/n]: " if default else " [y/N]: "
    try:
        response = input(message + suffix).strip().lower()
        if not response:
            return default
        return response in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        logger.info("Aborted.")
        sys.exit(1)


def is_webdataset_mode(data_source: str) -> bool:
    """Check if data_source uses WebDataset format."""
    return data_source.endswith("-webdataset")


def get_base_data_source(data_source: str) -> str:
    """Get base data source name (without -webdataset suffix)."""
    if data_source.endswith("-webdataset"):
        return data_source[:-len("-webdataset")]
    return data_source


def get_cache_dir(data_source: str, layer: int) -> Path:
    """Get the cache directory for a given data source and layer."""
    return CACHE_BASE / data_source / MODEL_NAME / f"layer_{layer}"


def is_cache_complete(cache_dir: Path) -> bool:
    """Check if cache is complete by verifying _meta.json exists in both train and val."""
    train_meta = cache_dir / "train" / "_meta.json"
    val_meta = cache_dir / "val" / "_meta.json"
    return train_meta.exists() and val_meta.exists()


def get_cache_info(cache_dir: Path) -> dict:
    """Get information about existing cache for display."""
    info = {"train": None, "val": None}
    for split in ["train", "val"]:
        meta_file = cache_dir / split / "_meta.json"
        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
                info[split] = {
                    "num_files": meta.get("num_files", "?"),
                    "total_tokens": meta.get("total_tokens", "?"),
                }
            except (json.JSONDecodeError, IOError):
                pass
    return info


def generate_cache_id(seed: int) -> str:
    """Generate cache job identifier: cache_s{seed}_{timestamp}."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"cache_s{seed}_{timestamp}"


def generate_train_barcode(expansion: int, k: int, seed: int) -> str:
    """Generate training job barcode: topk_x{exp}_k{k}_s{seed}_{timestamp}."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"topk_x{expansion}_k{k}_s{seed}_{timestamp}"


def get_log_dir(log_type: str, data_source: str, layer: int) -> Path:
    """
    Get log directory path.
    
    Args:
        log_type: "sae_cache" or "runs"
        data_source: "nuplan" or "covla"
        layer: Layer number
    """
    if log_type == "sae_cache":
        return LOG_BASE / "sae_cache" / data_source / MODEL_NAME / f"layer_{layer}"
    else:  # runs
        return LOG_BASE / "runs" / data_source / MODEL_NAME / f"layer_{layer}" / "train"


def submit_job(
    script_path: Path,
    args: list,
    job_name: str = None,
    output_path: Path = None,
    error_path: Path = None,
    dependency: str = None,
    dry_run: bool = False,
) -> str:
    """
    Submit a SLURM job using sbatch.
    
    Args:
        script_path: Path to the SLURM script
        args: List of arguments to pass to the script
        job_name: Override job name
        output_path: Override stdout log path
        error_path: Override stderr log path
        dependency: SLURM dependency string (e.g., "afterok:12345")
        dry_run: If True, print command without executing
        
    Returns:
        Job ID if submitted, or empty string for dry run
    """
    cmd = ["sbatch"]
    
    if job_name:
        cmd.extend(["--job-name", job_name])
    if output_path:
        cmd.extend(["--output", str(output_path)])
    if error_path:
        cmd.extend(["--error", str(error_path)])
    if dependency:
        cmd.extend(["--dependency", dependency])
    
    cmd.append(str(script_path))
    cmd.extend(args)
    
    if dry_run:
        logger.info(f"[DRY RUN] {' '.join(cmd)}")
        return ""
    
    logger.info(f"Submitting: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"sbatch failed: {result.stderr}")
        sys.exit(1)
    
    # Parse job ID from output like "Submitted batch job 12345"
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if match:
        job_id = match.group(1)
        logger.info(f"Job submitted: {job_id}")
        return job_id
    else:
        logger.warning(f"Could not parse job ID from: {result.stdout}")
        return ""


# =============================================================================
# Calibration Resource Management (Self-Discovery)
# =============================================================================

# Import batch size calibration utilities (lazy to avoid circular imports)
_calibrate_batch_size_module = None

def _get_calibrate_batch_size_module():
    """Lazy import of batch size calibration module."""
    global _calibrate_batch_size_module
    if _calibrate_batch_size_module is None:
        from sae import calibrate_batch_size
        _calibrate_batch_size_module = calibrate_batch_size
    return _calibrate_batch_size_module


def get_gpu_slug_for_partition(partition: str) -> str:
    """
    Get GPU slug for a partition using the self-discovery partition map.
    
    The partition map is populated by calibration jobs as they run on
    different partitions. If the partition is unknown, returns None.
    
    Args:
        partition: SLURM partition name
        
    Returns:
        GPU slug if partition is in map, None otherwise
    """
    cal = _get_calibrate_batch_size_module()
    return cal.resolve_partition_to_gpu_slug(partition)


def get_resource_path(layer: int, expansion: int, k: int, gpu_slug: str) -> Path:
    """
    Get the expected resource file path for a given configuration and GPU.
    
    Path pattern: logs_sae/resources/{model}/layer_{layer}/topk_x{exp}_k{k}/{gpu_slug}.json
    
    Args:
        layer: Transformer layer number
        expansion: SAE expansion factor
        k: Top-K sparsity value
        gpu_slug: GPU slug (e.g., "nvidia_geforce_rtx_3090_24gb")
        
    Returns:
        Path to resource file
    """
    cal = _get_calibrate_batch_size_module()
    return cal.get_sae_resource_path(layer, expansion, k, gpu_slug)


def load_calibration_resource(resource_path: Path) -> dict:
    """
    Load batch size calibration resource if it exists.
    
    Returns:
        Resource dict if exists, None otherwise.
    """
    cal = _get_calibrate_batch_size_module()
    return cal.load_resource(resource_path)


def generate_calibrate_id(expansion: int, k: int) -> str:
    """Generate calibration job identifier."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"calibrate_x{expansion}_k{k}_{timestamp}"


def submit_calibration_job(
    layer: int,
    expansion: int,
    k: int,
    dry_run: bool = False,
) -> str:
    """
    Submit a calibration job to find optimal SAE batch size.
    
    The calibration job will:
    1. Detect GPU hardware on the compute node
    2. Run OOM test to find max batch size (if not already calibrated)
    3. Update partition_map.json with partition -> gpu_slug mapping
    
    Args:
        layer: Layer number
        expansion: SAE expansion factor
        k: Top-K sparsity
        dry_run: If True, print command without executing
        
    Returns:
        Job ID if submitted, or empty string for dry run
    """
    calibrate_id = generate_calibrate_id(expansion, k)
    calibrate_log_dir = LOG_BASE / "calibrate" / MODEL_NAME / f"layer_{layer}"
    calibrate_log_dir.mkdir(parents=True, exist_ok=True)
    
    calibrate_args = [
        "--layer", str(layer),
        "--expansion", str(expansion),
        "--k", str(k),
    ]
    
    return submit_job(
        script_path=CALIBRATE_BATCH_SIZE_SAE_SCRIPT,
        args=calibrate_args,
        job_name=f"sae_{calibrate_id}",
        output_path=calibrate_log_dir / f"{calibrate_id}.out",
        error_path=calibrate_log_dir / f"{calibrate_id}.err",
        dry_run=dry_run,
    )


# =============================================================================
# Main Orchestration Logic
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SAE Training Orchestrator: Manage caching and training SLURM jobs"
    )
    
    # Required arguments
    parser.add_argument("--data_source", type=str, required=True,
                        choices=["nuplan", "nuplan-webdataset", "covla", "covla-webdataset"],
                        help="Data source to use. Suffix '-webdataset' uses WebDataset shards "
                             "(faster on NFS) instead of individual .pt files.")
    parser.add_argument("--layer", type=int, required=True,
                        help="Layer to extract activations from")
    
    # SAE arguments (only for training)
    parser.add_argument("--k", type=int, default=64,
                        help="Top-K sparsity (default: 64)")
    parser.add_argument("--expansion", type=int, default=16,
                        help="SAE expansion factor (default: 16)")
    
    # Caching arguments
    parser.add_argument("--num_videos", type=int, default=None,
                        help="Number of videos to use (default: dataset default)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for caching (default: 4)")
    parser.add_argument("--cache_seed", type=int, default=42,
                        help="Seed for caching noise (default: 42)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs (default: 2)")
    parser.add_argument("--sae_batch_size", type=int, default=None,
                        help="SAE batch size in tokens (default: from calibration)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Training random seed (default: 42)")
    
    # WebDataset options (only used when data_source ends with "-webdataset")
    parser.add_argument("--num_shards", type=int, default=None,
                        help="Number of shards to use (for partial training, e.g., 59 for half)")
    parser.add_argument("--shuffle_buffer", type=int, default=500000,
                        help="Shuffle buffer size for WebDataset (default: 500000 tokens). "
                             "Larger = better IID mixing. 500k uses ~1.5GB RAM.")
    
    # Orchestration control
    parser.add_argument("--rebuild_cache", action="store_true",
                        help="Force rebuild of activation cache (requires confirmation)")
    parser.add_argument("--cache_only", action="store_true",
                        help="Only run caching, skip training")
    parser.add_argument("--train_only", action="store_true",
                        help="Only run training, skip caching (assumes cache exists)")
    parser.add_argument("--skip_calibration", action="store_true",
                        help="Skip automatic batch size calibration (use --sae_batch_size directly)")
    parser.add_argument("--partition", type=str, default=DEFAULT_PARTITION,
                        help=f"SLURM partition for GPU type inference (default: {DEFAULT_PARTITION})")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="Skip confirmation prompts (for automation)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.cache_only and args.train_only:
        logger.error("Cannot specify both --cache_only and --train_only")
        sys.exit(1)
    
    # Check script existence
    if not CACHE_SCRIPT.exists():
        logger.error(f"Cache script not found: {CACHE_SCRIPT}")
        sys.exit(1)
    if not TRAIN_SCRIPT.exists():
        logger.error(f"Train script not found: {TRAIN_SCRIPT}")
        sys.exit(1)
    
    # Check if using WebDataset mode
    use_webdataset = is_webdataset_mode(args.data_source)
    base_data_source = get_base_data_source(args.data_source)
    
    # Get cache directory and check status
    cache_dir = get_cache_dir(args.data_source, args.layer)
    
    if use_webdataset:
        # For WebDataset, check if shards exist (not _meta.json).
        # Expected layout:
        #   <cache_dir>/train/layer_<L>-train-000000.tar, ...
        wds_train_dir = cache_dir / "train"
        shard_files = list(wds_train_dir.glob("layer_*-train-*.tar"))
        cache_complete = len(shard_files) > 0
        cache_info = {"train": None, "val": None}
        if cache_complete:
            cache_info["train"] = {"num_files": len(shard_files), "total_tokens": "N/A (WebDataset)"}
        partial_cache_exists = False  # WebDataset is all-or-nothing
    else:
        cache_complete = is_cache_complete(cache_dir)
        cache_info = get_cache_info(cache_dir)
        # Check if there's a partial cache (some files exist but not complete)
        train_cache_exists = (cache_dir / "train").exists() and any((cache_dir / "train").glob("batch_*.pt"))
        val_cache_exists = (cache_dir / "val").exists() and any((cache_dir / "val").glob("batch_*.pt"))
        partial_cache_exists = (train_cache_exists or val_cache_exists) and not cache_complete
    
    logger.info("=== SAE Training Orchestrator ===")
    logger.info(f"Data source: {args.data_source}")
    if use_webdataset:
        logger.info(f"  Base source: {base_data_source}")
        logger.info(f"  Format: WebDataset shards (optimized for NFS)")
        if args.num_shards:
            logger.info(f"  Shards to use: {args.num_shards}")
    logger.info(f"Layer: {args.layer}")
    logger.info(f"Cache directory: {cache_dir}")
    if use_webdataset:
        logger.info(f"WebDataset train directory: {wds_train_dir}")
        logger.info(f"Shards status: {'FOUND' if cache_complete else 'NOT FOUND'} ({len(shard_files) if cache_complete else 0} shards)")
    else:
        logger.info(f"Cache status: {'COMPLETE' if cache_complete else 'INCOMPLETE'}")
    
    # Show cache info if available (skip for WebDataset since we already showed shard count)
    if not use_webdataset:
        if cache_info["train"]:
            logger.info(f"  Train: {cache_info['train']['num_files']} files, {cache_info['train']['total_tokens']:,} tokens")
        if cache_info["val"]:
            logger.info(f"  Val: {cache_info['val']['num_files']} files, {cache_info['val']['total_tokens']:,} tokens")
    
    # === Confirmation prompts ===
    
    # Confirm rebuild_cache (destructive operation) - not applicable for WebDataset
    if args.rebuild_cache and not args.dry_run and not use_webdataset:
        if cache_complete or partial_cache_exists:
            logger.warning("--rebuild_cache will DELETE the existing cache!")
            if cache_info["train"]:
                logger.warning(f"  This will delete {cache_info['train']['num_files']} train files ({cache_info['train']['total_tokens']:,} tokens)")
            if cache_info["val"]:
                logger.warning(f"  This will delete {cache_info['val']['num_files']} val files ({cache_info['val']['total_tokens']:,} tokens)")
            
            if not args.yes:
                if not confirm("Are you sure you want to rebuild the cache?"):
                    logger.info("Aborted.")
                    sys.exit(0)
    
    # Confirm cache resume (resuming partial cache)
    if partial_cache_exists and not args.rebuild_cache and not args.train_only and not args.dry_run:
        logger.info("Found incomplete cache. Will resume from existing progress.")
        if cache_info["train"]:
            logger.info(f"  Train: {cache_info['train']['num_files']} files already cached")
        if cache_info["val"]:
            logger.info(f"  Val: {cache_info['val']['num_files']} files already cached")
        
        if not args.yes:
            if not confirm("Continue with cache resume?", default=True):
                logger.info("Aborted. Use --rebuild_cache to start fresh.")
                sys.exit(0)
    
    # Determine what to do
    need_cache = not cache_complete or args.rebuild_cache
    need_train = not args.cache_only
    skip_cache = args.train_only
    
    # In WebDataset mode, caching is handled by an offline conversion step.
    # If shards are missing, fail fast with an actionable message.
    if use_webdataset and not cache_complete and not args.dry_run:
        logger.error("WebDataset shards not found for this layer.")
        logger.error(f"  Expected train shards under: {cache_dir / 'train'}")
        logger.error("Run the conversion first (or use non-WebDataset data_source):")
        logger.error("  python sae/scripts/convert_to_webdataset.py")
        sys.exit(1)
    
    if skip_cache and not cache_complete:
        logger.error("--train_only specified but cache is incomplete")
        logger.error(f"  Train meta: {cache_dir / 'train' / '_meta.json'}")
        logger.error(f"  Val meta: {cache_dir / 'val' / '_meta.json'}")
        sys.exit(1)
    
    # ==========================================================================
    # Calibration Resource Check (Self-Discovery)
    # ==========================================================================
    calibrate_job_id = None
    use_auto_batch_size = False  # Will train job read batch size from resource file?
    
    # If user explicitly provides --sae_batch_size, skip calibration entirely
    if args.sae_batch_size is not None:
        logger.info(f"User-specified SAE batch size: {args.sae_batch_size:,} tokens")
        logger.info("Skipping calibration (explicit batch size provided)")
    elif args.skip_calibration:
        logger.info("Skipping calibration (--skip_calibration flag)")
        logger.info("Train job will use default batch size (4096)")
    elif need_train:
        # Need to determine batch size via calibration
        gpu_slug = get_gpu_slug_for_partition(args.partition)
        
        if gpu_slug is None:
            # Partition not in map - need to run calibration to discover GPU
            logger.info(f"Partition '{args.partition}' not in partition_map.json")
            logger.info("Will submit calibration job to discover GPU and find optimal batch size...")
            
            if not CALIBRATE_BATCH_SIZE_SAE_SCRIPT.exists():
                logger.error(f"Calibration script not found: {CALIBRATE_BATCH_SIZE_SAE_SCRIPT}")
                logger.error("Use --sae_batch_size to specify batch size manually")
                sys.exit(1)
            
            if not args.yes and not args.dry_run:
                if not confirm("Submit calibration job first?", default=True):
                    logger.info("Aborted. Use --sae_batch_size to specify batch size manually.")
                    sys.exit(0)
            
            calibrate_job_id = submit_calibration_job(
                layer=args.layer,
                expansion=args.expansion,
                k=args.k,
                dry_run=args.dry_run,
            )
            
            if calibrate_job_id:
                logger.info(f"Calibration job submitted: {calibrate_job_id}")
                logger.info("Train job will wait for calibration and read batch size from resource file")
                use_auto_batch_size = True
        else:
            # Check if hardware-specific calibration exists
            logger.info(f"Partition '{args.partition}' maps to GPU: {gpu_slug}")
            resource_path = get_resource_path(args.layer, args.expansion, args.k, gpu_slug)
            calibration = load_calibration_resource(resource_path)
            
            if calibration is None:
                logger.info(f"No calibration resource for GPU '{gpu_slug}'")
                logger.info(f"  Expected at: {resource_path}")
                logger.info("Will submit calibration job...")
                
                if not CALIBRATE_BATCH_SIZE_SAE_SCRIPT.exists():
                    logger.error(f"Calibration script not found: {CALIBRATE_BATCH_SIZE_SAE_SCRIPT}")
                    logger.error("Use --sae_batch_size to specify batch size manually")
                    sys.exit(1)
                
                if not args.yes and not args.dry_run:
                    if not confirm("Submit calibration job first?", default=True):
                        logger.info("Aborted. Use --sae_batch_size to specify batch size manually.")
                        sys.exit(0)
                
                calibrate_job_id = submit_calibration_job(
                    layer=args.layer,
                    expansion=args.expansion,
                    k=args.k,
                    dry_run=args.dry_run,
                )
                
                if calibrate_job_id:
                    logger.info(f"Calibration job submitted: {calibrate_job_id}")
                    logger.info("Train job will wait for calibration and read batch size from resource file")
                    use_auto_batch_size = True
            else:
                # Calibration already exists - train job will read it at runtime
                logger.info(f"Calibration resource found for GPU '{gpu_slug}'")
                logger.info(f"  Recommended batch size: {calibration['recommended_batch_size_tokens']:,} tokens")
                logger.info("Train job will read batch size from resource file at runtime")
                use_auto_batch_size = True
    
    cache_job_id = None
    train_job_id = None
    
    # For WebDataset mode, we don't need to cache - shards should already exist
    if use_webdataset and need_cache and not skip_cache:
        logger.warning("WebDataset mode does not support caching - shards must already exist")
        logger.warning("Skipping cache job submission")
        need_cache = False
        skip_cache = True  # Treat as if --train_only was specified
    
    # Submit cache job if needed (only for non-WebDataset mode)
    if need_cache and not skip_cache and not use_webdataset:
        cache_id = generate_cache_id(args.cache_seed)
        cache_log_dir = get_log_dir("sae_cache", args.data_source, args.layer)
        cache_log_dir.mkdir(parents=True, exist_ok=True)
        
        cache_args = [
            "--data_source", args.data_source,
            "--layer", str(args.layer),
            "--seed", str(args.cache_seed),
            "--batch_size", str(args.batch_size),
            "--run_name", cache_id,
        ]
        if args.num_videos:
            cache_args.extend(["--num_videos", str(args.num_videos)])
        if args.rebuild_cache:
            cache_args.append("--rebuild_cache")
        
        # Cache depends on calibration if calibration job was submitted
        cache_dependency = f"afterok:{calibrate_job_id}" if calibrate_job_id else None
        
        logger.info(f"Submitting cache job: {cache_id}")
        if cache_dependency:
            logger.info(f"  Dependency: {cache_dependency}")
        
        cache_job_id = submit_job(
            script_path=CACHE_SCRIPT,
            args=cache_args,
            job_name=f"sae_cache_{cache_id}",
            output_path=cache_log_dir / f"{cache_id}.out",
            error_path=cache_log_dir / f"{cache_id}.err",
            dependency=cache_dependency,
            dry_run=args.dry_run,
        )
    
    # Submit train job if needed
    if need_train:
        train_barcode = generate_train_barcode(args.expansion, args.k, args.seed)
        train_log_dir = get_log_dir("runs", args.data_source, args.layer)
        train_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine batch size argument: explicit value or "auto"
        if args.sae_batch_size is not None:
            batch_size_arg = str(args.sae_batch_size)
        elif use_auto_batch_size:
            batch_size_arg = "auto"
        else:
            batch_size_arg = "4096"  # Default fallback
        
        train_args = [
            "--data_source", args.data_source,
            "--layer", str(args.layer),
            "--k", str(args.k),
            "--expansion", str(args.expansion),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--sae_batch_size", batch_size_arg,
            "--seed", str(args.seed),
            "--barcode", train_barcode,
        ]
        if args.num_videos:
            train_args.extend(["--num_videos", str(args.num_videos)])
        
        # WebDataset options (only when using WebDataset format)
        if use_webdataset:
            train_args.extend(["--shuffle_buffer", str(args.shuffle_buffer)])
            if args.num_shards:
                train_args.extend(["--num_shards", str(args.num_shards)])
        
        # Add --train_only if cache job was submitted or skip_cache is set
        if cache_job_id or skip_cache:
            train_args.append("--train_only")
        
        # Build dependency chain: calibrate -> cache -> train
        dependencies = []
        if calibrate_job_id:
            dependencies.append(f"afterok:{calibrate_job_id}")
        if cache_job_id:
            dependencies.append(f"afterok:{cache_job_id}")
        
        dependency = ",".join(dependencies) if dependencies else None
        
        logger.info(f"Submitting train job: {train_barcode}")
        if dependency:
            logger.info(f"  Dependency: {dependency}")
        logger.info(f"  SAE batch size: {batch_size_arg}")
        
        train_job_id = submit_job(
            script_path=TRAIN_SCRIPT,
            args=train_args,
            job_name=f"sae_train_{train_barcode}",
            output_path=train_log_dir / f"{train_barcode}.out",
            error_path=train_log_dir / f"{train_barcode}.err",
            dependency=dependency,
            dry_run=args.dry_run,
        )
    
    # Summary
    logger.info("=== Summary ===")
    if calibrate_job_id:
        logger.info(f"Calibrate job: {calibrate_job_id}")
    if cache_job_id:
        logger.info(f"Cache job: {cache_job_id}")
    if train_job_id:
        logger.info(f"Train job: {train_job_id}")
    if not calibrate_job_id and not cache_job_id and not train_job_id:
        logger.info("No jobs submitted (dry run or nothing to do)")
    
    if args.sae_batch_size is not None:
        logger.info(f"SAE batch size: {args.sae_batch_size:,} tokens (user-specified)")
    elif use_auto_batch_size:
        logger.info("SAE batch size: auto (will read from calibration resource at runtime)")
    else:
        logger.info("SAE batch size: 4096 tokens (default)")
    
    logger.info("Monitor jobs with: squeue -u $USER")
    

if __name__ == "__main__":
    main()
