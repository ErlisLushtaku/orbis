#!/usr/bin/env python3
"""
SAE Training Orchestrator: Manages caching and training as separate SLURM jobs.

This script checks cache status and submits appropriate SLURM jobs:
1. If cache is incomplete: Submit cache job, then train job with dependency
2. If cache is complete: Submit train job directly

Usage:
    # Full pipeline (cache + train)
    python sae/scripts/launch.py --data_source nuplan --layer 22 --k 64 --expansion 16
    
    # Force cache rebuild (with confirmation)
    python sae/scripts/launch.py --data_source covla --layer 12 --rebuild_cache
    
    # Cache only (no training)
    python sae/scripts/launch.py --data_source nuplan --layer 22 --cache_only
    
    # Train only (assume cache exists)
    python sae/scripts/launch.py --data_source covla --layer 12 --train_only
    
    # Dry run (print commands without executing)
    python sae/scripts/launch.py --data_source nuplan --layer 22 --dry_run
    
    # Skip confirmation prompts (for automation)
    python sae/scripts/launch.py --data_source nuplan --layer 22 --yes
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# =============================================================================
# Configuration
# =============================================================================

ORBIS_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "orbis_288x512"

# Cache and log base directories
CACHE_BASE = ORBIS_ROOT / "logs_sae" / "sae_cache"
LOG_BASE = ORBIS_ROOT / "sae" / "slurm" / "logs"

# Unified script paths (parametric by --data_source)
CACHE_SCRIPT = ORBIS_ROOT / "sae" / "slurm" / "cache.sh"
TRAIN_SCRIPT = ORBIS_ROOT / "sae" / "slurm" / "train.sh"


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
        print("\nAborted.")
        sys.exit(1)


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
        return LOG_BASE / data_source / MODEL_NAME / f"layer_{layer}" / "train"


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
        print(f"[DRY RUN] {' '.join(cmd)}")
        return ""
    
    print(f"[SUBMIT] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[ERROR] sbatch failed: {result.stderr}")
        sys.exit(1)
    
    # Parse job ID from output like "Submitted batch job 12345"
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if match:
        job_id = match.group(1)
        print(f"[OK] Job submitted: {job_id}")
        return job_id
    else:
        print(f"[WARNING] Could not parse job ID from: {result.stdout}")
        return ""


# =============================================================================
# Main Orchestration Logic
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SAE Training Orchestrator: Manage caching and training SLURM jobs"
    )
    
    # Required arguments
    parser.add_argument("--data_source", type=str, required=True,
                        choices=["nuplan", "covla"],
                        help="Data source to use")
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
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--sae_batch_mult", type=int, default=1024,
                        help="SAE batch multiplier (default: 1024)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Training random seed (default: 42)")
    
    # Orchestration control
    parser.add_argument("--rebuild_cache", action="store_true",
                        help="Force rebuild of activation cache (requires confirmation)")
    parser.add_argument("--cache_only", action="store_true",
                        help="Only run caching, skip training")
    parser.add_argument("--train_only", action="store_true",
                        help="Only run training, skip caching (assumes cache exists)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="Skip confirmation prompts (for automation)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.cache_only and args.train_only:
        print("[ERROR] Cannot specify both --cache_only and --train_only")
        sys.exit(1)
    
    # Check script existence
    if not CACHE_SCRIPT.exists():
        print(f"[ERROR] Cache script not found: {CACHE_SCRIPT}")
        sys.exit(1)
    if not TRAIN_SCRIPT.exists():
        print(f"[ERROR] Train script not found: {TRAIN_SCRIPT}")
        sys.exit(1)
    
    # Get cache directory and check status
    cache_dir = get_cache_dir(args.data_source, args.layer)
    cache_complete = is_cache_complete(cache_dir)
    cache_info = get_cache_info(cache_dir)
    
    # Check if there's a partial cache (some files exist but not complete)
    train_cache_exists = (cache_dir / "train").exists() and any((cache_dir / "train").glob("batch_*.pt"))
    val_cache_exists = (cache_dir / "val").exists() and any((cache_dir / "val").glob("batch_*.pt"))
    partial_cache_exists = (train_cache_exists or val_cache_exists) and not cache_complete
    
    print(f"=== SAE Training Orchestrator ===")
    print(f"Data source: {args.data_source}")
    print(f"Layer: {args.layer}")
    print(f"Cache directory: {cache_dir}")
    print(f"Cache status: {'COMPLETE' if cache_complete else 'INCOMPLETE'}")
    
    # Show cache info if available
    if cache_info["train"]:
        print(f"  Train: {cache_info['train']['num_files']} files, {cache_info['train']['total_tokens']:,} tokens")
    if cache_info["val"]:
        print(f"  Val: {cache_info['val']['num_files']} files, {cache_info['val']['total_tokens']:,} tokens")
    print()
    
    # === Confirmation prompts ===
    
    # Confirm rebuild_cache (destructive operation)
    if args.rebuild_cache and not args.dry_run:
        if cache_complete or partial_cache_exists:
            print("[WARNING] --rebuild_cache will DELETE the existing cache!")
            if cache_info["train"]:
                print(f"  This will delete {cache_info['train']['num_files']} train files ({cache_info['train']['total_tokens']:,} tokens)")
            if cache_info["val"]:
                print(f"  This will delete {cache_info['val']['num_files']} val files ({cache_info['val']['total_tokens']:,} tokens)")
            
            if not args.yes:
                if not confirm("Are you sure you want to rebuild the cache?"):
                    print("Aborted.")
                    sys.exit(0)
            print()
    
    # Confirm cache resume (resuming partial cache)
    if partial_cache_exists and not args.rebuild_cache and not args.train_only and not args.dry_run:
        print("[INFO] Found incomplete cache. Will resume from existing progress.")
        if cache_info["train"]:
            print(f"  Train: {cache_info['train']['num_files']} files already cached")
        if cache_info["val"]:
            print(f"  Val: {cache_info['val']['num_files']} files already cached")
        
        if not args.yes:
            if not confirm("Continue with cache resume?", default=True):
                print("Aborted. Use --rebuild_cache to start fresh.")
                sys.exit(0)
        print()
    
    # Determine what to do
    need_cache = not cache_complete or args.rebuild_cache
    need_train = not args.cache_only
    skip_cache = args.train_only
    
    if skip_cache and not cache_complete:
        print("[ERROR] --train_only specified but cache is incomplete")
        print(f"  Train meta: {cache_dir / 'train' / '_meta.json'}")
        print(f"  Val meta: {cache_dir / 'val' / '_meta.json'}")
        sys.exit(1)
    
    cache_job_id = None
    train_job_id = None
    
    # Submit cache job if needed
    if need_cache and not skip_cache:
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
        
        print(f"[CACHE] Submitting cache job: {cache_id}")
        cache_job_id = submit_job(
            script_path=CACHE_SCRIPT,
            args=cache_args,
            job_name=f"sae_cache_{cache_id}",
            output_path=cache_log_dir / f"{cache_id}.out",
            error_path=cache_log_dir / f"{cache_id}.err",
            dry_run=args.dry_run,
        )
        print()
    
    # Submit train job if needed
    if need_train:
        train_barcode = generate_train_barcode(args.expansion, args.k, args.seed)
        train_log_dir = get_log_dir("runs", args.data_source, args.layer)
        train_log_dir.mkdir(parents=True, exist_ok=True)
        
        train_args = [
            "--data_source", args.data_source,
            "--layer", str(args.layer),
            "--k", str(args.k),
            "--expansion", str(args.expansion),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--sae_batch_mult", str(args.sae_batch_mult),
            "--seed", str(args.seed),
            "--barcode", train_barcode,
        ]
        if args.num_videos:
            train_args.extend(["--num_videos", str(args.num_videos)])
        
        # Add --train_only if cache job was submitted or skip_cache is set
        if cache_job_id or skip_cache:
            train_args.append("--train_only")
        
        # Set dependency if cache job was submitted
        dependency = f"afterok:{cache_job_id}" if cache_job_id else None
        
        print(f"[TRAIN] Submitting train job: {train_barcode}")
        if dependency:
            print(f"  Dependency: {dependency}")
        
        train_job_id = submit_job(
            script_path=TRAIN_SCRIPT,
            args=train_args,
            job_name=f"sae_train_{train_barcode}",
            output_path=train_log_dir / f"{train_barcode}.out",
            error_path=train_log_dir / f"{train_barcode}.err",
            dependency=dependency,
            dry_run=args.dry_run,
        )
        print()
    
    # Summary
    print("=== Summary ===")
    if cache_job_id:
        print(f"Cache job: {cache_job_id}")
    if train_job_id:
        print(f"Train job: {train_job_id}")
    if not cache_job_id and not train_job_id:
        print("No jobs submitted (dry run or nothing to do)")
    
    print()
    print("Monitor jobs with: squeue -u $USER")
    

if __name__ == "__main__":
    main()
