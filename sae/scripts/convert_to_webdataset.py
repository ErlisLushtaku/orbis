#!/usr/bin/env python3
"""Convert PyTorch .pt cache files to WebDataset sharded tar archives.

This script converts a directory of batch_*.pt files into WebDataset format
for optimized IO performance on NFS clusters. Files are read as raw binary
and written directly to shards without re-serialization.

Usage:
    python convert_to_webdataset.py

Configuration is done via constants at the top of the file.
"""

import glob
import logging
import os
import sys
import time

from webdataset import ShardWriter

# =============================================================================
# Configuration
# =============================================================================
SRC_DIR = "/work/dlclarge2/lushtake-thesis/orbis/logs_sae/sae_cache/nuplan/orbis_288x512/layer_22/val"
DST_PATTERN = "/data/lmbraid19/lushtake/sae_wds/layer_22-val-%06d.tar"
FILE_PATTERN = "batch_*.pt"
SHARD_MAX_SIZE = 10e9  # 10 GB per shard
PROGRESS_INTERVAL = 100  # Print progress every N files

# =============================================================================
# Logging Setup
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Convert .pt files to WebDataset sharded tar archives."""
    # Discover and sort files for reproducible sharding order
    pattern = os.path.join(SRC_DIR, FILE_PATTERN)
    logger.info(f"Searching for files matching: {pattern}")
    
    files = sorted(glob.glob(pattern))
    num_files = len(files)
    
    if num_files == 0:
        logger.error(f"No files found matching pattern: {pattern}")
        sys.exit(1)
    
    logger.info(f"Found {num_files} files to convert")
    
    # Calculate expected number of shards (approximate)
    sample_size = os.path.getsize(files[0])
    estimated_shards = (num_files * sample_size) / SHARD_MAX_SIZE
    logger.info(f"Sample file size: {sample_size / 1e6:.2f} MB")
    logger.info(f"Estimated number of shards: ~{int(estimated_shards) + 1}")
    
    # Ensure output directory exists
    dst_dir = os.path.dirname(DST_PATTERN)
    os.makedirs(dst_dir, exist_ok=True)
    logger.info(f"Output directory: {dst_dir}")
    
    # Convert files to WebDataset shards
    start_time = time.time()
    bytes_written = 0
    
    with ShardWriter(DST_PATTERN, maxsize=SHARD_MAX_SIZE, verbose=1) as sink:
        for i, fpath in enumerate(files):
            # Progress logging
            if i % PROGRESS_INTERVAL == 0:
                elapsed = time.time() - start_time
                rate = bytes_written / elapsed / 1e6 if elapsed > 0 else 0
                logger.info(
                    f"Progress: {i}/{num_files} files ({i/num_files*100:.1f}%) | "
                    f"Rate: {rate:.1f} MB/s | "
                    f"File: {os.path.basename(fpath)}"
                )
            
            # Read file as raw binary (passthrough without re-encoding)
            with open(fpath, "rb") as f:
                data = f.read()
            
            bytes_written += len(data)
            
            # Use basename without extension as the sample key
            key = os.path.splitext(os.path.basename(fpath))[0]
            
            # Write to shard - bytes pass through encoder unchanged
            sink.write({"__key__": key, "pt": data})
    
    # Final statistics
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Conversion complete!")
    logger.info(f"Total files converted: {num_files}")
    logger.info(f"Total data written: {bytes_written / 1e9:.2f} GB")
    logger.info(f"Total time: {elapsed / 60:.1f} minutes")
    logger.info(f"Average rate: {bytes_written / elapsed / 1e6:.1f} MB/s")
    logger.info(f"Output pattern: {DST_PATTERN}")


if __name__ == "__main__":
    main()
