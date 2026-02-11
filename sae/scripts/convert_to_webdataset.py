#!/usr/bin/env python3
"""Convert PyTorch .pt cache files to WebDataset sharded tar archives.

This script converts directories of batch_*.pt files into WebDataset format
for optimized IO performance on NFS clusters. Supports converting multiple
splits (e.g., train and val) in a single invocation.

Shard naming follows the convention expected by train_sae.py:
    {dst_dir}/{split}/layer_{layer}-{split}-%06d.tar

Usage:
    python convert_to_webdataset.py --src_dir /path/to/cache --dst_dir /path/to/wds --layer 12
    python convert_to_webdataset.py --src_dir /path/to/cache --dst_dir /path/to/wds --layer 22 --splits val
"""

import argparse
import glob
import logging
import os
import sys
import time

from webdataset import ShardWriter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

SHARD_MAX_SIZE_DEFAULT = 10e9  # 10 GB per shard
FILE_PATTERN = "batch_*.pt"
PROGRESS_INTERVAL = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert .pt cache files to WebDataset sharded tar archives.",
    )
    parser.add_argument(
        "--src_dir",
        type=str,
        required=True,
        help="Base directory containing split subdirectories with .pt files",
    )
    parser.add_argument(
        "--dst_dir",
        type=str,
        required=True,
        help="Base output directory for WebDataset shards",
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer number (used in shard naming: layer_{N}-{split}-%06d.tar)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val",
        help="Comma-separated list of splits to convert (default: train,val)",
    )
    parser.add_argument(
        "--shard_max_size",
        type=float,
        default=SHARD_MAX_SIZE_DEFAULT,
        help=f"Maximum shard size in bytes (default: {SHARD_MAX_SIZE_DEFAULT:.0e})",
    )
    return parser.parse_args()


def convert_split(
    src_dir: str,
    dst_pattern: str,
    shard_max_size: float,
) -> None:
    """Convert .pt files from a single split directory to WebDataset shards."""
    pattern = os.path.join(src_dir, FILE_PATTERN)
    logger.info(f"Searching for files matching: {pattern}")

    files = sorted(glob.glob(pattern))
    num_files = len(files)

    if num_files == 0:
        logger.error(f"No files found matching pattern: {pattern}")
        sys.exit(1)

    logger.info(f"Found {num_files} files to convert")

    sample_size = os.path.getsize(files[0])
    estimated_shards = (num_files * sample_size) / shard_max_size
    logger.info(f"Sample file size: {sample_size / 1e6:.2f} MB")
    logger.info(f"Estimated number of shards: ~{int(estimated_shards) + 1}")

    dst_dir = os.path.dirname(dst_pattern)
    os.makedirs(dst_dir, exist_ok=True)
    logger.info(f"Output pattern: {dst_pattern}")

    start_time = time.time()
    bytes_written = 0

    with ShardWriter(dst_pattern, maxsize=shard_max_size, verbose=1) as sink:
        for i, fpath in enumerate(files):
            if i % PROGRESS_INTERVAL == 0:
                elapsed = time.time() - start_time
                rate = bytes_written / elapsed / 1e6 if elapsed > 0 else 0
                logger.info(
                    f"Progress: {i}/{num_files} files ({i / num_files * 100:.1f}%) | "
                    f"Rate: {rate:.1f} MB/s | "
                    f"File: {os.path.basename(fpath)}"
                )

            with open(fpath, "rb") as f:
                data = f.read()

            bytes_written += len(data)
            key = os.path.splitext(os.path.basename(fpath))[0]
            sink.write({"__key__": key, "pt": data})

    elapsed = time.time() - start_time
    logger.info(f"Split complete: {num_files} files, {bytes_written / 1e9:.2f} GB, "
                f"{elapsed / 60:.1f} min, {bytes_written / elapsed / 1e6:.1f} MB/s")


def main() -> None:
    args = parse_args()
    splits = [s.strip() for s in args.splits.split(",")]

    logger.info(f"Source directory: {args.src_dir}")
    logger.info(f"Destination directory: {args.dst_dir}")
    logger.info(f"Layer: {args.layer}")
    logger.info(f"Splits: {splits}")
    logger.info(f"Shard max size: {args.shard_max_size / 1e9:.1f} GB")

    for split in splits:
        split_src = os.path.join(args.src_dir, split)
        if not os.path.isdir(split_src):
            logger.error(f"Source split directory does not exist: {split_src}")
            sys.exit(1)

        split_dst_dir = os.path.join(args.dst_dir, split)
        dst_pattern = os.path.join(
            split_dst_dir, f"layer_{args.layer}-{split}-%06d.tar"
        )

        logger.info("=" * 60)
        logger.info(f"Converting split: {split}")
        logger.info(f"  Source: {split_src}")
        logger.info(f"  Destination pattern: {dst_pattern}")
        logger.info("=" * 60)

        convert_split(
            src_dir=split_src,
            dst_pattern=dst_pattern,
            shard_max_size=args.shard_max_size,
        )

    logger.info("=" * 60)
    logger.info("All splits converted successfully!")


if __name__ == "__main__":
    main()
