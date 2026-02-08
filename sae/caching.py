"""
Activation caching utilities for SAE training.

Adapted from semantic_stage2.py - provides efficient caching of ST-Transformer
activations to avoid repeated forward passes through the frozen Orbis model.

Supports:
- Stop/Resume: If caching is interrupted, it will resume from the last completed batch
- Dataset Extension: If you increase num_videos, it will cache only the new batches
- WebDataset: Efficient streaming from sharded tar archives for NFS performance
"""

import io
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from einops import rearrange

from .activation_hooks import ActivationExtractor
from .utils.logging_utils import get_logger, format_duration, format_bytes, format_throughput

logger = get_logger(__name__)


# =============================================================================
# Cache Resume Info
# =============================================================================

@dataclass
class CacheResumeInfo:
    """Information about cache state for zero-cost resume.
    
    Attributes:
        start_batch_idx: Next batch index to process (for file naming)
        start_sample_idx: Next sample index to process (for Subset creation)
        total_tokens: Number of tokens already cached
        hidden_dim: Hidden dimension from cached files (None if unknown)
        is_complete: True if cache is complete and nothing to do
        valid_files: List of existing valid cache files
    """
    start_batch_idx: int
    start_sample_idx: int
    total_tokens: int
    hidden_dim: Optional[int]
    is_complete: bool
    valid_files: List[Path]


def get_cache_resume_info(
    cache_dir: Path,
    batch_size: int,
    total_samples: int,
    rebuild: bool = False,
) -> CacheResumeInfo:
    """
    Check cache status and return resume information.
    
    This function should be called BEFORE creating DataLoaders to enable
    zero-cost resume via Subset.
    
    Args:
        cache_dir: Directory containing cached activations
        batch_size: Batch size used for caching (must match original)
        total_samples: Total number of samples in the dataset
        rebuild: If True, ignore existing cache and return fresh start
        
    Returns:
        CacheResumeInfo with all information needed for resume
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_file = cache_dir / "_meta.json"
    progress_file = cache_dir / "_progress.json"
    
    # Calculate expected total batches
    total_batches = (total_samples + batch_size - 1) // batch_size  # ceil division
    
    # Handle rebuild flag - return fresh start
    if rebuild:
        # Clean existing cache
        for path in cache_dir.glob("*.pt"):
            path.unlink()
        if meta_file.exists():
            meta_file.unlink()
        if progress_file.exists():
            progress_file.unlink()
        logger.info(f"Rebuild requested. Cleared {cache_dir}")
        return CacheResumeInfo(
            start_batch_idx=0,
            start_sample_idx=0,
            total_tokens=0,
            hidden_dim=None,
            is_complete=False,
            valid_files=[],
        )
    
    # Check for existing files
    existing_files = sorted(cache_dir.glob("batch_*.pt"))
    num_existing = len(existing_files)
    
    # Handle empty cache (fresh start)
    if not existing_files:
        logger.info(f"No existing cache at {cache_dir}")
        return CacheResumeInfo(
            start_batch_idx=0,
            start_sample_idx=0,
            total_tokens=0,
            hidden_dim=None,
            is_complete=False,
            valid_files=[],
        )
    
    # Check if cache is complete according to metadata
    if meta_file.exists():
        try:
            with meta_file.open("r") as f:
                meta = json.load(f)
            
            # Safety check: verify batch_size matches
            cached_batch_size = meta.get("batch_size")
            if cached_batch_size is not None and cached_batch_size != batch_size:
                raise ValueError(
                    f"[cache] BATCH SIZE MISMATCH: Cache was created with batch_size={cached_batch_size}, "
                    f"but current batch_size={batch_size}. This would corrupt the cache!\n"
                    f"Options: 1) Use --rebuild_cache to start fresh, or "
                    f"2) Use the same batch_size as the original run."
                )
            
            # Check if complete
            if meta["num_files"] == num_existing and num_existing >= total_batches:
                logger.info(f"Using complete existing cache from {cache_dir} ({num_existing} files)")
                return CacheResumeInfo(
                    start_batch_idx=num_existing,
                    start_sample_idx=total_samples,  # All samples processed
                    total_tokens=meta["total_tokens"],
                    hidden_dim=meta.get("hidden_dim"),
                    is_complete=True,
                    valid_files=existing_files,
                )
        except json.JSONDecodeError:
            logger.warning("Metadata corrupted, re-scanning files.")
    
    # === GAP DETECTION & RESUME LOGIC ===
    logger.info(f"Found {num_existing} existing files. Verifying integrity...")
    
    # Get actual indices present on disk
    actual_indices = sorted([int(f.stem.split('_')[1]) for f in existing_files])
    
    if not actual_indices:
        start_batch_idx = 0
        valid_files: List[Path] = []
    else:
        # Check for gaps - we expect indices to be [0, 1, 2, ... N-1]
        first_gap = -1
        for i, idx in enumerate(actual_indices):
            if idx != i:
                first_gap = i
                break
        
        if first_gap == -1:
            # No gaps found - resume from next index
            start_batch_idx = len(actual_indices)
            valid_files = existing_files
        else:
            # Gap detected!
            logger.warning(f"Gap detected at batch index {first_gap}. Resuming from there.")
            start_batch_idx = first_gap
            valid_files = [f for f in existing_files if int(f.stem.split('_')[1]) < first_gap]
    
    # Recover token counts and hidden_dim
    total_tokens = 0
    hidden_dim: Optional[int] = None
    
    if valid_files:
        num_files = len(valid_files)
        
        # Try reading from progress file first (fast path)
        if progress_file.exists():
            try:
                with progress_file.open("r") as f:
                    progress = json.load(f)
                
                # Verify progress file matches disk state
                progress_num_files = progress.get("num_files", 0)
                if progress_num_files == num_files:
                    total_tokens = progress["total_tokens"]
                    hidden_dim = progress.get("hidden_dim")
                    logger.info(f"Loaded progress: {total_tokens:,} tokens from {num_files} files")
                else:
                    logger.info(f"Progress file stale ({progress_num_files} vs {num_files} files), re-scanning...")
                    progress_file.unlink()  # Remove stale progress
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Progress file corrupted ({e}), re-scanning...")
                progress_file.unlink()
        
        # If we still need to recover tokens (no valid progress file), scan all files
        if total_tokens == 0:
            logger.info(f"Recovering token counts from {num_files} files...")
            for f in tqdm(valid_files, desc="Scanning cache", leave=False):
                try:
                    data = torch.load(f, map_location="cpu", weights_only=True)
                    if isinstance(data, dict):
                        acts = data["activations"]
                    else:
                        acts = data
                    total_tokens += acts.shape[0]
                    if hidden_dim is None:
                        hidden_dim = acts.shape[-1]
                except Exception as e:
                    logger.warning(f"Error reading {f}: {e}. Truncating cache here.")
                    corrupt_idx = int(f.stem.split('_')[1])
                    start_batch_idx = corrupt_idx
                    valid_files = [p for p in valid_files if int(p.stem.split('_')[1]) < corrupt_idx]
                    # Recount tokens for valid files only
                    total_tokens = 0
                    for valid_f in valid_files:
                        data = torch.load(valid_f, map_location="cpu", weights_only=True)
                        if isinstance(data, dict):
                            total_tokens += data["activations"].shape[0]
                        else:
                            total_tokens += data.shape[0]
                    break
    
    # Calculate start sample index
    start_sample_idx = start_batch_idx * batch_size
    
    # Check if we're already done (extension case where cache >= needed)
    is_complete = start_batch_idx >= total_batches
    
    if is_complete:
        logger.info(f"Cache covers all {total_batches} batches. Nothing to do.")
    else:
        logger.info(f"Will resume from batch {start_batch_idx}/{total_batches} "
                    f"(sample {start_sample_idx}/{total_samples}, {total_tokens:,} tokens cached)")
    
    return CacheResumeInfo(
        start_batch_idx=start_batch_idx,
        start_sample_idx=start_sample_idx,
        total_tokens=total_tokens,
        hidden_dim=hidden_dim,
        is_complete=is_complete,
        valid_files=valid_files,
    )


CACHE_DTYPE_MAP: Dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_cache_dtype(name: str) -> torch.dtype:
    """Convert string dtype name to torch.dtype."""
    key = name.lower()
    if key not in CACHE_DTYPE_MAP:
        valid = ", ".join(sorted(CACHE_DTYPE_MAP))
        raise ValueError(f"Unsupported cache dtype '{name}'. Valid options: {valid}")
    return CACHE_DTYPE_MAP[key]


class CachedActivationDataset(Dataset):
    """Dataset that serves cached activation tensors from disk."""
    
    def __init__(self, files: Sequence[Path]) -> None:
        super().__init__()
        self.files: List[Path] = list(files)
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load activation tensor from disk."""
        item = torch.load(self.files[idx], weights_only=True)
        if isinstance(item, dict):
            return item["activations"]
        return item


class StreamingActivationDataset(Dataset):
    """
    Dataset that concatenates multiple cached activation files.
    Handles variable-sized files (last batch may have fewer tokens).
    """
    
    def __init__(
        self, 
        files: Sequence[Path],
        total_tokens: Optional[int] = None,
        output_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.files: List[Path] = list(files)
        self.output_dtype = output_dtype
        
        # Build cumulative index for variable-sized files
        self._file_offsets: List[int] = [0]  # Start offset for each file
        self._file_sizes: List[int] = []
        
        if total_tokens is not None:
            # If total_tokens provided, assume uniform sizes except possibly last
            sample = torch.load(self.files[0], weights_only=True)
            if isinstance(sample, dict):
                sample = sample["activations"]
            tokens_first = sample.shape[0]
            
            # Calculate sizes: all files have tokens_first except possibly last
            remaining = total_tokens
            for i in range(len(files)):
                size = min(tokens_first, remaining)
                self._file_sizes.append(size)
                self._file_offsets.append(self._file_offsets[-1] + size)
                remaining -= size
            self._total_tokens = total_tokens
        else:
            # Scan all files to get exact sizes (slower but accurate)
            for f in self.files:
                data = torch.load(f, weights_only=True)
                if isinstance(data, dict):
                    data = data["activations"]
                size = data.shape[0]
                self._file_sizes.append(size)
                self._file_offsets.append(self._file_offsets[-1] + size)
            self._total_tokens = self._file_offsets[-1]
        
        self._cached_file_idx = -1
        self._cached_data = None
    
    def __len__(self) -> int:
        return self._total_tokens
    
    def _find_file(self, idx: int) -> Tuple[int, int]:
        """Find which file contains the given global index."""
        # Binary search for the file
        lo, hi = 0, len(self.files) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._file_offsets[mid] <= idx:
                lo = mid
            else:
                hi = mid - 1
        file_idx = lo
        token_idx = idx - self._file_offsets[file_idx]
        return file_idx, token_idx
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a single activation vector."""
        file_idx, token_idx = self._find_file(idx)
        
        # Cache file to avoid repeated loads
        if file_idx != self._cached_file_idx:
            data = torch.load(self.files[file_idx], weights_only=True)
            if isinstance(data, dict):
                data = data["activations"]
            # Convert to output dtype (typically float32 for training)
            self._cached_data = data.to(dtype=self.output_dtype)
            self._cached_file_idx = file_idx
        
        return self._cached_data[token_idx]


@torch.inference_mode()
def prepare_activation_cache(
    model: nn.Module,
    dataloader: DataLoader,
    cache_dir: Path,
    layer_idx: int,
    device: torch.device,
    dtype: torch.dtype,
    dtype_name: str,
    t_noise: float = 0.0,
    frame_rate: int = 5,
    show_progress: bool = True,
    # Resume parameters (from CacheResumeInfo)
    start_batch_idx: int = 0,
    existing_tokens: int = 0,
    existing_files: Optional[List[Path]] = None,
    existing_hidden_dim: Optional[int] = None,
    # Reproducibility parameters (stored in _meta.json)
    seed: int = 42,
    orbis_exp_dir: Optional[str] = None,
    data_source: Optional[str] = None,
    data_dir: Optional[str] = None,
    num_videos: Optional[int] = None,
    num_frames: int = 6,
    val_split: float = 0.1,
    source_size: Optional[Tuple[int, int]] = None,
    input_size: Tuple[int, int] = (288, 512),
    stored_frame_rate: Optional[int] = None,
) -> List[Path]:
    """
    Precompute and store ST-Transformer activations for SAE training.
    
    This function assumes the dataloader has already been subsetted for resume
    (via get_cache_resume_info + Subset). It uses enumerate with start offset
    to ensure correct file naming.
    
    Args:
        model: Orbis world model (will be put in eval mode)
        dataloader: DataLoader yielding batches of images (may be subsetted for resume)
        cache_dir: Directory to store cached activations
        layer_idx: Which transformer block to extract from
        device: Device to run inference on
        dtype: Storage dtype for cached activations
        dtype_name: String name of dtype (for metadata)
        t_noise: Noise timestep for denoising
        frame_rate: Frame rate conditioning value
        show_progress: Whether to show progress bar
        start_batch_idx: Batch index offset for file naming (from CacheResumeInfo)
        existing_tokens: Number of tokens already cached (from CacheResumeInfo)
        existing_files: List of existing cache files (from CacheResumeInfo)
        existing_hidden_dim: Hidden dim from existing files (from CacheResumeInfo)
        seed: Random seed for noise generation (for reproducibility)
        orbis_exp_dir: Path to Orbis experiment directory (for metadata)
        data_source: Data source type ("covla", "nuplan", etc.)
        data_dir: Path to source data directory
        num_videos: Number of videos used
        num_frames: Number of frames per clip
        val_split: Validation split fraction
        source_size: Original video resolution (H, W) before transforms
        input_size: Target resolution (H, W) after transforms
        stored_frame_rate: Native FPS of source videos
        
    Returns:
        List of paths to ALL cached activation files (existing + new)
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_file = cache_dir / "_meta.json"
    progress_file = cache_dir / "_progress.json"
    
    # Set seed for reproducible noise generation
    torch.manual_seed(seed)
    
    # Get batch_size from dataloader for metadata
    batch_size = dataloader.batch_size
    
    # Initialize from existing state
    saved_paths: List[Path] = list(existing_files) if existing_files else []
    total_tokens = existing_tokens
    hidden_dim = existing_hidden_dim
    
    # Calculate total batches for progress bar
    num_new_batches = len(dataloader)
    total_batches = start_batch_idx + num_new_batches
    
    # Early return if nothing to do
    if num_new_batches == 0:
        logger.info("Nothing to cache (dataloader is empty)")
        return saved_paths
    
    logger.info(f"Caching {num_new_batches} batches "
                f"(batch {start_batch_idx} to {total_batches - 1}) to {cache_dir}")
    
    # Setup model and extractor
    model.eval()
    extractor = ActivationExtractor(model, layer_idx=layer_idx, flatten_spatial=True)
    
    # Setup progress bar
    pbar = None
    if show_progress:
        pbar = tqdm(
            total=total_batches,
            initial=start_batch_idx,
            desc="Caching activations",
            unit="batch",
        )
    
    # Process batches with correct index offset for file naming
    for batch_idx, batch in enumerate(dataloader, start=start_batch_idx):
        # Handle different batch formats
        if isinstance(batch, (list, tuple)):
            imgs = batch[0]
        elif isinstance(batch, dict):
            imgs = batch.get('images', batch.get('image'))
        else:
            imgs = batch
        
        imgs = imgs.to(device, non_blocking=True)
        
        # Encode frames through tokenizer
        x = model.encode_frames(imgs)
        
        # Prepare for denoising step
        b = x.shape[0]
        t = torch.full((b,), t_noise, device=device)
        
        # Always add noise to match training distribution (sigma_min at t=0)
        target_t, _ = model.add_noise(x, t)
        
        # Ensure correct shape for vit input
        if target_t.dim() == 4:
            target_t = target_t.unsqueeze(1)
        
        # Run through transformer with activation capture
        fr = torch.full((b,), frame_rate, device=device)
        
        with extractor.capture():
            # Context is previous frames (if multi-frame)
            if target_t.shape[1] > 1:
                context = target_t[:, :-1]
                target = target_t[:, -1:]
            else:
                context = None
                target = target_t
            
            _ = model.vit(target, context, t, frame_rate=fr)
        
        # Get flattened activations: (batch * frames * spatial, hidden_dim)
        activations = extractor.get_activations()
        activations = activations.to(dtype=dtype).cpu()
        
        if hidden_dim is None:
            hidden_dim = activations.shape[-1]
        
        # Save to disk with correct batch index
        path = cache_dir / f"batch_{batch_idx:06d}.pt"
        torch.save({"activations": activations}, path)
        saved_paths.append(path)
        total_tokens += activations.shape[0]
        
        # Save progress periodically (every 100 batches) for fast resume
        if (batch_idx + 1) % 100 == 0:
            progress = {
                "num_files": len(saved_paths),
                "total_tokens": total_tokens,
                "hidden_dim": hidden_dim,
                "last_batch_idx": batch_idx,
                "seed": seed,  # Include seed for resume
            }
            with progress_file.open("w", encoding="utf-8") as f:
                json.dump(progress, f)
        
        # Update progress bar
        if pbar is not None:
            pbar.update(1)
    
    if pbar is not None:
        pbar.close()
    
    # Save final metadata with full reproducibility config (atomic write)
    meta = {
        # Cache statistics
        "num_files": len(saved_paths),
        "total_tokens": total_tokens,
        "hidden_dim": hidden_dim,
        "dtype": dtype_name,
        "batch_size": batch_size,
        # Model config
        "layer_idx": layer_idx,
        "t_noise": t_noise,
        "frame_rate": frame_rate,
        "orbis_exp_dir": orbis_exp_dir,
        # Data config
        "data_source": data_source,
        "data_dir": data_dir,
        "num_videos": num_videos,
        "num_frames": num_frames,
        "val_split": val_split,
        "source_size": list(source_size) if source_size else None,
        "input_size": list(input_size) if input_size else None,
        "stored_frame_rate": stored_frame_rate,
        # Reproducibility
        "seed": seed,
    }
    # Write atomically using temp file + os.replace() for crash safety
    fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp_path, meta_file)
    except Exception:
        # Clean up temp file on error
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    
    # Remove progress file now that we have complete metadata
    if progress_file.exists():
        progress_file.unlink()
    
    logger.info(f"Completed. Total: {len(saved_paths)} files with {total_tokens:,} tokens")
    
    return sorted(saved_paths)


def load_activation_cache(cache_dir: Path) -> Tuple[List[Path], Dict]:
    """
    Load cached activations from disk.
    
    Args:
        cache_dir: Directory containing cached activations
        
    Returns:
        Tuple of (list of file paths, metadata dict)
    """
    cache_dir = Path(cache_dir)
    meta_file = cache_dir / "_meta.json"
    
    if not meta_file.exists():
        raise FileNotFoundError(f"Cache metadata not found at {meta_file}")
    
    with meta_file.open("r") as f:
        meta = json.load(f)
    
    files = sorted(cache_dir.glob("batch_*.pt"))
    
    if len(files) != meta["num_files"]:
        # Warning instead of error to support partial caches during debugging
        logger.warning(f"Metadata says {meta['num_files']} files, but found {len(files)}.")
    
    return files, meta


class InMemoryActivationDataset(Dataset):
    """
    Dataset that loads ALL cached activations into RAM for fast training.
    Requires enough memory to hold all activations (~24GB for 7.7M tokens).
    """
    
    def __init__(
        self,
        files: Sequence[Path],
        total_tokens: int,
        output_dtype: torch.dtype = torch.float32,
        max_tokens: Optional[int] = None,
    ) -> None:
        super().__init__()
        
        logger.info(f"Loading {len(files)} files into memory...")
        
        # Load all activations into a single tensor
        all_activations = []
        for f in tqdm(files, desc="Loading activations"):
            data = torch.load(f, weights_only=True)
            if isinstance(data, dict):
                data = data["activations"]
            all_activations.append(data.to(dtype=output_dtype))
        
        # Concatenate into single tensor
        self.activations = torch.cat(all_activations, dim=0)
        
        # Verify size
        assert self.activations.shape[0] == total_tokens, \
            f"Expected {total_tokens} tokens, got {self.activations.shape[0]}"
        
        # Truncate if max_tokens specified
        if max_tokens is not None and max_tokens < total_tokens:
            logger.info(f"Truncating from {total_tokens:,} to {max_tokens:,} tokens")
            self.activations = self.activations[:max_tokens]
        
        memory_gb = self.activations.numel() * self.activations.element_size() / 1e9
        logger.info(f"Loaded {self.activations.shape[0]:,} tokens, "
                    f"shape={self.activations.shape}, "
                    f"memory={memory_gb:.2f} GB")
    
    def __len__(self) -> int:
        return self.activations.shape[0]
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.activations[idx]


def create_activation_dataloader(
    cache_dir: Path,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    in_memory: bool = True,
    max_tokens: Optional[int] = None,
) -> Tuple[DataLoader, Dict]:
    """
    Create a DataLoader for cached activations.
    
    Args:
        cache_dir: Directory containing cached activations
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
        in_memory: If True, load all activations into RAM (fast but memory-hungry)
        max_tokens: If set, truncate dataset to this many tokens (allows using
                    subset of cached data without rebuilding cache)
        
    Returns:
        Tuple of (DataLoader, metadata dict with actual tokens used)
    """
    files, meta = load_activation_cache(cache_dir)
    total_tokens = meta["total_tokens"]
    
    if in_memory:
        # Fast: load everything into RAM
        dataset = InMemoryActivationDataset(
            files, total_tokens=total_tokens, max_tokens=max_tokens
        )
        # With in-memory data, num_workers=0 is actually fastest (no IPC overhead)
        num_workers = 0
    else:
        # Slower: stream from disk
        dataset = StreamingActivationDataset(files, total_tokens=total_tokens)
        if max_tokens is not None:
            logger.warning(f"max_tokens not supported in streaming mode, using all {total_tokens:,} tokens")
    
    # Update meta with actual tokens used
    meta = meta.copy()
    meta["tokens_used"] = len(dataset)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    return dataloader, meta


# =============================================================================
# WebDataset Loader (for sharded tar archives)
# =============================================================================


def _pt_decoder(data: bytes) -> torch.Tensor:
    """
    Custom decoder for .pt files containing {"activations": tensor}.
    
    WebDataset only has built-in support for .pth files (via torch_loads),
    so we need this custom decoder for .pt extension.
    
    Note: wds.handle_extension("pt", _pt_decoder) already filters for .pt files,
    so this function receives only data bytes (not key).
    
    Args:
        data: Raw bytes from the tar file
        
    Returns:
        The activations tensor
    """
    stream = io.BytesIO(data)
    loaded = torch.load(stream, weights_only=True, map_location="cpu")
    
    # Extract activations from {"activations": tensor} format
    if isinstance(loaded, dict) and "activations" in loaded:
        return loaded["activations"]
    return loaded


def create_webdataset_dataloader(
    shard_pattern: Any,
    batch_size: int,
    shuffle_buffer: int = 500000,
    num_workers: int = 4,
    seed: int = 42,
    pin_memory: bool = True,
    cached_batch_size: int = 13824,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Create a WebDataset-based dataloader with shard and buffer shuffling.
    
    This provides efficient streaming from sharded tar archives on NFS, avoiding
    the high latency of random file access.
    
    IMPORTANT - Data Correlation Warning:
    The cached .pt files contain PRE-BATCHED activations from SEQUENTIAL video
    frames (4 clips × 6 frames × 576 spatial tokens = 13824 tokens per file).
    All tokens in one file are from the SAME video segment and highly correlated.
    
    To achieve proper IID mixing, we unbatch to individual tokens, shuffle with
    a large buffer, then rebatch. The shuffle_buffer should be large enough to
    mix tokens from many different videos.
    
    Memory usage: shuffle_buffer × 768 dims × 4 bytes
    - 1M buffer   ≈ 3 GB
    - 10M buffer  ≈ 30 GB  
    - 50M buffer  ≈ 150 GB (excellent IID mixing)
    
    Args:
        shard_pattern: Shard pattern or explicit list of shard paths, e.g.,
            "/path/to/layer_22-train-*.tar" or [".../layer_22-train-000000.tar", ...]
        batch_size: Desired batch size in tokens for training
        shuffle_buffer: Size of in-memory shuffle buffer (default: 500000)
            With 150GB RAM, can use up to ~50M for nearly full-dataset mixing.
            Set to 0 to disable shuffling (useful for validation).
        num_workers: Number of DataLoader workers (default: 4)
        seed: Random seed for shard and buffer shuffling (default: 42)
        pin_memory: Whether to pin memory for faster GPU transfer (default: True)
        cached_batch_size: Size of pre-batched tensors in .pt files (default: 13824)
        
    Returns:
        Tuple of (WebLoader dataloader, metadata dict)
        
    Example:
        >>> loader, meta = create_webdataset_dataloader(
        ...     shard_pattern="/data/layer_22-train-{000000..000057}.tar",
        ...     batch_size=4096,
        ...     shuffle_buffer=50_000_000,  # 50M tokens, ~150GB
        ... )
    """
    try:
        import webdataset as wds
    except ImportError as e:
        raise ImportError(
            "WebDataset is required for sharded data loading. "
            "Install with: pip install webdataset"
        ) from e
    
    # Calculate effective mixing
    files_in_buffer = (shuffle_buffer / cached_batch_size) if shuffle_buffer > 0 else 0.0
    memory_gb = (shuffle_buffer * 768 * 4 / 1e9) if shuffle_buffer > 0 else 0.0
    
    if isinstance(shard_pattern, (list, tuple)):
        shard_desc = f"{len(shard_pattern)} shards (explicit list)"
        if len(shard_pattern) > 0:
            shard_desc += f", first={shard_pattern[0]}, last={shard_pattern[-1]}"
    else:
        shard_desc = str(shard_pattern)
    logger.info(f"Creating WebDataset loader from: {shard_desc}")
    logger.info(f"Cached batch size: {cached_batch_size}, requested batch size: {batch_size}")
    if shuffle_buffer > 0:
        logger.info(f"Shuffle buffer: {shuffle_buffer:,} tokens (~{files_in_buffer:.1f} files mixed)")
        logger.info(f"Memory estimate: ~{memory_gb:.2f} GB for shuffle buffer")
    else:
        logger.info("Shuffle disabled (shuffle_buffer=0)")
    
    if 0 < shuffle_buffer < cached_batch_size * 10:
        logger.warning(
            f"shuffle_buffer={shuffle_buffer} is small relative to cached_batch_size={cached_batch_size}. "
            f"This means tokens from only ~{files_in_buffer:.1f} files are mixed at a time. "
            f"Consider using shuffle_buffer >= {cached_batch_size * 36} (~500k) for better IID mixing."
        )
    
    # Pipeline: unbatch → shuffle → rebatch
    # We unbatch because cached files contain sequential correlated tokens
    shardshuffle = 100 if shuffle_buffer > 0 else 0
    dataset = wds.WebDataset(
        shard_pattern,
        shardshuffle=shardshuffle,
        seed=seed,
        empty_check=False,  # Allow fewer shards than workers
    ).decode(wds.handle_extension("pt", _pt_decoder)).to_tuple("pt").unbatched()
    
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(shuffle_buffer)
    
    dataset = dataset.batched(batch_size, partial=True).map(lambda batch: batch[0].contiguous())
    
    # Use wds.WebLoader for proper IterableDataset handling
    loader = wds.WebLoader(
        dataset,
        num_workers=num_workers,
        batch_size=None,  # Batching done in pipeline
        pin_memory=pin_memory,
    )
    
    meta = {
        "source": "webdataset",
        "shard_pattern": shard_pattern,
        "batch_size": batch_size,
        "cached_batch_size": cached_batch_size,
        "shuffle_buffer": shuffle_buffer,
        "shuffle_buffer_gb": memory_gb,
        "files_in_buffer": files_in_buffer,
        "num_workers": num_workers,
        "seed": seed,
        "shardshuffle": shardshuffle,
    }
    
    return loader, meta
