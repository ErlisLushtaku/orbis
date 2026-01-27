"""
Activation caching utilities for SAE training.

Adapted from semantic_stage2.py - provides efficient caching of ST-Transformer
activations to avoid repeated forward passes through the frozen Orbis model.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from einops import rearrange

from .activation_hooks import ActivationExtractor


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
    rebuild: bool = False,
    t_noise: float = 0.0,
    frame_rate: int = 5,
    show_progress: bool = True,
) -> List[Path]:
    """
    Precompute and store ST-Transformer activations for SAE training.
    
    Args:
        model: Orbis world model (will be put in eval mode)
        dataloader: DataLoader yielding batches of images
        cache_dir: Directory to store cached activations
        layer_idx: Which transformer block to extract from
        device: Device to run inference on
        dtype: Storage dtype for cached activations
        dtype_name: String name of dtype (for metadata)
        rebuild: If True, rebuild cache even if it exists
        t_noise: Noise timestep for denoising
        frame_rate: Frame rate conditioning value
        show_progress: Whether to show progress bar
        
    Returns:
        List of paths to cached activation files
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_file = cache_dir / "_meta.json"
    
    # Check for existing cache
    existing_files = sorted(cache_dir.glob("*.pt"))
    if existing_files and meta_file.exists() and not rebuild:
        print(f"[cache] Using existing cache from {cache_dir} ({len(existing_files)} files)")
        return existing_files
    
    if rebuild:
        print(f"[cache] Rebuilding cache at {cache_dir}")
    else:
        print(f"[cache] Creating cache at {cache_dir}")
    
    # Clean stale cache
    for path in cache_dir.glob("*.pt"):
        path.unlink()
    if meta_file.exists():
        meta_file.unlink()
    
    model.eval()
    extractor = ActivationExtractor(model, layer_idx=layer_idx, flatten_spatial=True)
    
    saved_paths: List[Path] = []
    total_tokens = 0
    hidden_dim: Optional[int] = None
    
    iterator = tqdm(enumerate(dataloader), total=len(dataloader), desc="Caching activations") \
               if show_progress else enumerate(dataloader)
    
    for batch_idx, batch in iterator:
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
        
        # Add noise if t > 0
        if t_noise > 0:
            target_t, _ = model.add_noise(x, t)
        else:
            target_t = x
        
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
        
        # Save to disk
        path = cache_dir / f"batch_{batch_idx:06d}.pt"
        torch.save({"activations": activations}, path)
        saved_paths.append(path)
        total_tokens += activations.shape[0]
    
    # Save metadata
    meta = {
        "num_files": len(saved_paths),
        "total_tokens": total_tokens,
        "hidden_dim": hidden_dim,
        "dtype": dtype_name,
        "layer_idx": layer_idx,
        "t_noise": t_noise,
        "frame_rate": frame_rate,
    }
    with meta_file.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    
    print(f"[cache] Saved {len(saved_paths)} files with {total_tokens:,} total tokens")
    
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
    
    files = sorted(cache_dir.glob("*.pt"))
    
    if len(files) != meta["num_files"]:
        raise ValueError(
            f"Expected {meta['num_files']} cache files, found {len(files)}. "
            "Cache may be corrupted - try rebuilding."
        )
    
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
    ) -> None:
        super().__init__()
        
        print(f"[cache] Loading {len(files)} files into memory...")
        
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
        
        print(f"[cache] Loaded {self.activations.shape[0]:,} tokens, "
              f"shape={self.activations.shape}, "
              f"memory={self.activations.numel() * 4 / 1e9:.2f} GB")
    
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
        
    Returns:
        Tuple of (DataLoader, metadata dict)
    """
    files, meta = load_activation_cache(cache_dir)
    total_tokens = meta["total_tokens"]
    
    if in_memory:
        # Fast: load everything into RAM
        dataset = InMemoryActivationDataset(files, total_tokens=total_tokens)
        # With in-memory data, num_workers=0 is actually fastest (no IPC overhead)
        num_workers = 0
    else:
        # Slower: stream from disk
        dataset = StreamingActivationDataset(files, total_tokens=total_tokens)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    return dataloader, meta
