"""
SpatialTopKTracker: per-latent top-K spatial activation map tracker.

Used during SAE activation collection to maintain the top-K samples per
latent with their full spatial activation maps. This enables post-hoc
visualization of which spatial tokens most activate each latent, without
requiring a second GPU pass.

Memory budget: num_latents x top_k x spatial_tokens x 4 bytes.
For 12288 latents x 10 x 576 tokens = ~282 MB (CPU RAM).
"""

import heapq
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class SpatialEntry:
    """A single entry in the spatial top-K tracker for one latent."""

    score: float
    video_id: str
    frame_idx: int
    clip_frame_indices: List[int]
    spatial_map: np.ndarray  # (N,) per-token activation for this latent
    metadata: Dict[str, float]  # odometry (NuPlan) or caption metadata (CoVLA)


class SpatialTopKTracker:
    """Maintains per-latent top-K samples with spatial activation maps.

    During the forward pass, for each batch the tracker is updated with
    spatially-averaged scores and full spatial activations. Only the top-K
    entries per latent are retained, using min-heaps for efficient eviction.
    """

    def __init__(self, num_latents: int, top_k: int = 10):
        self.num_latents = num_latents
        self.top_k = top_k
        # Per-latent min-heaps. Each element: (score, counter, SpatialEntry)
        # counter breaks ties so heapq never compares SpatialEntry
        self._heaps: List[list] = [[] for _ in range(num_latents)]
        self._counter = 0
        # Track current minimums for vectorized filtering
        self._mins = np.full(num_latents, -np.inf, dtype=np.float32)

    def update(
        self,
        scores: np.ndarray,
        spatial_acts: np.ndarray,
        metadata: List[dict],
    ) -> None:
        """Update tracker with a batch of samples.

        Args:
            scores: (B, num_latents) spatially-averaged activation per sample.
            spatial_acts: (B, N, num_latents) full spatial activations.
            metadata: List of B dicts, each with keys:
                video_id, frame_idx, clip_frame_indices, and a data-source
                specific metadata dict (odometry for NuPlan, caption scores
                for CoVLA).
        """
        batch_size = scores.shape[0]

        for i in range(batch_size):
            sample_scores = scores[i]  # (num_latents,)
            # Vectorized check: which latents could this sample improve?
            if len(self._heaps[0]) >= self.top_k:
                mask = sample_scores > self._mins
            else:
                mask = sample_scores > 0
            candidate_latents = np.nonzero(mask)[0]

            if len(candidate_latents) == 0:
                continue

            meta = metadata[i]
            sample_spatial = spatial_acts[i]  # (N, num_latents)

            for lat_idx in candidate_latents:
                score = float(sample_scores[lat_idx])
                heap = self._heaps[lat_idx]

                entry = SpatialEntry(
                    score=score,
                    video_id=meta["video_id"],
                    frame_idx=meta["frame_idx"],
                    clip_frame_indices=meta.get("clip_frame_indices", []),
                    spatial_map=sample_spatial[:, lat_idx].copy(),
                    metadata=meta.get("metadata", meta.get("odometry", {})),
                )

                if len(heap) < self.top_k:
                    heapq.heappush(heap, (score, self._counter, entry))
                    self._counter += 1
                    if len(heap) == self.top_k:
                        self._mins[lat_idx] = heap[0][0]
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, (score, self._counter, entry))
                    self._counter += 1
                    self._mins[lat_idx] = heap[0][0]

    def get_top_k(self, latent_idx: int) -> List[SpatialEntry]:
        """Get top-K entries for a latent, sorted by score descending."""
        return sorted(
            [entry for _, _, entry in self._heaps[latent_idx]],
            key=lambda e: e.score,
            reverse=True,
        )

    def save(self, path: Path) -> None:
        """Serialize tracker data to disk."""
        data: Dict[int, list] = {}
        for lat_idx in range(self.num_latents):
            entries = self.get_top_k(lat_idx)
            if not entries:
                continue
            data[lat_idx] = [
                {
                    "score": e.score,
                    "video_id": e.video_id,
                    "frame_idx": e.frame_idx,
                    "clip_frame_indices": e.clip_frame_indices,
                    "spatial_map": e.spatial_map,
                    "metadata": e.metadata,
                }
                for e in entries
            ]
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"version": 1, "num_latents": self.num_latents, "data": data}, path,
        )
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(
            f"Saved spatial top-K cache ({len(data)} active latents) "
            f"to {path} ({size_mb:.1f} MB)",
        )

    @classmethod
    def load(cls, path: Path) -> "SpatialTopKTracker":
        """Load tracker from disk."""
        cache = torch.load(path, weights_only=False)
        num_latents = cache["num_latents"]
        data = cache["data"]
        max_k = max((len(entries) for entries in data.values()), default=10)
        tracker = cls(num_latents=num_latents, top_k=max_k)
        for lat_idx, entries in data.items():
            for entry_dict in entries:
                entry = SpatialEntry(
                    score=entry_dict["score"],
                    video_id=entry_dict["video_id"],
                    frame_idx=entry_dict["frame_idx"],
                    clip_frame_indices=entry_dict.get("clip_frame_indices", []),
                    spatial_map=entry_dict["spatial_map"],
                    metadata=entry_dict.get(
                        "metadata", entry_dict.get("odometry", {}),
                    ),
                )
                heapq.heappush(
                    tracker._heaps[lat_idx],
                    (entry.score, tracker._counter, entry),
                )
                tracker._counter += 1
            if tracker._heaps[lat_idx]:
                tracker._mins[lat_idx] = tracker._heaps[lat_idx][0][0]
        logger.info(
            f"Loaded spatial top-K cache ({len(data)} active latents) from {path}",
        )
        return tracker
