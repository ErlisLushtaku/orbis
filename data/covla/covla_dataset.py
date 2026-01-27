import os
import json
import time
import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from decord import VideoReader, cpu 


class CoVLAOrbisMultiFrame(Dataset):
    """
    Local CoVLA dataset compatible with Orbis pipeline.

    Uses aspect-ratio-aware "resize to cover + center crop" to match target resolution
    without distortion. This ensures the image fills the target size completely,
    cropping the excess from the dimension that overflows.

    For CoVLA (1928x1208, aspect 1.596) → Orbis (512x288, aspect 1.778):
        - Scale width to 512 → height becomes ~321
        - Center crop height from 321 to 288

    Multi-sampling: Each video yields multiple non-overlapping clips.
    - 30-second video at 20 FPS = 600 frames
    - Each clip: 6 frames × 4 frame_interval = 24 stored frames
    - Clips per video: 600 / 24 = 25 non-overlapping clips
    - Total samples = num_videos × clips_per_video

    Args:
        num_frames: number of frames per sample (context + target)
        stored_data_frame_rate: fps of stored videos (e.g. 20)
        target_frame_rate: logical fps for the model (e.g. 5)
        size: (H, W) final resolution of frames (default: 288x512 to match Orbis)
        captions_dir: directory containing <video_id>.jsonl caption files
        videos_dir: directory containing .mp4 videos
        num_videos: optional cap on number of videos used (None = all)
        clips_per_video: number of clips to extract per video (None = auto-compute)
        assumed_video_frames: assumed frames per video for auto-computing clips (default: 600)
        debug: if True, prints debug info
        aug: kept only for logging (no behavior differences)
        scale_min/scale_max: unused (kept for compatibility)
    """

    # CoVLA video source resolution (used for resize calculation)
    SOURCE_WIDTH = 1928
    SOURCE_HEIGHT = 1208

    def __init__(
        self,
        num_frames: int = 6,
        stored_data_frame_rate: int = 20,
        target_frame_rate: int = 5,
        size=(288, 512),  # (H, W) - default matches Orbis 288x512 checkpoint
        captions_dir: str | None = None,
        videos_dir: str = "data/covla_100_videos",
        num_videos: int | None = None,
        clips_per_video: int | None = None,
        assumed_video_frames: int = 600,  # 30 seconds at 20 FPS
        debug: bool = False,
        aug: str = "resize_center_crop",   # kept for compatibility / logging only
        scale_min: float = 0.75,  # unused
        scale_max: float = 1.0,   # unused
        # Legacy parameter name (maps to num_videos)
        num_samples: int | None = None,
    ):
        t0 = time.time() if debug else None

        # Handle legacy parameter name
        if num_samples is not None and num_videos is None:
            num_videos = num_samples

        # ----------- basic settings -----------
        self.num_frames = num_frames
        self.stored_rate = stored_data_frame_rate
        self.target_rate = target_frame_rate
        self.frame_interval = max(1, round(self.stored_rate / self.target_rate))
        
        # Frames needed per clip (for non-overlapping sampling)
        self.frames_per_clip = self.num_frames * self.frame_interval

        # normalize size → (H, W)
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = (int(size[0]), int(size[1]))  # (H, W)

        self.captions_dir = captions_dir
        self.videos_dir = videos_dir
        self.debug = debug
        self.aug = aug  # only for logging

        # ----------- transforms (resize to cover + center crop) -----------
        # Calculate intermediate resize size that covers target while maintaining aspect ratio
        resize_size = self._get_resize_size(
            source_size=(self.SOURCE_WIDTH, self.SOURCE_HEIGHT),
            target_size=self.size,
        )
        
        self.base_transform = transforms.Compose(
            [
                transforms.Resize(resize_size),      # Resize to cover target (maintains aspect ratio)
                transforms.CenterCrop(self.size),    # Crop to exact target size
                transforms.ToTensor(),
            ]
        )
        
        self._resize_size = resize_size  # Store for debug logging

        # ----------- collect video_ids -----------
        if debug:
            print("\n================= [INIT] CoVLAOrbisMultiFrame (LOCAL) =================")
            print(f"[INIT] Scanning videos in: {videos_dir}")

        all_files = [f for f in os.listdir(videos_dir) if f.endswith(".mp4")]
        all_files = sorted(all_files)
        self.video_ids = [os.path.splitext(f)[0] for f in all_files]

        if len(self.video_ids) == 0:
            raise RuntimeError(f"No .mp4 files found in {videos_dir}")

        # Limit videos if requested
        if num_videos is None:
            self.num_videos = len(self.video_ids)
        else:
            self.num_videos = min(num_videos, len(self.video_ids))

        # ----------- compute clips per video -----------
        if clips_per_video is None:
            # Auto-compute based on assumed video length
            self.clips_per_video = max(1, assumed_video_frames // self.frames_per_clip)
        else:
            self.clips_per_video = clips_per_video
        
        # Total samples = videos × clips
        self.total_samples = self.num_videos * self.clips_per_video

        if debug:
            print(f"[INIT] Found {len(self.video_ids)} videos.")
            print(f"[INIT] Using num_videos = {self.num_videos}")
            print(f"[INIT] frames_per_clip = {self.frames_per_clip} (num_frames × frame_interval)")
            print(f"[INIT] clips_per_video = {self.clips_per_video}")
            print(f"[INIT] total_samples = {self.total_samples} (videos × clips)")
            print(f"[INIT] num_frames={num_frames}, frame_interval={self.frame_interval}")
            print(f"[INIT] resize_size = {self._resize_size} (H, W) - intermediate")
            print(f"[INIT] final size = {self.size} (H, W)")
            print(f"[INIT] aug = {self.aug}")
            print(f"[INIT] Completed in {time.time() - t0:.2f} seconds")
            print("================================================================\n")

    # ------------------------------------------------------------------
    @staticmethod
    def _get_resize_size(source_size: tuple, target_size: tuple) -> tuple:
        """
        Calculate intermediate resize size that covers target while maintaining aspect ratio.
        
        This implements "resize to cover + center crop" logic:
        - Compare source and target aspect ratios
        - Scale using the LARGER scale factor (ensures image covers target in both dimensions)
        - The result will be >= target in both dimensions, ready for center crop
        
        Args:
            source_size: (W, H) of source image
            target_size: (H, W) of target - note different order!
            
        Returns:
            (H, W) tuple for transforms.Resize()
        """
        src_w, src_h = source_size
        tgt_h, tgt_w = target_size  # target_size is (H, W)
        
        # Calculate scale factors for each dimension
        scale_w = tgt_w / src_w
        scale_h = tgt_h / src_h
        
        # Use larger scale factor to ensure the image covers the target completely
        scale = max(scale_w, scale_h)
        
        # Calculate new size (maintaining aspect ratio)
        new_h = int(src_h * scale)
        new_w = int(src_w * scale)
        
        return (new_h, new_w)  # Return (H, W) for transforms.Resize()

    # ------------------------------------------------------------------
    def load_captions(self, video_id: str) -> dict[int, str]:
        """Load captions from <video_id>.jsonl if available."""
        if not self.captions_dir:
            return {}

        path = os.path.join(self.captions_dir, f"{video_id}.jsonl")
        if not os.path.exists(path):
            return {}

        caps: dict[int, str] = {}
        with open(path, "r") as f:
            for line in f:
                entry = json.loads(line)
                frame_idx_str = list(entry.keys())[0]
                frame_idx = int(frame_idx_str)
                caps[frame_idx] = entry[frame_idx_str].get("plain_caption", "")
        return caps

    # ------------------------------------------------------------------
    def _get_clip_frame_indices(self, clip_idx: int, total_frames: int) -> list[int]:
        """
        Get deterministic frame indices for a specific clip.
        
        Each clip starts at clip_idx * frames_per_clip, giving non-overlapping clips.
        If the clip would exceed the video length, we clamp to valid indices.
        
        Args:
            clip_idx: Which clip (0, 1, 2, ...) within the video
            total_frames: Total frames in the video
            
        Returns:
            List of frame indices for this clip
        """
        if total_frames <= 0:
            raise RuntimeError("Video has zero frames")

        # Start frame for this clip (non-overlapping)
        start = clip_idx * self.frames_per_clip
        
        # If start is beyond video, wrap around or clamp
        if start >= total_frames:
            # Wrap around to beginning with offset
            start = start % max(1, total_frames - self.frames_per_clip)
        
        # Generate frame indices with frame_interval spacing
        raw_idxs = [start + i * self.frame_interval for i in range(self.num_frames)]
        
        # Clamp to valid range
        idxs = [min(i, total_frames - 1) for i in raw_idxs]
        return idxs

    # ------------------------------------------------------------------
    def _apply_transforms_to_frames(self, pil_frames: list[Image.Image]) -> torch.Tensor:
        """
        Apply resize-to-cover + center crop, stack frames, and normalize to [-1, 1].
        """
        transformed = [self.base_transform(img) for img in pil_frames]
        frames = torch.stack(transformed, dim=0)  # (F, C, H, W)
        frames = frames * 2 - 1  # [0,1] → [-1,1]
        return frames

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range for dataset of length {self.total_samples}")

        # Map flat index to (video_idx, clip_idx)
        video_idx = idx // self.clips_per_video
        clip_idx = idx % self.clips_per_video

        video_id = self.video_ids[video_idx]
        video_path = os.path.join(self.videos_dir, f"{video_id}.mp4")

        if self.debug:
            print(f"\n================= [GETITEM] idx={idx} =================")
            print(f"[GETITEM] video_idx={video_idx}, clip_idx={clip_idx}")
            print(f"[GETITEM] video_id={video_id}")
            print(f"[GETITEM] video_path={video_path}")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # ---- open video with decord ----
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        if self.debug:
            print(f"[GETITEM] total_frames={total_frames}")

        # ---- captions ----
        captions = self.load_captions(video_id)

        # ---- frame indices (deterministic based on clip_idx) ----
        idxs = self._get_clip_frame_indices(clip_idx, total_frames)
        if self.debug:
            print(f"[GETITEM] Clip {clip_idx} frame indices: {idxs}")

        pil_frames: list[Image.Image] = []
        texts: list[str] = []

        for i in idxs:
            frame = vr[i].asnumpy()
            img = Image.fromarray(frame).convert("RGB")
            pil_frames.append(img)
            texts.append(captions.get(i, ""))

        # apply resize + stack
        frames = self._apply_transforms_to_frames(pil_frames)

        # choose global caption (like Orbis often does)
        global_caption = texts[0] if len(texts) > 0 and texts[0] != "" else ""
        if global_caption == "":
            global_caption = "no caption"

        if self.debug:
            print(f"[GETITEM] global_caption='{global_caption[:50]}…'")
            print(f"[GETITEM] frames.shape={tuple(frames.shape)} (F,C,H,W)")
            print("[GETITEM] Returning sample.\n")

        return {
            "images": frames,             # (F, 3, H, W) in [-1, 1]
            "caption": global_caption,
            "video_id": video_id,
            "frame_rate": self.target_rate,  # scalar, used to build [B]-tensor in training
        }

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.total_samples


if __name__ == "__main__":
    """Test the CoVLA dataset implementation."""
    print("=" * 60)
    print("Testing CoVLAOrbisMultiFrame Dataset (Multi-Sample)")
    print("=" * 60)
    
    # Test 1: Resize size calculation
    print("\n[Test 1] Resize size calculation")
    resize_size = CoVLAOrbisMultiFrame._get_resize_size(
        source_size=(1928, 1208),  # CoVLA (W, H)
        target_size=(288, 512),    # Orbis (H, W)
    )
    print(f"  CoVLA source: 1928x1208 (WxH)")
    print(f"  Orbis target: 512x288 (WxH) = (288, 512) in (H,W) format")
    print(f"  Intermediate resize: {resize_size} (H, W)")
    print(f"  Expected: ~(321, 512) - scale width to 512, then crop height 321->288")
    
    assert resize_size[1] == 512, f"Width should be 512, got {resize_size[1]}"
    assert resize_size[0] > 288, f"Height should be > 288 before crop, got {resize_size[0]}"
    print("  [PASS] Resize calculation correct!")
    
    # Test 2: Load dataset with multi-sampling
    print("\n[Test 2] Loading dataset with multi-sampling")
    try:
        ds = CoVLAOrbisMultiFrame(
            videos_dir="/work/dlclarge2/lushtake-thesis/data/covla/videos",
            captions_dir="/work/dlclarge2/lushtake-thesis/data/covla/captions",
            size=(288, 512),
            num_videos=2,  # Only use 2 videos for testing
            debug=True,
        )
        print(f"  num_videos = {ds.num_videos}")
        print(f"  clips_per_video = {ds.clips_per_video}")
        print(f"  total_samples = {len(ds)}")
        print(f"  Expected: 2 videos × 25 clips = 50 samples")
        assert len(ds) == ds.num_videos * ds.clips_per_video
        print("  [PASS] Multi-sampling setup works!")
    except Exception as e:
        print(f"  [FAIL] Dataset loading failed: {e}")
        raise
    
    # Test 3: Verify different clips have different frame indices
    print("\n[Test 3] Verifying clips have different starting frames")
    sample_0 = ds[0]   # Video 0, Clip 0
    sample_1 = ds[1]   # Video 0, Clip 1
    sample_25 = ds[25] # Video 1, Clip 0 (if clips_per_video=25)
    
    print(f"  Sample 0: video_id={sample_0['video_id']}")
    print(f"  Sample 1: video_id={sample_1['video_id']}")
    print(f"  Sample 25: video_id={sample_25['video_id']}")
    
    # Same video for samples 0 and 1
    assert sample_0['video_id'] == sample_1['video_id'], "Samples 0 and 1 should be from same video"
    # Different video for sample 25
    assert sample_0['video_id'] != sample_25['video_id'], "Sample 25 should be from different video"
    print("  [PASS] Clip indexing correct!")
    
    # Test 4: Verify output shape
    print("\n[Test 4] Verifying output shape")
    print(f"  Images shape: {tuple(sample_0['images'].shape)}")
    expected_shape = (6, 3, 288, 512)
    assert sample_0["images"].shape == expected_shape, \
        f"Shape mismatch: got {tuple(sample_0['images'].shape)}, expected {expected_shape}"
    print("  [PASS] Output shape correct!")
    
    # Test 5: Verify value range
    print("\n[Test 5] Verifying pixel value range")
    img_min = sample_0["images"].min().item()
    img_max = sample_0["images"].max().item()
    print(f"  Value range: [{img_min:.3f}, {img_max:.3f}]")
    assert img_min >= -1.0 and img_max <= 1.0, \
        f"Values outside [-1, 1]: min={img_min}, max={img_max}"
    print("  [PASS] Value range correct!")
    
    # Test 6: Full dataset stats
    print("\n[Test 6] Full dataset statistics")
    ds_full = CoVLAOrbisMultiFrame(
        videos_dir="/work/dlclarge2/lushtake-thesis/data/covla/videos",
        captions_dir="/work/dlclarge2/lushtake-thesis/data/covla/captions",
        size=(288, 512),
        debug=False,
    )
    print(f"  num_videos = {ds_full.num_videos}")
    print(f"  clips_per_video = {ds_full.clips_per_video}")
    print(f"  total_samples = {len(ds_full)}")
    print(f"  frames_per_clip = {ds_full.frames_per_clip}")
    print("  [PASS] Full dataset loaded!")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
