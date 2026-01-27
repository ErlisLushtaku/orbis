import os
import time
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import h5py


class NuPlanOrbisMultiFrame(Dataset):
    """
    NuPlan dataset compatible with Orbis pipeline.

    Reads frames from HDF5 files and includes synchronized odometry data
    (speed, acceleration, yaw rate) for semantic grounding analysis.

    Uses aspect-ratio-aware "resize to cover + center crop" to match target resolution.

    For nuPlan (600x360) -> Orbis (512x288):
        - Scale width to 512, height becomes ~307
        - Center crop height from 307 to 288

    Args:
        data_dir: Root directory containing video folders with frames.h5 and odometry.h5
        num_frames: Number of frames per sample (context + target)
        stored_data_frame_rate: FPS of stored videos (default: 10 for nuPlan)
        target_frame_rate: Logical FPS for the model (default: 5)
        size: (H, W) final resolution of frames (default: 288x512 to match Orbis)
        num_videos: Optional cap on number of videos used (None = all)
        clips_per_video: Number of clips to extract per video (None = auto-compute)
        video_ids_file: Optional path to .txt file listing video IDs to use
        include_odometry: Whether to include odometry metadata in samples
        debug: If True, prints debug info
    """

    SOURCE_WIDTH = 600
    SOURCE_HEIGHT = 360

    def __init__(
        self,
        data_dir: str,
        num_frames: int = 6,
        stored_data_frame_rate: int = 10,
        target_frame_rate: int = 5,
        size=(288, 512),
        num_videos: int | None = None,
        clips_per_video: int | None = None,
        video_ids_file: str | None = None,
        include_odometry: bool = True,
        debug: bool = False,
    ):
        t0 = time.time() if debug else None

        self.data_dir = data_dir
        self.num_frames = num_frames
        self.stored_rate = stored_data_frame_rate
        self.target_rate = target_frame_rate
        self.frame_interval = max(1, round(self.stored_rate / self.target_rate))
        self.include_odometry = include_odometry
        self.debug = debug
        self.frames_per_clip = self.num_frames * self.frame_interval

        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = (int(size[0]), int(size[1]))

        resize_size = self._get_resize_size(
            source_size=(self.SOURCE_WIDTH, self.SOURCE_HEIGHT),
            target_size=self.size,
        )

        self.base_transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
        ])
        self._resize_size = resize_size

        if debug:
            print("\n================= [INIT] NuPlanOrbisMultiFrame =================")
            print(f"[INIT] Scanning videos in: {data_dir}")

        if video_ids_file and os.path.exists(video_ids_file):
            with open(video_ids_file, 'r') as f:
                all_video_ids = [line.strip() for line in f if line.strip()]
        else:
            all_video_ids = []
            for name in sorted(os.listdir(data_dir)):
                video_path = os.path.join(data_dir, name)
                if os.path.isdir(video_path) and os.path.exists(os.path.join(video_path, "frames.h5")):
                    all_video_ids.append(name)

        self.all_video_ids = all_video_ids
        if len(self.all_video_ids) == 0:
            raise RuntimeError(f"No video folders found in {data_dir}")

        if num_videos is None:
            self.num_videos = len(self.all_video_ids)
        else:
            self.num_videos = min(num_videos, len(self.all_video_ids))

        self.video_ids = self.all_video_ids[:self.num_videos]

        if clips_per_video is not None:
            self.clips_per_video = clips_per_video
            self.total_samples = self.num_videos * self.clips_per_video
            self._video_clips = None
        else:
            self._video_clips = []
            total = 0
            for vid in self.video_ids:
                frames_h5 = os.path.join(data_dir, vid, "frames.h5")
                with h5py.File(frames_h5, 'r') as f:
                    total_frames = sum(f[k].shape[0] for k in f.keys())
                num_clips = max(1, total_frames // self.frames_per_clip)
                self._video_clips.append((vid, num_clips, total_frames))
                total += num_clips
            self.total_samples = total
            self.clips_per_video = None

        if debug:
            print(f"[INIT] Found {len(self.all_video_ids)} total videos")
            print(f"[INIT] Using num_videos = {self.num_videos}")
            print(f"[INIT] frames_per_clip = {self.frames_per_clip}")
            print(f"[INIT] total_samples = {self.total_samples}")
            print(f"[INIT] resize_size = {self._resize_size}, final = {self.size}")
            print(f"[INIT] Completed in {time.time() - t0:.2f}s")
            print("================================================================\n")

    @staticmethod
    def _get_resize_size(source_size: tuple, target_size: tuple) -> tuple:
        src_w, src_h = source_size
        tgt_h, tgt_w = target_size
        scale = max(tgt_w / src_w, tgt_h / src_h)
        return (int(src_h * scale), int(src_w * scale))

    def _idx_to_video_clip(self, idx: int) -> tuple:
        if self._video_clips is None:
            video_idx = idx // self.clips_per_video
            clip_idx = idx % self.clips_per_video
            return self.video_ids[video_idx], clip_idx, -1
        else:
            cumulative = 0
            for video_id, num_clips, total_frames in self._video_clips:
                if idx < cumulative + num_clips:
                    return video_id, idx - cumulative, total_frames
                cumulative += num_clips
            raise IndexError(f"Index {idx} out of range")

    def _get_clip_frame_indices(self, clip_idx: int, total_frames: int) -> list:
        start = clip_idx * self.frames_per_clip
        if start >= total_frames:
            start = start % max(1, total_frames - self.frames_per_clip)
        raw = [start + i * self.frame_interval for i in range(self.num_frames)]
        return [min(i, total_frames - 1) for i in raw]

    def _load_frames_from_h5(self, h5_path: str, indices: list) -> list:
        frames = []
        with h5py.File(h5_path, 'r') as f:
            keys = sorted(f.keys())
            chunk_starts = []
            cum = 0
            for k in keys:
                chunk_starts.append(cum)
                cum += f[k].shape[0]
            total = cum
            
            for idx in indices:
                idx = min(idx, total - 1)
                chunk_idx = 0
                for i, start in enumerate(chunk_starts):
                    if i + 1 < len(chunk_starts) and idx >= chunk_starts[i + 1]:
                        chunk_idx = i + 1
                    elif idx >= start:
                        chunk_idx = i
                local = idx - chunk_starts[chunk_idx]
                frames.append(f[keys[chunk_idx]][local])
        return frames

    def _load_odometry_from_h5(self, h5_path: str, indices: list) -> dict:
        with h5py.File(h5_path, 'r') as f:
            keys = sorted(f.keys())
            chunk_starts, cum = [], 0
            for k in keys:
                chunk_starts.append(cum)
                cum += f[k].shape[0]
            total = cum
            
            speeds, accels, yaws = [], [], []
            target_odom = None
            
            for i, idx in enumerate(indices):
                idx = min(idx, total - 1)
                chunk_idx = 0
                for j, start in enumerate(chunk_starts):
                    if j + 1 < len(chunk_starts) and idx >= chunk_starts[j + 1]:
                        chunk_idx = j + 1
                    elif idx >= start:
                        chunk_idx = j
                local = idx - chunk_starts[chunk_idx]
                odom = f[keys[chunk_idx]][local]
                
                spd = np.sqrt(odom['vx']**2 + odom['vy']**2)
                acc = np.sqrt(odom['acceleration_x']**2 + odom['acceleration_y']**2)
                yaw = abs(odom['angular_rate_z'])
                speeds.append(spd)
                accels.append(acc)
                yaws.append(yaw)
                
                if i == len(indices) - 1:
                    target_odom = {'speed': float(spd), 'accel': float(acc), 
                                   'yaw': float(odom['angular_rate_z'])}
        
        speeds, accels, yaws = np.array(speeds), np.array(accels), np.array(yaws)
        return {
            'speed_mean': float(speeds.mean()),
            'speed_kmh_mean': float(speeds.mean() * 3.6),
            'acceleration_mean': float(accels.mean()),
            'yaw_rate_mean': float(yaws.mean()),
            'is_stopped_rate': float((speeds < 0.5).mean()),
            'is_turning_rate': float((yaws > 0.05).mean()),
            'target_speed': target_odom['speed'] if target_odom else 0.0,
            'target_speed_kmh': target_odom['speed'] * 3.6 if target_odom else 0.0,
            'target_acceleration': target_odom['accel'] if target_odom else 0.0,
            'target_yaw_rate': abs(target_odom['yaw']) if target_odom else 0.0,
        }

    def _apply_transforms(self, np_frames: list) -> torch.Tensor:
        pil = [Image.fromarray(f).convert("RGB") for f in np_frames]
        tensors = [self.base_transform(img) for img in pil]
        frames = torch.stack(tensors, dim=0)
        return frames * 2 - 1

    def __getitem__(self, idx: int) -> dict:
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range")

        video_id, clip_idx, cached_total = self._idx_to_video_clip(idx)
        video_dir = os.path.join(self.data_dir, video_id)
        frames_h5 = os.path.join(video_dir, "frames.h5")
        odometry_h5 = os.path.join(video_dir, "odometry.h5")

        if cached_total < 0:
            with h5py.File(frames_h5, 'r') as f:
                total_frames = sum(f[k].shape[0] for k in f.keys())
        else:
            total_frames = cached_total

        indices = self._get_clip_frame_indices(clip_idx, total_frames)
        np_frames = self._load_frames_from_h5(frames_h5, indices)
        frames = self._apply_transforms(np_frames)

        result = {
            "images": frames,
            "video_id": video_id,
            "frame_idx": indices[-1],
            "frame_rate": self.target_rate,
        }

        if self.include_odometry and os.path.exists(odometry_h5):
            result.update(self._load_odometry_from_h5(odometry_h5, indices))

        return result

    def __len__(self) -> int:
        return self.total_samples


if __name__ == "__main__":
    print("=" * 60)
    print("Testing NuPlanOrbisMultiFrame Dataset")
    print("=" * 60)

    DATA_DIR = "/work/dlcsmall2/galessos-nuPlan/nuPlan_640x360_10Hz"

    print("\n[Test 1] Resize calculation")
    rs = NuPlanOrbisMultiFrame._get_resize_size((600, 360), (288, 512))
    print(f"  nuPlan 600x360 -> Orbis 512x288: resize to {rs}")
    assert rs[1] == 512 and rs[0] >= 288
    print("  [PASS]")

    print("\n[Test 2] Load dataset (3 videos)")
    ds = NuPlanOrbisMultiFrame(data_dir=DATA_DIR, num_videos=3, debug=True)
    print(f"  total_samples = {len(ds)}")
    print("  [PASS]")

    print("\n[Test 3] Get sample")
    s = ds[0]
    print(f"  video_id: {s['video_id']}")
    print(f"  images: {tuple(s['images'].shape)}")
    print(f"  speed: {s['target_speed_kmh']:.1f} km/h")
    assert s['images'].shape == (6, 3, 288, 512)
    print("  [PASS]")

    print("\n[Test 4] Value range")
    print(f"  min={s['images'].min():.3f}, max={s['images'].max():.3f}")
    assert -1.0 <= s['images'].min() and s['images'].max() <= 1.0
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("All tests passed!")
