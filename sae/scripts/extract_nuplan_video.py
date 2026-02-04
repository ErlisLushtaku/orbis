#!/usr/bin/env python3
"""Extract a NuPlan video to MP4 format for visualization."""

import argparse
import logging
import h5py
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def inspect_h5_structure(h5_path: str):
    """Inspect the structure of an HDF5 file."""
    logger.info(f"=== Inspecting: {h5_path} ===")
    with h5py.File(h5_path, 'r') as f:
        def print_attrs(name, obj):
            logger.info(f"  {name}: {type(obj)}")
            if isinstance(obj, h5py.Dataset):
                logger.info(f"    shape: {obj.shape}, dtype: {obj.dtype}")
        f.visititems(print_attrs)

def extract_video_to_mp4(video_dir: str, output_path: str = None, fps: int = 10, max_frames: int = None):
    """Extract frames from HDF5 and save as MP4.
    
    Args:
        video_dir: Path to nuPlan video directory containing frames.h5
        output_path: Output MP4 path. If None, saves next to video_dir
        fps: Frames per second for output video (default 10, matching nuPlan 10Hz)
        max_frames: Maximum number of frames to extract (None = all)
    """
    import cv2
    
    video_dir = Path(video_dir)
    frames_h5 = video_dir / "frames.h5"
    
    if not frames_h5.exists():
        raise FileNotFoundError(f"frames.h5 not found in {video_dir}")
    
    if output_path is None:
        output_path = video_dir.parent / f"{video_dir.name}.mp4"
    
    logger.info(f"Reading frames from: {frames_h5}")
    
    with h5py.File(frames_h5, 'r') as f:
        # Check structure - nuPlan stores frames in chunks
        keys = sorted(f.keys())
        logger.info(f"HDF5 keys: {keys}")
        
        # Read first chunk to get frame dimensions
        first_chunk = f[keys[0]][:]
        _, height, width, channels = first_chunk.shape
        logger.info(f"Frame dimensions: {width}x{height}, {channels} channels")
        
        # Calculate total frames
        total_frames = sum(f[k].shape[0] for k in keys)
        logger.info(f"Total frames: {total_frames}")
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        logger.info(f"Extracting {total_frames} frames at {fps} FPS")
        logger.info(f"Duration: {total_frames / fps:.1f} seconds")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        for key in keys:
            if max_frames and frame_count >= max_frames:
                break
                
            chunk_frames = f[key][:]
            
            for frame in chunk_frames:
                if max_frames and frame_count >= max_frames:
                    break
                    
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
                frame_count += 1
                
                if frame_count % 200 == 0:
                    logger.info(f"  Processed {frame_count}/{total_frames} frames")
        
        out.release()
    
    logger.info(f"Saved video to: {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Extract NuPlan video to MP4")
    parser.add_argument("video_dir", type=str, help="Path to nuPlan video directory")
    parser.add_argument("--output", "-o", type=str, help="Output MP4 path")
    parser.add_argument("--fps", type=int, default=10, help="Output FPS (default: 10)")
    parser.add_argument("--max_frames", "-n", type=int, help="Max frames to extract")
    parser.add_argument("--inspect", action="store_true", help="Only inspect H5 structure")
    
    args = parser.parse_args()
    
    video_dir = Path(args.video_dir)
    
    if args.inspect:
        inspect_h5_structure(video_dir / "frames.h5")
        inspect_h5_structure(video_dir / "odometry.h5")
    else:
        extract_video_to_mp4(args.video_dir, args.output, args.fps, args.max_frames)


if __name__ == "__main__":
    main()
