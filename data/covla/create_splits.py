#!/usr/bin/env python3
"""
Create split files for CoVLA dataset.

Structure:
data/covla/
├── videos/           # All raw .mp4 files
├── captions/         # All raw .jsonl files
└── splits/           # Split definitions
    ├── train_split.jsonl
    ├── val_split.jsonl
    └── test_split.jsonl

Each split file contains one video_id per line.
"""

import json
from pathlib import Path
from typing import List


def create_splits(
    videos_dir: str,
    captions_dir: str,
    output_dir: str,
    train_videos: int = 90,
    val_videos: int = 10,
    test_videos: int = 50,
):
    """
    Create train/val/test splits based on available videos and captions.
    
    - Train and val use EXISTING videos (already downloaded)
    - Test uses NEW captions (videos to be downloaded)
    """
    videos_dir = Path(videos_dir)
    captions_dir = Path(captions_dir)
    output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get existing video IDs (sorted for reproducibility)
    existing_videos = sorted([p.stem for p in videos_dir.glob("*.mp4")])
    print(f"[splits] Found {len(existing_videos)} existing videos")
    
    # Get all caption IDs (potential videos)
    all_captions = sorted([p.stem for p in captions_dir.glob("*.jsonl")])
    print(f"[splits] Found {len(all_captions)} caption files")
    
    # Create train/val splits from existing videos
    if len(existing_videos) < train_videos + val_videos:
        print(f"[WARN] Only {len(existing_videos)} videos, adjusting split sizes")
        val_videos = max(1, len(existing_videos) // 10)
        train_videos = len(existing_videos) - val_videos
    
    train_ids = existing_videos[:train_videos]
    val_ids = existing_videos[train_videos:train_videos + val_videos]
    
    print(f"[splits] Train: {len(train_ids)} videos")
    print(f"[splits] Val: {len(val_ids)} videos")
    
    # Find captions that DON'T have corresponding videos (for test)
    existing_set = set(existing_videos)
    new_caption_ids = [c for c in all_captions if c not in existing_set]
    
    # Select test videos from new captions
    test_ids = new_caption_ids[:test_videos]
    print(f"[splits] Test: {len(test_ids)} videos (to be downloaded)")
    
    # Write split files
    def write_split(split_name: str, video_ids: List[str]):
        split_file = output_dir / f"{split_name}_split.jsonl"
        with open(split_file, 'w') as f:
            for vid in video_ids:
                f.write(json.dumps({"video_id": vid}) + "\n")
        print(f"[splits] Wrote {split_file}")
    
    write_split("train", train_ids)
    write_split("val", val_ids)
    write_split("test", test_ids)
    
    # Write summary
    summary = {
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "test_count": len(test_ids),
        "total_existing_videos": len(existing_videos),
        "total_captions": len(all_captions),
    }
    with open(output_dir / "splits_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[done] Splits created in {output_dir}")
    print(f"  Train: {len(train_ids)} (existing)")
    print(f"  Val:   {len(val_ids)} (existing)")
    print(f"  Test:  {len(test_ids)} (need to download)")
    
    return train_ids, val_ids, test_ids


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create CoVLA dataset splits")
    parser.add_argument("--videos_dir", type=str, 
                        default="/work/dlclarge2/lushtake-thesis/data/covla/videos")
    parser.add_argument("--captions_dir", type=str,
                        default="/work/dlclarge2/lushtake-thesis/data/covla/captions")
    parser.add_argument("--output_dir", type=str,
                        default="/work/dlclarge2/lushtake-thesis/data/covla/splits")
    parser.add_argument("--train_videos", type=int, default=90)
    parser.add_argument("--val_videos", type=int, default=10)
    parser.add_argument("--test_videos", type=int, default=50)
    
    args = parser.parse_args()
    
    create_splits(
        videos_dir=args.videos_dir,
        captions_dir=args.captions_dir,
        output_dir=args.output_dir,
        train_videos=args.train_videos,
        val_videos=args.val_videos,
        test_videos=args.test_videos,
    )
