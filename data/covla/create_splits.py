#!/usr/bin/env python3
"""
Create split files for CoVLA dataset.

Splits match the SAE training pipeline: videos are sorted alphabetically,
then split by index into train (first 90%) and val (last 10%).

Structure:
data/covla/
├── videos/           # All raw .mp4 files
├── captions/         # All raw .jsonl files
└── splits/           # Split definitions
    ├── train_split.jsonl
    ├── val_split.jsonl
    ├── test_split.jsonl   (optional: from undownloaded captions)
    └── splits_summary.json

Each split file contains one video_id per line as {"video_id": "..."}.
"""

import json
from pathlib import Path
from typing import List, Tuple


def create_splits(
    videos_dir: str,
    captions_dir: str,
    output_dir: str,
    val_split: float = 0.1,
    test_videos: int = 0,
) -> Tuple[List[str], List[str], List[str]]:
    """Create train/val/test splits matching SAE training pipeline.

    Train and val are created from existing videos using the same sorted-order
    split as CoVLAOrbisMultiFrame + create_datasets_covla in train_sae.py.

    Test videos (optional) are selected from captions that have no local video,
    requiring a separate download step before use.

    Args:
        videos_dir: Directory containing .mp4 video files.
        captions_dir: Directory containing .jsonl caption files.
        output_dir: Where to write the split files.
        val_split: Fraction of videos for validation (must match SAE training).
        test_videos: Number of test videos to select from undownloaded captions.
    """
    videos_dir = Path(videos_dir)
    captions_dir = Path(captions_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Sorted for deterministic ordering (matches CoVLAOrbisMultiFrame)
    existing_videos = sorted([p.stem for p in videos_dir.glob("*.mp4")])
    all_captions = sorted([p.stem for p in captions_dir.glob("*.jsonl")])
    print(f"[splits] Found {len(existing_videos)} existing videos")
    print(f"[splits] Found {len(all_captions)} caption files")

    # Train/val split (same logic as create_datasets_covla in train_sae.py)
    total = len(existing_videos)
    num_val = max(1, int(total * val_split))
    num_train = total - num_val

    train_ids = existing_videos[:num_train]
    val_ids = existing_videos[num_train:]

    print(f"[splits] Train: {len(train_ids)} videos (indices 0..{num_train - 1})")
    print(f"[splits] Val:   {len(val_ids)} videos (indices {num_train}..{total - 1})")

    # Test: captions without local video (need download)
    existing_set = set(existing_videos)
    available_test = [c for c in all_captions if c not in existing_set]
    test_ids = available_test[:test_videos] if test_videos > 0 else []
    if test_ids:
        print(f"[splits] Test:  {len(test_ids)} videos (to be downloaded)")
    else:
        print(f"[splits] Test:  0 (no separate test set; {len(available_test)} captions available for download)")

    # Write split files
    def write_split(split_name: str, video_ids: List[str]) -> None:
        split_file = output_dir / f"{split_name}_split.jsonl"
        with open(split_file, "w") as f:
            for vid in video_ids:
                f.write(json.dumps({"video_id": vid}) + "\n")
        print(f"[splits] Wrote {split_file} ({len(video_ids)} entries)")

    write_split("train", train_ids)
    write_split("val", val_ids)
    write_split("test", test_ids)

    summary = {
        "total_existing_videos": len(existing_videos),
        "total_captions": len(all_captions),
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "test_count": len(test_ids),
        "val_split": val_split,
        "available_test_captions": len(available_test),
    }
    with open(output_dir / "splits_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[done] Splits created in {output_dir}")

    return train_ids, val_ids, test_ids


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create CoVLA dataset splits")
    parser.add_argument(
        "--videos_dir", type=str,
        default="/work/dlclarge2/lushtake-thesis/data/covla/videos",
    )
    parser.add_argument(
        "--captions_dir", type=str,
        default="/work/dlclarge2/lushtake-thesis/data/covla/captions",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/work/dlclarge2/lushtake-thesis/data/covla/splits",
    )
    parser.add_argument(
        "--val_split", type=float, default=0.1,
        help="Fraction of videos for validation (must match SAE training config)",
    )
    parser.add_argument(
        "--test_videos", type=int, default=0,
        help="Number of test videos to select from undownloaded captions (0 = none)",
    )

    args = parser.parse_args()

    create_splits(
        videos_dir=args.videos_dir,
        captions_dir=args.captions_dir,
        output_dir=args.output_dir,
        val_split=args.val_split,
        test_videos=args.test_videos,
    )
