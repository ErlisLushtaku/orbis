#!/usr/bin/env python3
"""
CoVLA Dataset Video Downloader

Downloads videos from the CoVLA HuggingFace dataset based on existing caption files
or a split file.

Usage:
    # Download all videos matching captions
    python download_covla.py --captions_dir /path/to/captions --output_dir /path/to/videos
    
    # Download only first 100 videos
    python download_covla.py --captions_dir /path/to/captions --output_dir /path/to/videos --num_videos 100
    
    # Download videos from a split file
    python download_covla.py --split_file /path/to/test_split.jsonl --output_dir /path/to/videos
"""

import os
import sys
import time
import json
import shutil
import argparse
from pathlib import Path


def download_covla_videos(
    output_dir: str,
    captions_dir: str | None = None,
    split_file: str | None = None,
    num_videos: int | None = None,
    resume: bool = True,
):
    """
    Download CoVLA videos based on existing caption files or a split file.
    
    Args:
        output_dir: Directory to save downloaded videos
        captions_dir: Directory containing <video_id>.jsonl caption files
        split_file: Path to split file (jsonl with video_id per line)
        num_videos: Number of videos to download (None = all)
        resume: Skip already downloaded files
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[ERROR] huggingface_hub not installed. Install with:")
        print("  pip install huggingface_hub")
        sys.exit(1)
    
    output_dir = Path(output_dir)
    
    # Create directories
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    cache_dir = output_dir / "_hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    REPO_ID = "turing-motors/CoVLA-Dataset"
    REPO_TYPE = "dataset"
    
    # Get video IDs from split file OR caption files
    if split_file:
        split_file = Path(split_file)
        video_ids = []
        with open(split_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                video_ids.append(entry["video_id"])
        print(f"[INFO] Loaded {len(video_ids)} video IDs from split file: {split_file}")
    elif captions_dir:
        captions_dir = Path(captions_dir)
        caption_files = sorted(captions_dir.glob("*.jsonl"))
        video_ids = [p.stem for p in caption_files]
        
        if not video_ids:
            print(f"[FATAL] No .jsonl files found in: {captions_dir}")
            sys.exit(1)
        
        print(f"[INFO] Found {len(video_ids)} caption files in: {captions_dir}")
    else:
        print("[FATAL] Must provide either --captions_dir or --split_file")
        sys.exit(1)
    
    print(f"[INFO] Output directory: {videos_dir}")
    
    # Limit number of videos if specified
    if num_videos is not None:
        video_ids = video_ids[:num_videos]
        print(f"[INFO] Limiting to first {num_videos} videos")
    
    # Filter out already downloaded videos if resume mode
    if resume:
        wanted = []
        for vid in video_ids:
            out_mp4 = videos_dir / f"{vid}.mp4"
            if out_mp4.exists() and out_mp4.stat().st_size > 0:
                continue
            wanted.append(vid)
        print(f"[INFO] Already downloaded: {len(video_ids) - len(wanted)}")
        print(f"[INFO] Remaining to download: {len(wanted)}")
    else:
        wanted = video_ids
        print(f"[INFO] Videos to download: {len(wanted)}")
    
    if not wanted:
        print("[DONE] Nothing to download. All videos already exist.")
        return
    
    def safe_copy(src: Path, dst: Path):
        """Atomic copy with temporary file"""
        dst_tmp = dst.with_suffix(dst.suffix + ".partial")
        shutil.copyfile(src, dst_tmp)
        os.replace(dst_tmp, dst)
    
    def download_one(vid: str, retries: int = 3, sleep_s: float = 2.0) -> tuple[bool, str | None]:
        """Download a single video with retries"""
        filename = f"videos/{vid}.mp4"
        out_mp4 = videos_dir / f"{vid}.mp4"
        
        for attempt in range(1, retries + 1):
            try:
                local_path = hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type=REPO_TYPE,
                    filename=filename,
                    local_dir=str(cache_dir),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
                local_path = Path(local_path)
                
                if not local_path.exists() or local_path.stat().st_size == 0:
                    raise RuntimeError(f"Downloaded file missing/empty: {local_path}")
                
                safe_copy(local_path, out_mp4)
                
                if out_mp4.exists() and out_mp4.stat().st_size > 0:
                    return True, None
                
                raise RuntimeError(f"Output file missing/empty after copy: {out_mp4}")
                
            except Exception as e:
                if attempt < retries:
                    time.sleep(sleep_s * attempt)
                    continue
                return False, str(e)
        
        return False, "Unknown error"
    
    # Download loop
    missing = []
    failed = []
    saved = 0
    total = len(wanted)
    t0 = time.time()
    
    for i, vid in enumerate(wanted, start=1):
        if i % 25 == 0 or i == 1:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate if rate > 0 else float("inf")
            print(f"[PROGRESS] {i}/{total} | saved={saved} | elapsed={elapsed/60:.1f}m | eta={eta/60:.1f}m")
        
        ok, err = download_one(vid, retries=3)
        if ok:
            saved += 1
        else:
            msg = (err or "").lower()
            if "404" in msg or "not found" in msg or "entry not found" in msg:
                missing.append(vid)
                print(f"[MISSING] {vid} -> {err}")
            else:
                failed.append((vid, err))
                print(f"[ERROR] {vid} -> {err}")
    
    # Summary
    elapsed = time.time() - t0
    print("\n" + "="*60)
    print("[DONE] Download complete!")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Requested: {len(video_ids)}")
    print(f"  Newly saved: {saved}")
    print(f"  Missing (not found in repo): {len(missing)}")
    print(f"  Failed (other errors): {len(failed)}")
    
    if missing:
        p = videos_dir / "_missing_video_ids.txt"
        p.write_text("\n".join(missing) + "\n")
        print(f"[WARN] Wrote missing IDs to: {p}")
    
    if failed:
        p = videos_dir / "_failed_video_ids.txt"
        lines = [f"{vid}\t{err}" for vid, err in failed]
        p.write_text("\n".join(lines) + "\n")
        print(f"[WARN] Wrote failed IDs to: {p}")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Download CoVLA videos based on caption files or split file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download all videos matching captions
    python download_covla.py \\
        --captions_dir /work/dlclarge2/lushtake-thesis/data/covla/captions \\
        --output_dir /work/dlclarge2/lushtake-thesis/data/covla
    
    # Download only first 100 videos
    python download_covla.py \\
        --captions_dir /work/dlclarge2/lushtake-thesis/data/covla/captions \\
        --output_dir /work/dlclarge2/lushtake-thesis/data/covla \\
        --num_videos 100
    
    # Download test split videos
    python download_covla.py \\
        --split_file /work/dlclarge2/lushtake-thesis/data/covla/splits/test_split.jsonl \\
        --output_dir /work/dlclarge2/lushtake-thesis/data/covla
        """
    )
    
    parser.add_argument(
        "--captions_dir", "-c",
        type=str,
        default=None,
        help="Directory containing caption JSONL files"
    )
    parser.add_argument(
        "--split_file", "-s",
        type=str,
        default=None,
        help="Path to split file (jsonl with video_id per line)"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="/work/dlclarge2/lushtake-thesis/data/covla",
        help="Output directory (videos will be saved in <output_dir>/videos/)"
    )
    parser.add_argument(
        "--num_videos", "-n",
        type=int,
        default=None,
        help="Number of videos to download (default: all)"
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Don't skip existing files, re-download everything"
    )
    
    args = parser.parse_args()
    
    # Default to captions_dir if neither provided
    if not args.split_file and not args.captions_dir:
        args.captions_dir = "/work/dlclarge2/lushtake-thesis/data/covla/captions"
    
    download_covla_videos(
        output_dir=args.output_dir,
        captions_dir=args.captions_dir,
        split_file=args.split_file,
        num_videos=args.num_videos,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
