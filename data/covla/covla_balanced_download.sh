#!/bin/bash
#SBATCH --job-name=covla_balanced_download
#SBATCH --output=covla_balanced_download.out
#SBATCH --error=covla_balanced_download.err
#SBATCH --time=24:00:00
#SBATCH --partition=lmbhiwi_gpu-rtx2080
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -eo pipefail

echo "=== CoVLA Balanced Downloader (direct MP4 download by video_id) ==="
echo "Node: $(hostname)"
echo "Start: $(date)"

# Filenames are <video_id>.jsonl
CAPTIONS_DIR="/work/dlclarge2/alidemaa-text-control-orbis/orbis/data/covla_captions_balanced"

# Output mp4s: <video_id>.mp4
TARGET_DIR="/work/dlclarge2/alidemaa-text-control-orbis/orbis/data/covla_videos_balanced"
mkdir -p "$TARGET_DIR"

# Optional: pin to a dataset commit. Empty => default branch head.
COVLA_REVISION="0a6d39e41659903a26dde957744e70dbc360bb6d"

# Local cache for downloads (keeps partials + resumes)
HF_LOCAL_CACHE="${TARGET_DIR}/_hf_download_cache"
mkdir -p "$HF_LOCAL_CACHE"

# ------------------------------
# Activate Conda (fix MKL_INTERFACE_LAYER unbound variable)
# ------------------------------
# Some conda activate.d scripts reference MKL_INTERFACE_LAYER; with "set -u" this crashes if unset.
export MKL_INTERFACE_LAYER="GNU"
export MKL_THREADING_LAYER="GNU"
set +u
source ~/.bashrc
conda activate orbis_env

echo "Using Python: $(which python)"
echo "Conda env: ${CONDA_DEFAULT_ENV:-unknown}"
echo "MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER}"

# Pass vars into the python heredoc via environment
export CAPTIONS_DIR
export TARGET_DIR
export COVLA_REVISION
export HF_LOCAL_CACHE

python3 << 'EOF'
import os
import sys
import time
import shutil
from pathlib import Path

CAPTIONS_DIR = Path(os.environ["CAPTIONS_DIR"])
OUT_DIR      = Path(os.environ["TARGET_DIR"])
REVISION     = os.environ.get("COVLA_REVISION", "").strip() or None
CACHE_DIR    = Path(os.environ.get("HF_LOCAL_CACHE", str(OUT_DIR / "_hf_download_cache")))

OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

print("[INFO] Python version:", sys.version)
print("[INFO] CAPTIONS_DIR:", CAPTIONS_DIR)
print("[INFO] OUT_DIR:", OUT_DIR)
print("[INFO] REVISION:", REVISION)
print("[INFO] CACHE_DIR:", CACHE_DIR)

try:
    from huggingface_hub import hf_hub_download
except Exception as e:
    print("[FATAL] Could not import huggingface_hub. Install it in this env:")
    print("  pip install -U huggingface_hub")
    print("Error:", e)
    sys.exit(1)

caption_files = sorted(CAPTIONS_DIR.rglob("*.jsonl"))
video_ids = [p.stem for p in caption_files]

if not video_ids:
    print(f"[FATAL] No .jsonl files found under: {CAPTIONS_DIR}")
    sys.exit(1)

# Resume-friendly: skip already-downloaded mp4s
wanted = []
for vid in video_ids:
    out_mp4 = OUT_DIR / f"{vid}.mp4"
    if out_mp4.exists() and out_mp4.stat().st_size > 0:
        continue
    wanted.append(vid)

print(f"[INFO] Caption files (requested): {len(video_ids)}")
print(f"[INFO] Remaining to download: {len(wanted)}")

if not wanted:
    print("[DONE] Nothing to download. All videos already exist.")
    sys.exit(0)

REPO_ID = "turing-motors/CoVLA-Dataset"
REPO_TYPE = "dataset"

missing = []
failed = []
saved = 0

def safe_copy(src: Path, dst: Path):
    dst_tmp = dst.with_suffix(dst.suffix + ".partial")
    shutil.copyfile(src, dst_tmp)
    os.replace(dst_tmp, dst)

def download_one(vid: str, retries: int = 3, sleep_s: float = 2.0):
    filename = f"videos/{vid}.mp4"
    out_mp4 = OUT_DIR / f"{vid}.mp4"

    for attempt in range(1, retries + 1):
        try:
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                filename=filename,
                revision=REVISION,
                local_dir=str(CACHE_DIR),
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

print("\n[DONE]")
print(f"  Requested: {len(video_ids)}")
print(f"  Newly saved: {saved}")
print(f"  Missing (not found): {len(missing)}")
print(f"  Failed (other errors): {len(failed)}")

if missing:
    p = OUT_DIR / "_missing_video_ids.txt"
    p.write_text("\n".join(missing) + "\n")
    print(f"[WARN] Wrote missing IDs to: {p}")

if failed:
    p = OUT_DIR / "_failed_video_ids.txt"
    lines = [f"{vid}\t{err}" for vid, err in failed]
    p.write_text("\n".join(lines) + "\n")
    print(f"[WARN] Wrote failed IDs to: {p}")

EOF

echo "Finished: $(date)"
