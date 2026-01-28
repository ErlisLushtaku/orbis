#!/bin/bash
#
# SLURM Job Script: Train SAE on CoVLA Videos
#
# Usage:
#   sbatch sae/slurm/train_covla.sh [OPTIONS]
#
# Options (override defaults):
#   --layer N              Layer to extract activations from (default: 12)
#   --k N                  Top-K sparsity (default: 32)
#   --expansion N          SAE expansion factor (default: 10)
#   --num_videos N         Number of videos to use (default: 3000)
#   --epochs N             Number of training epochs (default: 5)
#   --batch_size N         Batch size for caching (default: 4)
#   --sae_batch_mult N     SAE batch multiplier (default: 1024)
#   --seed N               Random seed (default: 42)
#   --max_tokens N         Limit tokens used for training (default: all)
#   --rebuild_cache        Force rebuild of activation cache
#   --no_streaming         Disable streaming mode (load all into RAM)
#   --train_only           Skip caching, assume cache exists (used by orchestrator)
#   --barcode NAME         Use specific barcode instead of auto-generating
#
# Examples:
#   sbatch sae/slurm/train_covla.sh --layer 22
#   sbatch sae/slurm/train_covla.sh --layer 12 --k 64 --expansion 16
#   sbatch sae/slurm/train_covla.sh --num_videos 100 --no_streaming
#
# Logs will be saved to: sae/slurm/logs/covla/{model}/layer_{N}/train/{barcode}.{out,err}
#

#SBATCH --job-name=sae_covla
#SBATCH --time=24:00:00
#SBATCH --partition=lmbhiwidlc_gpu-rtx2080
#SBATCH --account=lmbhiwi-dlc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/work/dlclarge2/lushtake-thesis/orbis/sae/slurm/logs/slurm_init_%j.out
#SBATCH --error=/work/dlclarge2/lushtake-thesis/orbis/sae/slurm/logs/slurm_init_%j.err

set -eo pipefail

# ------------------------------
# Default Configuration
# ------------------------------
DATA_SOURCE="covla"
MODEL_NAME="orbis_288x512"

# Default training parameters (can be overridden via command line)
LAYER=22
BATCH_SIZE=4
NUM_EPOCHS=5
K=32
EXPANSION_FACTOR=10
SAE_BATCH_MULTIPLIER=1024
SEED=42
NUM_VIDEOS=3000
MAX_TOKENS=""
REBUILD_CACHE="false"
STREAMING="true"
TRAIN_ONLY="false"
BARCODE=""

# ------------------------------
# Parse Command Line Arguments
# ------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --layer)
            LAYER="$2"
            shift 2
            ;;
        --k)
            K="$2"
            shift 2
            ;;
        --expansion)
            EXPANSION_FACTOR="$2"
            shift 2
            ;;
        --num_videos)
            NUM_VIDEOS="$2"
            shift 2
            ;;
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --sae_batch_mult)
            SAE_BATCH_MULTIPLIER="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --max_tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --rebuild_cache)
            REBUILD_CACHE="true"
            shift
            ;;
        --no_streaming)
            STREAMING="false"
            shift
            ;;
        --train_only)
            TRAIN_ONLY="true"
            shift
            ;;
        --barcode)
            BARCODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: sbatch train_covla.sh [--layer N] [--k N] [--expansion N] [--num_videos N] [--epochs N] [--batch_size N] [--sae_batch_mult N] [--seed N] [--max_tokens N] [--rebuild_cache] [--no_streaming] [--train_only] [--barcode NAME]"
            exit 1
            ;;
    esac
done

# ------------------------------
# Generate unique barcode (or use provided one)
# ------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ -z "$BARCODE" ]; then
    BARCODE="topk_x${EXPANSION_FACTOR}_k${K}_s${SEED}_${TIMESTAMP}"
fi

# ------------------------------
# Setup log directories
# ------------------------------
ORBIS_ROOT="/work/dlclarge2/lushtake-thesis/orbis"
LOG_DIR="${ORBIS_ROOT}/sae/slurm/logs/${DATA_SOURCE}/${MODEL_NAME}/layer_${LAYER}/train"
mkdir -p "$LOG_DIR"

# Redirect stdout and stderr to log files
exec > "${LOG_DIR}/${BARCODE}.out" 2> "${LOG_DIR}/${BARCODE}.err"

# Clean up initial SLURM logs
rm -f /work/dlclarge2/lushtake-thesis/orbis/sae/slurm/logs/slurm_init_${SLURM_JOB_ID}.out \
      /work/dlclarge2/lushtake-thesis/orbis/sae/slurm/logs/slurm_init_${SLURM_JOB_ID}.err 2>/dev/null || true

# ------------------------------
# Environment Setup
# ------------------------------
echo "=== SAE Training on CoVLA Dataset ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Barcode: $BARCODE"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"
echo ""

export MKL_INTERFACE_LAYER="GNU"
export MKL_THREADING_LAYER="GNU"
export TK_WORK_DIR="${ORBIS_ROOT}/logs_tk"
export OMP_NUM_THREADS=1

set +u
source ~/.bashrc
conda activate orbis_env
set -u

echo "Python: $(which python)"
echo "Conda env: ${CONDA_DEFAULT_ENV:-unknown}"
echo "CUDA visible: ${CUDA_VISIBLE_DEVICES:-all}"
echo ""

# ------------------------------
# Paths
# ------------------------------
EXP_DIR="${ORBIS_ROOT}/logs_wm/${MODEL_NAME}"
COVLA_VIDEOS="/work/dlclarge2/lushtake-thesis/data/covla/videos"
COVLA_CAPTIONS="/work/dlclarge2/lushtake-thesis/data/covla/captions"

cd "$ORBIS_ROOT"

echo "Orbis dir: $ORBIS_ROOT"
echo "Exp dir: $EXP_DIR"
echo "CoVLA videos: $COVLA_VIDEOS"
echo "CoVLA captions: $COVLA_CAPTIONS"
echo ""

echo "Training parameters:"
echo "  data_source=$DATA_SOURCE"
echo "  model=$MODEL_NAME"
echo "  layer=$LAYER"
echo "  batch_size=$BATCH_SIZE"
echo "  sae_batch_multiplier=$SAE_BATCH_MULTIPLIER"
echo "  num_epochs=$NUM_EPOCHS"
echo "  k=$K"
echo "  expansion_factor=$EXPANSION_FACTOR"
echo "  seed=$SEED"
echo "  num_videos=$NUM_VIDEOS"
echo "  max_tokens=$MAX_TOKENS"
echo "  rebuild_cache=$REBUILD_CACHE"
echo "  streaming=$STREAMING"
echo "  train_only=$TRAIN_ONLY"
echo ""
echo "Output paths:"
echo "  Run dir: logs_sae/runs/${DATA_SOURCE}/${MODEL_NAME}/layer_${LAYER}/${BARCODE}/"
echo "  Log dir: ${LOG_DIR}/${BARCODE}.{out,err}"
echo ""

echo "Starting SAE training..."
echo ""

# ------------------------------
# Build and run training command
# ------------------------------
TRAIN_CMD="python sae/scripts/train_sae.py \
    --exp_dir \"$EXP_DIR\" \
    --data_source \"$DATA_SOURCE\" \
    --covla_videos_dir \"$COVLA_VIDEOS\" \
    --covla_captions_dir \"$COVLA_CAPTIONS\" \
    --num_videos \"$NUM_VIDEOS\" \
    --stored_frame_rate 20 \
    --input_size 288 512 \
    --batch_size \"$BATCH_SIZE\" \
    --sae_batch_multiplier \"$SAE_BATCH_MULTIPLIER\" \
    --num_epochs \"$NUM_EPOCHS\" \
    --k \"$K\" \
    --expansion_factor \"$EXPANSION_FACTOR\" \
    --layer \"$LAYER\" \
    --seed \"$SEED\" \
    --run_name \"$BARCODE\""

# Add streaming flag if enabled
if [ "$STREAMING" = "true" ]; then
    TRAIN_CMD="$TRAIN_CMD --streaming"
fi

# Add rebuild_cache flag if needed
if [ "$REBUILD_CACHE" = "true" ]; then
    TRAIN_CMD="$TRAIN_CMD --rebuild_cache"
    echo "[WARNING] REBUILD_CACHE=true - existing cache will be deleted and rebuilt!"
fi

# Add max_tokens if set
if [ -n "$MAX_TOKENS" ]; then
    TRAIN_CMD="$TRAIN_CMD --max_tokens $MAX_TOKENS"
    echo "[INFO] Using max_tokens=$MAX_TOKENS (subset of cached data)"
fi

# Add train_only flag if needed
if [ "$TRAIN_ONLY" = "true" ]; then
    TRAIN_CMD="$TRAIN_CMD --train_only"
    echo "[INFO] TRAIN_ONLY=true - skipping caching, assuming cache exists"
fi

eval $TRAIN_CMD

echo ""
echo "=== Training Complete ==="
echo "End: $(date)"
