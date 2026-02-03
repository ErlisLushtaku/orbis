#!/bin/bash
#
# SLURM Job Script: Cache Activations for SAE Training
#
# This is a unified caching script for use with the orchestrator (launch.py).
# It caches activations without training an SAE.
#
# Usage:
#   sbatch sae/slurm/cache.sh --data_source nuplan [OPTIONS]
#   sbatch sae/slurm/cache.sh --data_source covla [OPTIONS]
#
# Required:
#   --data_source STR  Data source: "nuplan" or "covla"
#
# Options:
#   --layer N          Layer to extract activations from (default: 22 for nuplan, 12 for covla)
#   --num_videos N     Number of videos to use (default: 988 for nuplan, 3000 for covla)
#   --batch_size N     Batch size for caching (default: 4)
#   --seed N           Random seed for noise (default: 42)
#   --rebuild_cache    Force rebuild of activation cache
#   --run_name NAME    Override run name (default: cache_s{seed}_{timestamp})
#
# Note: Log paths are set via sbatch CLI when called from launch.py
#

#SBATCH --job-name=sae_cache
#SBATCH --time=48:00:00
#SBATCH --partition=lmbhiwidlc_gpu-rtx2080
#SBATCH --account=lmbhiwi-dlc
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/work/dlclarge2/lushtake-thesis/orbis/sae/slurm/logs/slurm_init_%j.out
#SBATCH --error=/work/dlclarge2/lushtake-thesis/orbis/sae/slurm/logs/slurm_init_%j.err

set -eo pipefail

# ------------------------------
# Default Configuration
# ------------------------------
MODEL_NAME="orbis_288x512"
DATA_SOURCE=""

# Default caching parameters (will be overridden based on data_source)
BATCH_SIZE=4
LAYER=""
SEED=42
NUM_VIDEOS=""
REBUILD_CACHE="false"
RUN_NAME=""

# ------------------------------
# Parse Command Line Arguments
# ------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_source)
            DATA_SOURCE="$2"
            shift 2
            ;;
        --layer)
            LAYER="$2"
            shift 2
            ;;
        --num_videos)
            NUM_VIDEOS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --rebuild_cache)
            REBUILD_CACHE="true"
            shift
            ;;
        --run_name)
            RUN_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: sbatch cache.sh --data_source {nuplan,covla} [--layer N] [--num_videos N] [--batch_size N] [--seed N] [--rebuild_cache] [--run_name NAME]"
            exit 1
            ;;
    esac
done

# ------------------------------
# Validate data_source
# ------------------------------
if [ -z "$DATA_SOURCE" ]; then
    echo "ERROR: --data_source is required"
    echo "Usage: sbatch cache.sh --data_source {nuplan,covla} [OPTIONS]"
    exit 1
fi

if [ "$DATA_SOURCE" != "nuplan" ] && [ "$DATA_SOURCE" != "covla" ]; then
    echo "ERROR: Invalid data_source '$DATA_SOURCE'. Must be 'nuplan' or 'covla'"
    exit 1
fi

# ------------------------------
# Set data source specific defaults
# ------------------------------
ORBIS_ROOT="/work/dlclarge2/lushtake-thesis/orbis"
EXP_DIR="${ORBIS_ROOT}/logs_wm/${MODEL_NAME}"

if [ "$DATA_SOURCE" = "nuplan" ]; then
    DATA_DIR="/work/dlcsmall2/galessos-nuPlan/nuPlan_640x360_10Hz"
    STORED_FRAME_RATE=10
    [ -z "$LAYER" ] && LAYER=22
    [ -z "$NUM_VIDEOS" ] && NUM_VIDEOS=988
    DATA_ARG="--nuplan_data_dir"
elif [ "$DATA_SOURCE" = "covla" ]; then
    DATA_DIR="/work/dlclarge2/lushtake-thesis/data/covla/videos"
    CAPTIONS_DIR="/work/dlclarge2/lushtake-thesis/data/covla/captions"
    STORED_FRAME_RATE=20
    [ -z "$LAYER" ] && LAYER=12
    [ -z "$NUM_VIDEOS" ] && NUM_VIDEOS=3000
    DATA_ARG="--covla_videos_dir"
fi

# ------------------------------
# Generate run name if not provided
# ------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ -z "$RUN_NAME" ]; then
    RUN_NAME="cache_s${SEED}_${TIMESTAMP}"
fi

# ------------------------------
# Setup log directories
# ------------------------------
LOG_DIR="${ORBIS_ROOT}/sae/slurm/logs/sae_cache/${DATA_SOURCE}/${MODEL_NAME}/layer_${LAYER}"
mkdir -p "$LOG_DIR"

# Redirect stdout and stderr to log files
exec > "${LOG_DIR}/${RUN_NAME}.out" 2> "${LOG_DIR}/${RUN_NAME}.err"

# Clean up initial SLURM logs
rm -f /work/dlclarge2/lushtake-thesis/orbis/sae/slurm/logs/slurm_init_${SLURM_JOB_ID}.out \
      /work/dlclarge2/lushtake-thesis/orbis/sae/slurm/logs/slurm_init_${SLURM_JOB_ID}.err 2>/dev/null || true

# ------------------------------
# Environment Setup
# ------------------------------
echo "=== SAE Activation Caching for ${DATA_SOURCE^^} Dataset ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Run Name: $RUN_NAME"
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

echo "Orbis dir: $ORBIS_ROOT"
echo "Exp dir: $EXP_DIR"
echo "Data dir: $DATA_DIR"
echo ""

echo "Caching parameters:"
echo "  data_source=$DATA_SOURCE"
echo "  model=$MODEL_NAME"
echo "  layer=$LAYER"
echo "  batch_size=$BATCH_SIZE"
echo "  seed=$SEED"
echo "  num_videos=$NUM_VIDEOS"
echo "  stored_frame_rate=$STORED_FRAME_RATE"
echo "  rebuild_cache=$REBUILD_CACHE"
echo ""
echo "Cache directory: logs_sae/sae_cache/${DATA_SOURCE}/${MODEL_NAME}/layer_${LAYER}/"
echo "Log files: ${LOG_DIR}/${RUN_NAME}.{out,err}"
echo ""

cd "$ORBIS_ROOT"

echo "Starting activation caching..."
echo ""

# Build command (caching only - no SAE-specific args)
CACHE_CMD="python sae/scripts/train_sae.py \
    --exp_dir \"$EXP_DIR\" \
    --data_source \"$DATA_SOURCE\" \
    $DATA_ARG \"$DATA_DIR\" \
    --num_videos \"$NUM_VIDEOS\" \
    --stored_frame_rate $STORED_FRAME_RATE \
    --input_size 288 512 \
    --batch_size \"$BATCH_SIZE\" \
    --layer \"$LAYER\" \
    --cache_seed \"$SEED\" \
    --run_name \"$RUN_NAME\" \
    --cache_only"

# Add captions dir for covla
if [ "$DATA_SOURCE" = "covla" ]; then
    CACHE_CMD="$CACHE_CMD --covla_captions_dir \"$CAPTIONS_DIR\""
fi

# Add rebuild_cache flag if needed
if [ "$REBUILD_CACHE" = "true" ]; then
    CACHE_CMD="$CACHE_CMD --rebuild_cache"
    echo "[WARNING] REBUILD_CACHE=true - existing cache will be deleted and rebuilt!"
fi

eval $CACHE_CMD

echo ""
echo "=== Caching Complete ==="
echo "End: $(date)"
