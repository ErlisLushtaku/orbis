#!/bin/bash
#
# SLURM Job Script: Cache Activations for NuPlan Videos
#
# This is a caching-only script for use with the orchestrator (launch.py).
# It caches activations without training an SAE.
#
# Usage:
#   sbatch sae/slurm/cache_nuplan.sh [OPTIONS]
#
# Options:
#   --layer N          Layer to extract activations from (default: 22)
#   --num_videos N     Number of videos to use (default: 988)
#   --batch_size N     Batch size for caching (default: 4)
#   --seed N           Random seed for noise (default: 42)
#   --rebuild_cache    Force rebuild of activation cache
#   --run_name NAME    Override run name (default: cache_s{seed}_{timestamp})
#
# Note: Log paths are set via sbatch CLI when called from launch.py
#

#SBATCH --job-name=sae_cache_nuplan
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
EXP_DIR="/work/dlclarge2/lushtake-thesis/orbis/logs_wm/orbis_288x512"
NUPLAN_DATA="/work/dlcsmall2/galessos-nuPlan/nuPlan_640x360_10Hz"
DATA_SOURCE="nuplan"
MODEL_NAME="orbis_288x512"

# Default caching parameters
BATCH_SIZE=4
LAYER=22
SEED=42
NUM_VIDEOS=988
REBUILD_CACHE="false"
RUN_NAME=""

# ------------------------------
# Parse Command Line Arguments
# ------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
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
            echo "Usage: sbatch cache_nuplan.sh [--layer N] [--num_videos N] [--batch_size N] [--seed N] [--rebuild_cache] [--run_name NAME]"
            exit 1
            ;;
    esac
done

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
ORBIS_ROOT="/work/dlclarge2/lushtake-thesis/orbis"
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
echo "=== SAE Activation Caching for NuPlan Dataset ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Run Name: $RUN_NAME"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

source ~/.bashrc
conda activate orbis_env

cd "$ORBIS_ROOT"

# Set tokenizer work directory
export TK_WORK_DIR="${ORBIS_ROOT}/logs_tk"
export OMP_NUM_THREADS=1

echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "CUDA visible: $CUDA_VISIBLE_DEVICES"
echo ""

echo "Orbis dir: $ORBIS_ROOT"
echo "Exp dir: $EXP_DIR"
echo "NuPlan data: $NUPLAN_DATA"
echo ""

echo "Caching parameters:"
echo "  data_source=$DATA_SOURCE"
echo "  model=$MODEL_NAME"
echo "  layer=$LAYER"
echo "  batch_size=$BATCH_SIZE"
echo "  seed=$SEED"
echo "  num_videos=$NUM_VIDEOS"
echo "  rebuild_cache=$REBUILD_CACHE"
echo ""
echo "Cache directory: logs_sae/sae_cache/${DATA_SOURCE}/${MODEL_NAME}/layer_${LAYER}/"
echo "Log files: ${LOG_DIR}/${RUN_NAME}.{out,err}"
echo ""

echo "Starting activation caching..."
echo ""

# Build command (caching only - no SAE-specific args)
CACHE_CMD="python sae/scripts/train_sae.py \
    --exp_dir \"$EXP_DIR\" \
    --data_source \"$DATA_SOURCE\" \
    --nuplan_data_dir \"$NUPLAN_DATA\" \
    --num_videos \"$NUM_VIDEOS\" \
    --stored_frame_rate 10 \
    --input_size 288 512 \
    --batch_size \"$BATCH_SIZE\" \
    --layer \"$LAYER\" \
    --cache_seed \"$SEED\" \
    --run_name \"$RUN_NAME\" \
    --cache_only"

# Add rebuild_cache flag if needed
if [ "$REBUILD_CACHE" = "true" ]; then
    CACHE_CMD="$CACHE_CMD --rebuild_cache"
    echo "[WARNING] REBUILD_CACHE=true - existing cache will be deleted and rebuilt!"
fi

eval $CACHE_CMD

echo ""
echo "=== Caching Complete ==="
echo "End: $(date)"
