#!/bin/bash
#
# SLURM Job Script: Train SAE on NuPlan Videos
#
# Usage:
#   sbatch sae/slurm/train_nuplan.sh
#
# Logs will be saved to: sae/slurm/logs/nuplan/{model}/layer_{N}/train/{barcode}.{out,err}
#

#SBATCH --job-name=sae_nuplan
#SBATCH --time=24:00:00
#SBATCH --partition=lmbhiwidlc_gpu-rtx2080
#SBATCH --account=lmbhiwi-dlc
#SBATCH --gres=gpu:1
#SBATCH --mem=450G
#SBATCH --cpus-per-task=4
#SBATCH --output=/work/dlclarge2/lushtake-thesis/orbis/sae/slurm/logs/slurm_init_%j.out
#SBATCH --error=/work/dlclarge2/lushtake-thesis/orbis/sae/slurm/logs/slurm_init_%j.err

set -eo pipefail

# ------------------------------
# Configuration
# ------------------------------
EXP_DIR="/work/dlclarge2/lushtake-thesis/orbis/logs_wm/orbis_288x512"
NUPLAN_DATA="/work/dlcsmall2/galessos-nuPlan/nuPlan_640x360_10Hz"
DATA_SOURCE="nuplan"
MODEL_NAME="orbis_288x512"

# Training parameters
BATCH_SIZE=4          # Reduced for 10GB GPU (RTX 2080); used during caching phase
NUM_EPOCHS=50
LAYER=22
K=64
EXPANSION_FACTOR=16
SEED=42
NUM_VIDEOS=988  # All NuPlan videos (~160,000 clips at ~162 clips/video)

# Cache behavior (set to "true" if changing NUM_VIDEOS)
REBUILD_CACHE="false"

# ------------------------------
# Generate unique barcode
# ------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP="$EXPANSION_FACTOR"
BARCODE="topk_x${EXP}_k${K}_s${SEED}_${TIMESTAMP}"

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
echo "=== SAE Training on NuPlan Dataset ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Barcode: $BARCODE"
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

echo "Training parameters:"
echo "  data_source=$DATA_SOURCE"
echo "  model=$MODEL_NAME"
echo "  layer=$LAYER"
echo "  batch_size=$BATCH_SIZE"
echo "  num_epochs=$NUM_EPOCHS"
echo "  k=$K"
echo "  expansion_factor=$EXPANSION_FACTOR"
echo "  seed=$SEED"
echo "  num_videos=$NUM_VIDEOS"
echo "  rebuild_cache=$REBUILD_CACHE"
echo ""
echo "Output paths:"
echo "  Run dir: logs_sae/runs/${DATA_SOURCE}/${MODEL_NAME}/layer_${LAYER}/${BARCODE}/"
echo "  Log dir: ${LOG_DIR}/${BARCODE}.{out,err}"
echo ""

echo "Starting SAE training..."
echo ""

# Build command
TRAIN_CMD="python sae/scripts/train_sae.py \
    --exp_dir \"$EXP_DIR\" \
    --data_source \"$DATA_SOURCE\" \
    --nuplan_data_dir \"$NUPLAN_DATA\" \
    --num_videos \"$NUM_VIDEOS\" \
    --stored_frame_rate 10 \
    --input_size 288 512 \
    --batch_size \"$BATCH_SIZE\" \
    --num_epochs \"$NUM_EPOCHS\" \
    --k \"$K\" \
    --expansion_factor \"$EXPANSION_FACTOR\" \
    --layer \"$LAYER\" \
    --seed \"$SEED\" \
    --run_name \"$BARCODE\""

# Add rebuild_cache flag if needed
if [ "$REBUILD_CACHE" = "true" ]; then
    TRAIN_CMD="$TRAIN_CMD --rebuild_cache"
    echo "[WARNING] REBUILD_CACHE=true - existing cache will be deleted and rebuilt!"
fi

# Append any arguments passed to sbatch (e.g., --layer 12 --k 128)
TRAIN_CMD="$TRAIN_CMD $@"

eval $TRAIN_CMD

echo ""
echo "=== Training Complete ==="
echo "End: $(date)"
