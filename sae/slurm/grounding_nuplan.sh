#!/bin/bash
#
# SLURM Job Script: Semantic Grounding Analysis on NuPlan
#
# Usage:
#   sbatch sae/slurm/grounding_nuplan.sh
#

#SBATCH --job-name=grounding_nuplan
#SBATCH --time=6:00:00
#SBATCH --partition=lmbhiwidlc_gpu-rtx2080
#SBATCH --account=lmbhiwi-dlc
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# ------------------------------
# Configuration - EDIT THESE
# ------------------------------
# Path to trained SAE run directory (contains best_sae.pt)
SAE_RUN_DIR="/work/dlclarge2/lushtake-thesis/orbis/logs_sae/runs/nuplan/orbis_288x512/layer_12/EDIT_ME"

# Orbis experiment directory
EXP_DIR="/work/dlclarge2/lushtake-thesis/orbis/logs_wm/orbis_288x512"

# NuPlan data
NUPLAN_DATA="/work/dlcsmall2/galessos-nuPlan/nuPlan_640x360_10Hz"

# Analysis parameters
LAYER=12
BATCH_SIZE=4
NUM_FRAMES=6
NUM_TEST_VIDEOS=100  # Number of test videos
SKIP_VIDEOS=400      # Skip first N videos used for training

# ------------------------------
# Derived paths
# ------------------------------
SAE_CHECKPOINT="${SAE_RUN_DIR}/best_sae.pt"
OUTPUT_DIR="${SAE_RUN_DIR}/analysis_nuplan"

# Generate barcode for logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BARCODE="analysis_${TIMESTAMP}"

# ------------------------------
# Setup log directories
# ------------------------------
ORBIS_ROOT="/work/dlclarge2/lushtake-thesis/orbis"
MODEL_NAME=$(basename "$EXP_DIR")
LOG_DIR="${ORBIS_ROOT}/sae/slurm/logs/nuplan/${MODEL_NAME}/layer_${LAYER}/eval"
mkdir -p "$LOG_DIR"

# Redirect stdout and stderr
exec > "${LOG_DIR}/${BARCODE}.out" 2> "${LOG_DIR}/${BARCODE}.err"

# ------------------------------
# Environment Setup
# ------------------------------
echo "=== SLURM Job: Semantic Grounding Analysis (NuPlan) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Barcode: $BARCODE"
echo ""

source ~/.bashrc
conda activate orbis_env

cd "$ORBIS_ROOT"

export TK_WORK_DIR="${ORBIS_ROOT}/logs_tk"
export OMP_NUM_THREADS=1

echo "Environment:"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  TK_WORK_DIR=$TK_WORK_DIR"
echo ""

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo "Configuration:"
echo "  sae_checkpoint=$SAE_CHECKPOINT"
echo "  exp_dir=$EXP_DIR"
echo "  nuplan_data_dir=$NUPLAN_DATA"
echo "  output_dir=$OUTPUT_DIR"
echo "  layer=$LAYER"
echo "  batch_size=$BATCH_SIZE"
echo "  num_frames=$NUM_FRAMES"
echo "  num_test_videos=$NUM_TEST_VIDEOS"
echo "  skip_videos=$SKIP_VIDEOS"
echo ""

# Verify checkpoint exists
if [ ! -f "$SAE_CHECKPOINT" ]; then
    echo "ERROR: SAE checkpoint not found: $SAE_CHECKPOINT"
    echo "Please edit SAE_RUN_DIR in this script to point to a valid training run."
    exit 1
fi

python sae/scripts/semantic_grounding_nuplan.py \
    --exp_dir "$EXP_DIR" \
    --sae_checkpoint "$SAE_CHECKPOINT" \
    --nuplan_data_dir "$NUPLAN_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --layer "$LAYER" \
    --batch_size "$BATCH_SIZE" \
    --num_frames "$NUM_FRAMES" \
    --num_test_videos "$NUM_TEST_VIDEOS" \
    --skip_videos "$SKIP_VIDEOS" \
    --top_k 30

echo ""
echo "=== Analysis Complete ==="
echo "Finished: $(date)"
echo "Results: $OUTPUT_DIR"
