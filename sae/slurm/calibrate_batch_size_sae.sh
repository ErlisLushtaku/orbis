#!/bin/bash
#
# SLURM Job Script: Calibrate SAE Batch Size (Self-Discovery)
#
# Finds the maximum feasible batch size for SAE training on the current GPU
# and saves the result to a resource file. Also updates the partition_map.json
# with the current partition -> GPU mapping.
#
# Self-Discovery Features:
# - Detects GPU hardware on compute node
# - Checks if hardware-specific calibration already exists (skips OOM search)
# - Updates partition_map.json for future lookups by launch.py
#
# The resource file is saved to:
#   logs_sae/resources/{model}/layer_{layer}/topk_x{exp}_k{k}/{gpu_slug}.json
#
# Usage:
#   sbatch sae/slurm/calibrate_batch_size_sae.sh --layer 22 [OPTIONS]
#
# Required:
#   --layer N          Layer to calibrate for
#
# Options:
#   --expansion N      SAE expansion factor (default: 16)
#   --k N              Top-K sparsity (default: 64)
#   --d_in N           Input dimension / hidden size (default: 768)
#   --baseline N       Starting batch size (default: 512)
#   --warmup_steps N   Training cycles per OOM test (default: 5)
#   --safety_margin F  Safety margin fraction (default: 0.10)
#   --force            Overwrite existing resource file
#
# Note: Log paths are set via sbatch CLI when called from launch.py
#

#SBATCH --job-name=sae_calibrate
#SBATCH --time=01:00:00
#SBATCH --partition=tflmb_gpu-rtx4090
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
LAYER=""
EXPANSION=16
K=64
D_IN=768
BASELINE=512
WARMUP_STEPS=5
SAFETY_MARGIN=0.10
FORCE="false"

# ------------------------------
# Parse Command Line Arguments
# ------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --layer)
            LAYER="$2"
            shift 2
            ;;
        --expansion)
            EXPANSION="$2"
            shift 2
            ;;
        --k)
            K="$2"
            shift 2
            ;;
        --d_in)
            D_IN="$2"
            shift 2
            ;;
        --baseline)
            BASELINE="$2"
            shift 2
            ;;
        --warmup_steps)
            WARMUP_STEPS="$2"
            shift 2
            ;;
        --safety_margin)
            SAFETY_MARGIN="$2"
            shift 2
            ;;
        --force)
            FORCE="true"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: sbatch calibrate_batch_size_sae.sh --layer N [--expansion N] [--k N] [--d_in N] [--baseline N] [--warmup_steps N] [--safety_margin F] [--force]"
            exit 1
            ;;
    esac
done

# ------------------------------
# Validate required arguments
# ------------------------------
if [ -z "$LAYER" ]; then
    echo "ERROR: --layer is required"
    echo "Usage: sbatch calibrate_batch_size_sae.sh --layer N [OPTIONS]"
    exit 1
fi

# ------------------------------
# Setup directories
# ------------------------------
ORBIS_ROOT="/work/dlclarge2/lushtake-thesis/orbis"
SAE_TYPE="topk_x${EXPANSION}_k${K}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="calibrate_${SAE_TYPE}_${TIMESTAMP}"

LOG_DIR="${ORBIS_ROOT}/sae/slurm/logs/calibrate/${MODEL_NAME}/layer_${LAYER}"
mkdir -p "$LOG_DIR"

# Redirect stdout and stderr to log files
exec > "${LOG_DIR}/${RUN_NAME}.out" 2> "${LOG_DIR}/${RUN_NAME}.err"

# Clean up initial SLURM logs
rm -f /work/dlclarge2/lushtake-thesis/orbis/sae/slurm/logs/slurm_init_${SLURM_JOB_ID}.out \
      /work/dlclarge2/lushtake-thesis/orbis/sae/slurm/logs/slurm_init_${SLURM_JOB_ID}.err 2>/dev/null || true

# ------------------------------
# Environment Setup
# ------------------------------
echo "=== SAE Batch Size Calibration (Self-Discovery) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Run Name: $RUN_NAME"
echo "Node: $(hostname)"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"
echo ""

export MKL_INTERFACE_LAYER="GNU"
export MKL_THREADING_LAYER="GNU"
export OMP_NUM_THREADS=1

set +u
source ~/.bashrc
conda activate orbis_env
set -u

echo "Python: $(which python)"
echo "Conda env: ${CONDA_DEFAULT_ENV:-unknown}"
echo "CUDA visible: ${CUDA_VISIBLE_DEVICES:-all}"
echo ""

echo "Calibration parameters:"
echo "  model=$MODEL_NAME"
echo "  layer=$LAYER"
echo "  sae_type=$SAE_TYPE"
echo "  d_in=$D_IN"
echo "  baseline=$BASELINE"
echo "  warmup_steps=$WARMUP_STEPS"
echo "  safety_margin=$SAFETY_MARGIN"
echo "  force=$FORCE"
echo ""
echo "Resource directory: logs_sae/resources/${MODEL_NAME}/layer_${LAYER}/${SAE_TYPE}/"
echo "Partition map: logs_sae/resources/partition_map.json"
echo "Log files: ${LOG_DIR}/${RUN_NAME}.{out,err}"
echo ""

cd "$ORBIS_ROOT"

echo "Starting batch size calibration..."
echo ""

# Build command
CALIBRATE_CMD="python -m sae.calibrate_batch_size.sae \
    --layer \"$LAYER\" \
    --expansion-factor \"$EXPANSION\" \
    --k \"$K\" \
    --d-in \"$D_IN\" \
    --baseline \"$BASELINE\" \
    --warmup-steps \"$WARMUP_STEPS\" \
    --safety-margin \"$SAFETY_MARGIN\""

# Add force flag if needed
if [ "$FORCE" = "true" ]; then
    CALIBRATE_CMD="$CALIBRATE_CMD --force"
fi

eval $CALIBRATE_CMD

echo ""
echo "=== Calibration Complete ==="
echo "End: $(date)"
