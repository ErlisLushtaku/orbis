#!/bin/bash
#
# SLURM Job Script: Calibrate Orbis Caching Batch Size (Self-Discovery)
#
# Finds the maximum feasible batch size (in clips) for Orbis activation caching
# on the current GPU and saves the result to a resource file. Also updates the
# partition_map.json with the current partition -> GPU mapping.
#
# IMPORTANT: batch_size refers to the number of CLIPS (video sequences),
# NOT individual frames. Each clip contains num_frames sequential frames.
#
# Self-Discovery Features:
# - Detects GPU hardware on compute node
# - Checks if hardware-specific calibration already exists (skips OOM search)
# - Updates partition_map.json for future lookups by launch.py
#
# The resource file is saved to:
#   resources/{model}/layer_{layer}/{gpu_slug}.json
#
# Usage:
#   sbatch sae/slurm/calibrate_batch_size_orbis.sh --orbis_exp_dir /path/to/orbis/exp --layer 22 [OPTIONS]
#
# Required:
#   --orbis_exp_dir PATH   Path to Orbis experiment directory
#   --layer N              Layer number to extract activations from
#
# Options:
#   --input_size H W       Input frame size (default: 288 512)
#   --num_frames N         Number of frames per clip (default: 6)
#   --model_name NAME      Model name for organization (default: orbis_288x512)
#   --baseline N           Starting batch size in clips (default: 4)
#   --warmup_steps N       Forward passes per OOM test (default: 5)
#   --safety_margin F      Safety margin fraction (default: 0.10)
#   --force                Overwrite existing resource file
#
# Note: Log paths are set via sbatch CLI when called from launch.py
#

#SBATCH --job-name=orbis_calibrate
#SBATCH --time=01:00:00
#SBATCH --partition=lmbhiwidlc_gpu-rtx2080
#SBATCH --account=lmbhiwi-dlc
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=/work/dlclarge2/lushtake-thesis/orbis/sae/slurm/logs/slurm_init_%j.out
#SBATCH --error=/work/dlclarge2/lushtake-thesis/orbis/sae/slurm/logs/slurm_init_%j.err

set -eo pipefail

# ------------------------------
# Default Configuration
# ------------------------------
MODEL_NAME="orbis_288x512"
ORBIS_EXP_DIR=""
LAYER=""
INPUT_H=288
INPUT_W=512
NUM_FRAMES=6
BASELINE=4
WARMUP_STEPS=5
SAFETY_MARGIN=0.10
FORCE="false"

# ------------------------------
# Parse Command Line Arguments
# ------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --orbis_exp_dir)
            ORBIS_EXP_DIR="$2"
            shift 2
            ;;
        --layer)
            LAYER="$2"
            shift 2
            ;;
        --input_size)
            INPUT_H="$2"
            INPUT_W="$3"
            shift 3
            ;;
        --num_frames)
            NUM_FRAMES="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
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
            echo "Usage: sbatch calibrate_batch_size_orbis.sh --orbis_exp_dir PATH --layer N [OPTIONS]"
            exit 1
            ;;
    esac
done

# ------------------------------
# Validate required arguments
# ------------------------------
if [ -z "$ORBIS_EXP_DIR" ]; then
    echo "ERROR: --orbis_exp_dir is required"
    echo "Usage: sbatch calibrate_batch_size_orbis.sh --orbis_exp_dir PATH --layer N [OPTIONS]"
    exit 1
fi

if [ -z "$LAYER" ]; then
    echo "ERROR: --layer is required"
    echo "Usage: sbatch calibrate_batch_size_orbis.sh --orbis_exp_dir PATH --layer N [OPTIONS]"
    exit 1
fi

if [ ! -d "$ORBIS_EXP_DIR" ]; then
    echo "ERROR: Orbis experiment directory not found: $ORBIS_EXP_DIR"
    exit 1
fi

# ------------------------------
# Setup directories
# ------------------------------
ORBIS_ROOT="/work/dlclarge2/lushtake-thesis/orbis"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="calibrate_orbis_${INPUT_H}x${INPUT_W}_${TIMESTAMP}"

LOG_DIR="${ORBIS_ROOT}/sae/slurm/logs/calibrate_orbis/${MODEL_NAME}"
mkdir -p "$LOG_DIR"

# Redirect stdout and stderr to log files
exec > "${LOG_DIR}/${RUN_NAME}.out" 2> "${LOG_DIR}/${RUN_NAME}.err"

# Clean up initial SLURM logs
rm -f /work/dlclarge2/lushtake-thesis/orbis/sae/slurm/logs/slurm_init_${SLURM_JOB_ID}.out \
      /work/dlclarge2/lushtake-thesis/orbis/sae/slurm/logs/slurm_init_${SLURM_JOB_ID}.err 2>/dev/null || true

# ------------------------------
# Environment Setup
# ------------------------------
echo "=== Orbis Caching Batch Size Calibration (Self-Discovery) ==="
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
echo "  orbis_exp_dir=$ORBIS_EXP_DIR"
echo "  layer=$LAYER"
echo "  model_name=$MODEL_NAME"
echo "  input_size=${INPUT_H}x${INPUT_W}"
echo "  num_frames=$NUM_FRAMES"
echo "  baseline=$BASELINE clips"
echo "  warmup_steps=$WARMUP_STEPS"
echo "  safety_margin=$SAFETY_MARGIN"
echo "  force=$FORCE"
echo ""
echo "Resource directory: resources/${MODEL_NAME}/layer_${LAYER}/"
echo "Partition map: resources/partition_map.json"
echo "Log files: ${LOG_DIR}/${RUN_NAME}.{out,err}"
echo ""

cd "$ORBIS_ROOT"

echo "Starting Orbis caching batch size calibration..."
echo ""

# Build command
CALIBRATE_CMD="python -m sae.calibrate_batch_size.orbis \
    --orbis-exp-dir \"$ORBIS_EXP_DIR\" \
    --layer \"$LAYER\" \
    --input-size $INPUT_H $INPUT_W \
    --num-frames $NUM_FRAMES \
    --model-name \"$MODEL_NAME\" \
    --baseline $BASELINE \
    --warmup-steps $WARMUP_STEPS \
    --safety-margin $SAFETY_MARGIN"

# Add force flag if needed
if [ "$FORCE" = "true" ]; then
    CALIBRATE_CMD="$CALIBRATE_CMD --force"
fi

eval $CALIBRATE_CMD

echo ""
echo "=== Calibration Complete ==="
echo "End: $(date)"
