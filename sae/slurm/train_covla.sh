#!/bin/bash
#
# SAE Training on CoVLA Dataset
#
# Generates a unique barcode at job start and uses it for both:
# - Run directory: logs_sae/runs/{dataset}/{model}/layer_{N}/{barcode}/
# - Log files: sae/slurm/logs/{dataset}/{model}/layer_{N}/train/{barcode}.{out,err}
#
# Submit with: sbatch sae/slurm/train_covla.sh
#

#SBATCH --job-name=sae_covla
#SBATCH --time=24:00:00
#SBATCH --partition=lmbhiwidlc_gpu-rtx2080
#SBATCH --account=lmbhiwi-dlc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=450G
# Initial output goes to a temp location, then redirected at runtime
#SBATCH --output=/work/dlclarge2/lushtake-thesis/orbis/sae/slurm/logs/slurm_init_%j.out
#SBATCH --error=/work/dlclarge2/lushtake-thesis/orbis/sae/slurm/logs/slurm_init_%j.err

set -eo pipefail

# ------------------------------
# Experiment Configuration
# ------------------------------
DATA_SOURCE="covla"
MODEL_NAME="orbis_288x512"
LAYER=12

# Training parameters
BATCH_SIZE=4          # Reduced for 10GB GPU
NUM_EPOCHS=50
K=64                  # Top-k sparsity
EXPANSION_FACTOR=16   # SAE expansion
SEED=42
NUM_VIDEOS=3000       # Number of CoVLA videos to use (None = all available)

# Cache behavior
# Set to "true" to force rebuild cache (needed when changing NUM_VIDEOS)
# Set to "false" to reuse existing cache (faster if dataset size unchanged)
REBUILD_CACHE="false"

# ------------------------------
# Generate unique barcode with timestamp
# ------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BARCODE="topk_x${EXPANSION_FACTOR}_k${K}_s${SEED}_${TIMESTAMP}"

# Log directory following runs/ hierarchy
LOG_DIR="/work/dlclarge2/lushtake-thesis/orbis/sae/slurm/logs/${DATA_SOURCE}/${MODEL_NAME}/layer_${LAYER}/train"
mkdir -p "$LOG_DIR"

# Redirect all output to the proper log files
exec > "${LOG_DIR}/${BARCODE}.out" 2> "${LOG_DIR}/${BARCODE}.err"

# Clean up initial SLURM logs (they'll be empty or minimal)
rm -f /work/dlclarge2/lushtake-thesis/orbis/sae/slurm/logs/slurm_init_${SLURM_JOB_ID}.out \
      /work/dlclarge2/lushtake-thesis/orbis/sae/slurm/logs/slurm_init_${SLURM_JOB_ID}.err 2>/dev/null || true

echo "=== SAE Training on CoVLA Dataset ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Barcode: $BARCODE"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"
echo ""

# ------------------------------
# Environment setup
# ------------------------------
export MKL_INTERFACE_LAYER="GNU"
export MKL_THREADING_LAYER="GNU"

# Tokenizer path for Orbis
export TK_WORK_DIR=/work/dlclarge2/lushtake-thesis/orbis/logs_tk

# Activate conda
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
ORBIS_DIR="/work/dlclarge2/lushtake-thesis/orbis"
EXP_DIR="${ORBIS_DIR}/logs_wm/${MODEL_NAME}"
COVLA_VIDEOS="/work/dlclarge2/lushtake-thesis/data/covla/videos"
COVLA_CAPTIONS="/work/dlclarge2/lushtake-thesis/data/covla/captions"

cd "$ORBIS_DIR"

echo "Orbis dir: $ORBIS_DIR"
echo "Exp dir: $EXP_DIR"
echo "CoVLA videos: $COVLA_VIDEOS"
echo "CoVLA captions: $COVLA_CAPTIONS"
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

# ------------------------------
# Run training
# ------------------------------
echo "Starting SAE training..."
echo ""

# Build command with optional rebuild_cache flag
TRAIN_CMD="python sae/scripts/train_sae.py \
    --exp_dir \"$EXP_DIR\" \
    --data_source \"$DATA_SOURCE\" \
    --covla_videos_dir \"$COVLA_VIDEOS\" \
    --covla_captions_dir \"$COVLA_CAPTIONS\" \
    --num_videos \"$NUM_VIDEOS\" \
    --stored_frame_rate 20 \
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
