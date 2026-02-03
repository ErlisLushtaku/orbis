#!/bin/bash
#
# SLURM Job Script: SAE Semantic Grounding Analysis
#
# This is a unified grounding script that works with both NuPlan and CoVLA datasets.
#
# Usage:
#   sbatch sae/slurm/grounding.sh --data_source covla --sae_barcode topk_x16_k64_s42_20260121_111905
#   sbatch sae/slurm/grounding.sh --data_source nuplan --sae_barcode topk_x16_k64_s42_20260121_111905
#
# Required:
#   --data_source STR   Data source: "nuplan" or "covla"
#   --sae_barcode STR   SAE run barcode to analyze (e.g., topk_x16_k64_s42_20260121_111905)
#
# Options:
#   --layer N           Layer (default: 22 for nuplan, 12 for covla)
#   --batch_size N      Batch size (default: 4)
#   --num_test_videos N Number of test videos (default: 100)
#   --skip_videos N     Videos to skip (default: 400 for nuplan, 0 for covla)
#   --top_k N           Top-K features to analyze (default: 30)
#
# Logs: sae/slurm/logs/{dataset}/{model}/layer_{N}/eval/{barcode}.{out,err}
#

#SBATCH --job-name=sae_grounding
#SBATCH --time=06:00:00
#SBATCH --partition=lmbhiwidlc_gpu-rtx2080
#SBATCH --account=lmbhiwi-dlc
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/work/dlclarge2/lushtake-thesis/orbis/sae/slurm/logs/slurm_init_%j.out
#SBATCH --error=/work/dlclarge2/lushtake-thesis/orbis/sae/slurm/logs/slurm_init_%j.err

set -eo pipefail

# ------------------------------
# Default Configuration
# ------------------------------
MODEL_NAME="orbis_288x512"
DATA_SOURCE=""
SAE_BARCODE=""

# Default parameters (will be overridden based on data_source)
LAYER=""
BATCH_SIZE=4
NUM_TEST_VIDEOS=100
SKIP_VIDEOS=""
TOP_K=30
NUM_FRAMES=6

# ------------------------------
# Parse Command Line Arguments
# ------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_source)
            DATA_SOURCE="$2"
            shift 2
            ;;
        --sae_barcode)
            SAE_BARCODE="$2"
            shift 2
            ;;
        --layer)
            LAYER="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_test_videos)
            NUM_TEST_VIDEOS="$2"
            shift 2
            ;;
        --skip_videos)
            SKIP_VIDEOS="$2"
            shift 2
            ;;
        --top_k)
            TOP_K="$2"
            shift 2
            ;;
        --num_frames)
            NUM_FRAMES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: sbatch grounding.sh --data_source {nuplan,covla} --sae_barcode BARCODE [OPTIONS]"
            exit 1
            ;;
    esac
done

# ------------------------------
# Validate required arguments
# ------------------------------
if [ -z "$DATA_SOURCE" ]; then
    echo "ERROR: --data_source is required"
    echo "Usage: sbatch grounding.sh --data_source {nuplan,covla} --sae_barcode BARCODE"
    exit 1
fi

if [ -z "$SAE_BARCODE" ]; then
    echo "ERROR: --sae_barcode is required"
    echo "Usage: sbatch grounding.sh --data_source {nuplan,covla} --sae_barcode BARCODE"
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
    [ -z "$LAYER" ] && LAYER=22
    [ -z "$SKIP_VIDEOS" ] && SKIP_VIDEOS=400
    GROUNDING_SCRIPT="sae/scripts/semantic_grounding_nuplan.py"
elif [ "$DATA_SOURCE" = "covla" ]; then
    DATA_DIR="/work/dlclarge2/lushtake-thesis/data/covla/videos"
    CAPTIONS_DIR="/work/dlclarge2/lushtake-thesis/data/covla/captions"
    SPLIT_FILE="/work/dlclarge2/lushtake-thesis/data/covla/splits/test_split.jsonl"
    [ -z "$LAYER" ] && LAYER=12
    [ -z "$SKIP_VIDEOS" ] && SKIP_VIDEOS=0
    GROUNDING_SCRIPT="sae/scripts/semantic_grounding.py"
fi

# ------------------------------
# Derived paths
# ------------------------------
SAE_RUN_DIR="${ORBIS_ROOT}/logs_sae/runs/${DATA_SOURCE}/${MODEL_NAME}/layer_${LAYER}/${SAE_BARCODE}"
SAE_CHECKPOINT="${SAE_RUN_DIR}/best_sae.pt"
OUTPUT_DIR="${SAE_RUN_DIR}/analysis"

# ------------------------------
# Setup log directories
# ------------------------------
LOG_DIR="${ORBIS_ROOT}/sae/slurm/logs/${DATA_SOURCE}/${MODEL_NAME}/layer_${LAYER}/eval"
mkdir -p "$LOG_DIR"

# Redirect stdout and stderr to log files
exec > "${LOG_DIR}/${SAE_BARCODE}.out" 2> "${LOG_DIR}/${SAE_BARCODE}.err"

# Clean up initial SLURM logs
rm -f /work/dlclarge2/lushtake-thesis/orbis/sae/slurm/logs/slurm_init_${SLURM_JOB_ID}.out \
      /work/dlclarge2/lushtake-thesis/orbis/sae/slurm/logs/slurm_init_${SLURM_JOB_ID}.err 2>/dev/null || true

# ------------------------------
# Environment Setup
# ------------------------------
echo "=== SAE Semantic Grounding Analysis for ${DATA_SOURCE^^} ==="
echo "Job ID: $SLURM_JOB_ID"
echo "SAE Barcode: $SAE_BARCODE"
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

echo "Configuration:"
echo "  data_source=$DATA_SOURCE"
echo "  model=$MODEL_NAME"
echo "  layer=$LAYER"
echo "  sae_barcode=$SAE_BARCODE"
echo "  batch_size=$BATCH_SIZE"
echo "  num_test_videos=$NUM_TEST_VIDEOS"
echo "  skip_videos=$SKIP_VIDEOS"
echo "  top_k=$TOP_K"
echo ""

echo "Paths:"
echo "  SAE checkpoint: $SAE_CHECKPOINT"
echo "  Output directory: $OUTPUT_DIR"
echo "  Log files: ${LOG_DIR}/${SAE_BARCODE}.{out,err}"
echo ""

cd "$ORBIS_ROOT"

# Verify checkpoint exists
if [ ! -f "$SAE_CHECKPOINT" ]; then
    echo "ERROR: SAE checkpoint not found: $SAE_CHECKPOINT"
    echo "Please verify the sae_barcode and layer are correct."
    exit 1
fi

echo "Starting semantic grounding analysis..."
echo ""

# ------------------------------
# Run analysis based on data source
# ------------------------------
if [ "$DATA_SOURCE" = "nuplan" ]; then
    python "$GROUNDING_SCRIPT" \
        --exp_dir "$EXP_DIR" \
        --sae_checkpoint "$SAE_CHECKPOINT" \
        --nuplan_data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --layer "$LAYER" \
        --batch_size "$BATCH_SIZE" \
        --num_frames "$NUM_FRAMES" \
        --num_test_videos "$NUM_TEST_VIDEOS" \
        --skip_videos "$SKIP_VIDEOS" \
        --top_k "$TOP_K"
elif [ "$DATA_SOURCE" = "covla" ]; then
    python "$GROUNDING_SCRIPT" \
        --exp_dir "$EXP_DIR" \
        --sae_checkpoint "$SAE_CHECKPOINT" \
        --test_videos_dir "$DATA_DIR" \
        --test_captions_dir "$CAPTIONS_DIR" \
        --split_file "$SPLIT_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size "$BATCH_SIZE" \
        --num_workers 4 \
        --device cuda
fi

echo ""
echo "=== Semantic Grounding Analysis Complete ==="
echo "End: $(date)"
echo "Results: $OUTPUT_DIR"
