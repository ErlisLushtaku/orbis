#!/bin/bash
#
# SLURM Job Script: SAE Semantic Grounding Analysis
#
# This is a unified grounding script that works with both NuPlan and CoVLA datasets.
# The --data_source argument follows the same convention as training: use the full
# data source name (e.g., "nuplan-webdataset") for correct path resolution.
# The base data source (e.g., "nuplan") is derived automatically for data loading.
#
# Usage:
#   sbatch sae/slurm/grounding.sh --data_source nuplan-webdataset --sae_barcode topk_x16_k64_s42_20260209_120057
#   sbatch sae/slurm/grounding.sh --data_source covla --sae_barcode topk_x16_k64_s42_20260121_111905
#
# Required:
#   --data_source STR   Data source (same as training): "nuplan", "nuplan-webdataset", "covla", "covla-webdataset"
#   --sae_barcode STR   SAE run barcode to analyze (e.g., topk_x16_k64_s42_20260209_120057)
#
# Options:
#   --layer N           Layer (default: 22 for nuplan, 12 for covla)
#   --batch_size N      Batch size (default: 4)
#   --num_videos N      Total videos matching training config (default: 988 for nuplan)
#   --val_split F       Validation split fraction matching training (default: 0.1)
#   --top_k N           Top-K features to analyze (default: 30)
#
# Logs: sae/slurm/logs/runs/{data_source}/{model}/layer_{N}/eval/{barcode}.{out,err}
#

#SBATCH --job-name=sae_grounding
#SBATCH --time=06:00:00
#SBATCH --partition=tflmb_gpu-rtx4090
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
NUM_VIDEOS=""
VAL_SPLIT=0.1
TOP_K=30
NUM_FRAMES=6
USE_VAL_SPLIT=0

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
        --num_videos)
            NUM_VIDEOS="$2"
            shift 2
            ;;
        --val_split)
            VAL_SPLIT="$2"
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
        --use_val_split)
            USE_VAL_SPLIT=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: sbatch grounding.sh --data_source {nuplan,nuplan-webdataset,covla,covla-webdataset} --sae_barcode BARCODE [OPTIONS]"
            exit 1
            ;;
    esac
done

# ------------------------------
# Validate required arguments
# ------------------------------
if [ -z "$DATA_SOURCE" ]; then
    echo "ERROR: --data_source is required"
    echo "Usage: sbatch grounding.sh --data_source {nuplan,nuplan-webdataset,covla,covla-webdataset} --sae_barcode BARCODE"
    exit 1
fi

if [ -z "$SAE_BARCODE" ]; then
    echo "ERROR: --sae_barcode is required"
    echo "Usage: sbatch grounding.sh --data_source {nuplan,nuplan-webdataset,covla,covla-webdataset} --sae_barcode BARCODE"
    exit 1
fi

# Derive base data source by stripping -webdataset suffix (same as get_base_data_source() in Python)
BASE_DATA_SOURCE="${DATA_SOURCE%-webdataset}"

if [ "$BASE_DATA_SOURCE" != "nuplan" ] && [ "$BASE_DATA_SOURCE" != "covla" ]; then
    echo "ERROR: Invalid data_source '$DATA_SOURCE'. Base must be 'nuplan' or 'covla'"
    exit 1
fi

# ------------------------------
# Set data source specific defaults
# ------------------------------
ORBIS_ROOT="/work/dlclarge2/lushtake-thesis/orbis"
EXP_DIR="${ORBIS_ROOT}/logs_wm/${MODEL_NAME}"

if [ "$BASE_DATA_SOURCE" = "nuplan" ]; then
    DATA_DIR="/work/dlcsmall2/galessos-nuPlan/nuPlan_640x360_10Hz"
    [ -z "$LAYER" ] && LAYER=22
    [ -z "$NUM_VIDEOS" ] && NUM_VIDEOS=988
    GROUNDING_SCRIPT="sae/scripts/semantic_grounding_nuplan.py"
elif [ "$BASE_DATA_SOURCE" = "covla" ]; then
    DATA_DIR="/work/dlclarge2/lushtake-thesis/data/covla/videos"
    CAPTIONS_DIR="/work/dlclarge2/lushtake-thesis/data/covla/captions"
    SPLIT_FILE="/work/dlclarge2/lushtake-thesis/data/covla/splits/test_split.jsonl"
    [ -z "$LAYER" ] && LAYER=12
    [ -z "$NUM_VIDEOS" ] && NUM_VIDEOS=3000
    GROUNDING_SCRIPT="sae/scripts/semantic_grounding_covla.py"
fi

# ------------------------------
# Derived paths (DATA_SOURCE used as-is for directory names)
# ------------------------------
SAE_RUN_DIR="${ORBIS_ROOT}/logs_sae/runs/${DATA_SOURCE}/${MODEL_NAME}/layer_${LAYER}/${SAE_BARCODE}"
SAE_CHECKPOINT="${SAE_RUN_DIR}/best_sae.pt"
OUTPUT_DIR="${SAE_RUN_DIR}/analysis"

# ------------------------------
# Setup log directories
# ------------------------------
LOG_DIR="${ORBIS_ROOT}/sae/slurm/logs/runs/${DATA_SOURCE}/${MODEL_NAME}/layer_${LAYER}/eval"
mkdir -p "$LOG_DIR"

# Redirect stdout and stderr to log files
exec > "${LOG_DIR}/${SAE_BARCODE}.out" 2> "${LOG_DIR}/${SAE_BARCODE}.err"

# Clean up initial SLURM logs
rm -f /work/dlclarge2/lushtake-thesis/orbis/sae/slurm/logs/slurm_init_${SLURM_JOB_ID}.out \
      /work/dlclarge2/lushtake-thesis/orbis/sae/slurm/logs/slurm_init_${SLURM_JOB_ID}.err 2>/dev/null || true

# ------------------------------
# Environment Setup
# ------------------------------
echo "=== SAE Semantic Grounding Analysis for ${DATA_SOURCE} ==="
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
export TQDM_MININTERVAL=60

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
echo "  base_data_source=$BASE_DATA_SOURCE"
echo "  model=$MODEL_NAME"
echo "  layer=$LAYER"
echo "  sae_barcode=$SAE_BARCODE"
echo "  batch_size=$BATCH_SIZE"
echo "  num_videos=$NUM_VIDEOS"
echo "  val_split=$VAL_SPLIT"
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
# Val cache (pre-computed Orbis activations from training)
# Applies when the grounding uses the same val split as training:
# - NuPlan: always uses val split
# - CoVLA: only when --use_val_split is passed
# ------------------------------
VAL_CACHE_ARGS=""
USE_VAL_CACHE=0
if [ "$BASE_DATA_SOURCE" = "nuplan" ]; then
    USE_VAL_CACHE=1
elif [ "$BASE_DATA_SOURCE" = "covla" ] && [ "$USE_VAL_SPLIT" = "1" ]; then
    USE_VAL_CACHE=1
fi

if [ "$USE_VAL_CACHE" = "1" ]; then
    VAL_CACHE_DIR="${ORBIS_ROOT}/logs_sae/sae_cache/${DATA_SOURCE}/${MODEL_NAME}/layer_${LAYER}/val"
    if [ -d "$VAL_CACHE_DIR" ] && [ -f "${VAL_CACHE_DIR}/_meta.json" ]; then
        echo "Using pre-cached val activations from: $VAL_CACHE_DIR"
        VAL_CACHE_ARGS="--val_cache_dir $VAL_CACHE_DIR"
    else
        echo "No val cache found at $VAL_CACHE_DIR, will run Orbis forward passes"
    fi
fi
echo ""

# ------------------------------
# Run analysis based on base data source
# ------------------------------
if [ "$BASE_DATA_SOURCE" = "nuplan" ]; then
    python "$GROUNDING_SCRIPT" \
        --exp_dir "$EXP_DIR" \
        --sae_checkpoint "$SAE_CHECKPOINT" \
        --nuplan_data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --layer "$LAYER" \
        --batch_size "$BATCH_SIZE" \
        --num_frames "$NUM_FRAMES" \
        --num_videos "$NUM_VIDEOS" \
        --val_split "$VAL_SPLIT" \
        --top_k "$TOP_K" \
        $VAL_CACHE_ARGS
elif [ "$BASE_DATA_SOURCE" = "covla" ]; then
    COVLA_EXTRA_ARGS=""
    if [ "$USE_VAL_SPLIT" = "1" ]; then
        COVLA_EXTRA_ARGS="--use_val_split --num_videos $NUM_VIDEOS --val_split $VAL_SPLIT"
    elif [ -n "$SPLIT_FILE" ]; then
        COVLA_EXTRA_ARGS="--split_file $SPLIT_FILE"
    fi
    python "$GROUNDING_SCRIPT" \
        --exp_dir "$EXP_DIR" \
        --sae_checkpoint "$SAE_CHECKPOINT" \
        --test_videos_dir "$DATA_DIR" \
        --test_captions_dir "$CAPTIONS_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size "$BATCH_SIZE" \
        --num_workers 4 \
        --device cuda \
        $COVLA_EXTRA_ARGS \
        $VAL_CACHE_ARGS
fi

echo ""
echo "=== Semantic Grounding Analysis Complete ==="
echo "End: $(date)"
echo "Results: $OUTPUT_DIR"
