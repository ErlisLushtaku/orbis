#!/bin/bash
#
# SLURM Job Script: Convert PT cache to WebDataset sharded tar archives
#
# Converts batch_*.pt files from the activation cache into WebDataset format
# for optimized IO performance on the NFS cluster.
#
# Usage:
#   sbatch sae/slurm/convert_to_webdataset.sh --data_source covla [OPTIONS]
#   sbatch sae/slurm/convert_to_webdataset.sh --data_source nuplan --layer 22 [OPTIONS]
#
# Required:
#   --data_source STR    Data source: "nuplan" or "covla"
#
# Options (override defaults):
#   --layer N            Layer number (default: 22 for nuplan, 12 for covla)
#   --splits STR         Comma-separated splits to convert (default: "train,val")
#   --shard_max_size N   Max shard size in bytes (default: 10e9 = 10 GB)
#
# Examples:
#   sbatch sae/slurm/convert_to_webdataset.sh --data_source covla
#   sbatch sae/slurm/convert_to_webdataset.sh --data_source covla --layer 12 --splits train,val
#   sbatch sae/slurm/convert_to_webdataset.sh --data_source nuplan --layer 22 --splits val
#
# Output shards follow the naming convention expected by train_sae.py:
#   {dst_dir}/{split}/layer_{layer}-{split}-%06d.tar
#
# Logs: sae/slurm/logs/convert_to_webdataset/{data_source}/layer_{N}/pt2wds_{JOBID}.{out,err}
#

#SBATCH --job-name=pt2wds
#SBATCH --partition=lmbhiwidlc_gpu-rtx2080
#SBATCH --account=lmbhiwi-dlc
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/work/dlclarge2/lushtake-thesis/orbis/sae/slurm/logs/slurm_init_%j.out
#SBATCH --error=/work/dlclarge2/lushtake-thesis/orbis/sae/slurm/logs/slurm_init_%j.err

set -euo pipefail

# ------------------------------
# Default Configuration
# ------------------------------
MODEL_NAME="orbis_288x512"
DATA_SOURCE=""
LAYER=""
SPLITS="train,val"
SHARD_MAX_SIZE="10e9"

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
        --splits)
            SPLITS="$2"
            shift 2
            ;;
        --shard_max_size)
            SHARD_MAX_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: sbatch convert_to_webdataset.sh --data_source {nuplan,covla} [OPTIONS]"
            exit 1
            ;;
    esac
done

# ------------------------------
# Validate data_source
# ------------------------------
if [ -z "$DATA_SOURCE" ]; then
    echo "ERROR: --data_source is required"
    echo "Usage: sbatch convert_to_webdataset.sh --data_source {nuplan,covla} [OPTIONS]"
    exit 1
fi

if [ "$DATA_SOURCE" != "nuplan" ] && [ "$DATA_SOURCE" != "covla" ]; then
    echo "ERROR: Invalid data_source '$DATA_SOURCE'. Must be 'nuplan' or 'covla'"
    exit 1
fi

# ------------------------------
# Set data-source-specific defaults
# ------------------------------
ORBIS_ROOT="/work/dlclarge2/lushtake-thesis/orbis"

if [ "$DATA_SOURCE" = "nuplan" ]; then
    [ -z "$LAYER" ] && LAYER=22
    SRC_DIR="${ORBIS_ROOT}/logs_sae/sae_cache/nuplan/${MODEL_NAME}/layer_${LAYER}"
    DST_DIR="/data/lmbraid19/lushtake/sae_wds"
elif [ "$DATA_SOURCE" = "covla" ]; then
    [ -z "$LAYER" ] && LAYER=12
    SRC_DIR="${ORBIS_ROOT}/logs_sae/sae_cache/covla/${MODEL_NAME}/layer_${LAYER}"
    DST_DIR="/data/lmbraid19/lushtake/sae_wds_covla"
fi

# ------------------------------
# Setup log directories
# ------------------------------
LOG_DIR="${ORBIS_ROOT}/sae/slurm/logs/convert_to_webdataset/${DATA_SOURCE}/layer_${LAYER}"
mkdir -p "$LOG_DIR"

exec > "${LOG_DIR}/pt2wds_${SLURM_JOB_ID}.out" 2> "${LOG_DIR}/pt2wds_${SLURM_JOB_ID}.err"

rm -f "${ORBIS_ROOT}/sae/slurm/logs/slurm_init_${SLURM_JOB_ID}.out" \
      "${ORBIS_ROOT}/sae/slurm/logs/slurm_init_${SLURM_JOB_ID}.err" 2>/dev/null || true

# ------------------------------
# Environment Setup
# ------------------------------
echo "=========================================="
echo "PT to WebDataset Conversion"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""
echo "Configuration:"
echo "  data_source=${DATA_SOURCE}"
echo "  model=${MODEL_NAME}"
echo "  layer=${LAYER}"
echo "  splits=${SPLITS}"
echo "  shard_max_size=${SHARD_MAX_SIZE}"
echo "  src_dir=${SRC_DIR}"
echo "  dst_dir=${DST_DIR}"
echo "=========================================="

set +u
source ~/.bashrc
conda activate orbis_env
set -u

echo "Python: $(which python)"
echo "Conda env: ${CONDA_DEFAULT_ENV:-unknown}"
echo ""

cd "$ORBIS_ROOT"

echo "Starting conversion..."
python -u sae/scripts/convert_to_webdataset.py \
    --src_dir "$SRC_DIR" \
    --dst_dir "$DST_DIR" \
    --layer "$LAYER" \
    --splits "$SPLITS" \
    --shard_max_size "$SHARD_MAX_SIZE"

echo "=========================================="
echo "Conversion complete!"
echo "Date: $(date)"
echo "=========================================="
