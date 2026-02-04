#!/bin/bash
#SBATCH --job-name=pt2wds
#SBATCH --partition=lmbhiwidlc_gpu-rtx2080
#SBATCH --account=lmbhiwi-dlc
#SBATCH --output=/work/dlclarge2/lushtake-thesis/orbis/sae/slurm/logs/convert_to_webdataset/pt2wds_%j.log
#SBATCH --error=/work/dlclarge2/lushtake-thesis/orbis/sae/slurm/logs/convert_to_webdataset/pt2wds_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# =============================================================================
# Convert PyTorch .pt cache files to WebDataset sharded tar archives
# =============================================================================
# This job converts ~55k .pt files (~1.1TB) into WebDataset format with 10GB
# shards for optimized IO performance on the NFS cluster.
#
# Expected runtime: 3-6 hours depending on IO load
# Expected output: ~110 shards of ~10GB each
# =============================================================================

set -euo pipefail

echo "=========================================="
echo "Starting PT to WebDataset conversion"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

# Create output directory
OUTPUT_DIR="/data/lmbraid19/lushtake/sae_wds"
mkdir -p "${OUTPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"

# Activate conda environment
set +u
source ~/.bashrc
conda activate orbis_env
set -u

# Change to script directory
cd /work/dlclarge2/lushtake-thesis/orbis

# Run conversion script
echo "Starting conversion..."
python -u sae/scripts/convert_to_webdataset.py

echo "=========================================="
echo "Conversion complete!"
echo "Date: $(date)"
echo "=========================================="
