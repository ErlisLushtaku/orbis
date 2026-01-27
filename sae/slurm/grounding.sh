#!/bin/bash
#
# SAE Semantic Grounding Analysis
#
# Log files are named after the SAE barcode being analyzed:
# - Log files: sae/slurm/logs/{dataset}/{model}/layer_{N}/eval/{barcode}.{out,err}
#
# Submit with: sbatch sae/slurm/grounding.sh
#

#SBATCH --job-name=sae_grounding
#SBATCH --partition=lmbhiwi_gpu-rtx2080
#SBATCH --account=lmbhiwi-dlc
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
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

# SAE run to analyze (barcode from training)
SAE_BARCODE="topk_x16_k64_s42_20260121_111905"

# ------------------------------
# Setup log redirection
# ------------------------------
LOG_DIR="/work/dlclarge2/lushtake-thesis/orbis/sae/slurm/logs/${DATA_SOURCE}/${MODEL_NAME}/layer_${LAYER}/eval"
mkdir -p "$LOG_DIR"

# Redirect all output to the proper log files (named after SAE barcode)
exec > "${LOG_DIR}/${SAE_BARCODE}.out" 2> "${LOG_DIR}/${SAE_BARCODE}.err"

# Clean up initial SLURM logs
rm -f /work/dlclarge2/lushtake-thesis/orbis/sae/slurm/logs/slurm_init_${SLURM_JOB_ID}.out \
      /work/dlclarge2/lushtake-thesis/orbis/sae/slurm/logs/slurm_init_${SLURM_JOB_ID}.err 2>/dev/null || true

echo "=== SAE Semantic Grounding Analysis ==="
echo "Job ID: $SLURM_JOB_ID"
echo "SAE Barcode: $SAE_BARCODE"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

# Setup environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate orbis_env

export TK_WORK_DIR=/work/dlclarge2/lushtake-thesis/orbis/logs_tk

cd /work/dlclarge2/lushtake-thesis/orbis

# ------------------------------
# Paths
# ------------------------------
ORBIS_DIR="/work/dlclarge2/lushtake-thesis/orbis"
EXP_DIR="${ORBIS_DIR}/logs_wm/${MODEL_NAME}"

# SAE checkpoint path
SAE_RUN_DIR="${ORBIS_DIR}/logs_sae/runs/${DATA_SOURCE}/${MODEL_NAME}/layer_${LAYER}/${SAE_BARCODE}"
SAE_CHECKPOINT="${SAE_RUN_DIR}/best_sae.pt"

# Test data
TEST_VIDEOS="/work/dlclarge2/lushtake-thesis/data/covla/videos"
TEST_CAPTIONS="/work/dlclarge2/lushtake-thesis/data/covla/captions"
SPLIT_FILE="/work/dlclarge2/lushtake-thesis/data/covla/splits/test_split.jsonl"

# Output directory (analysis folder within the run)
OUTPUT_DIR="${SAE_RUN_DIR}/analysis"

echo "Configuration:"
echo "  data_source=$DATA_SOURCE"
echo "  model=$MODEL_NAME"
echo "  layer=$LAYER"
echo "  sae_barcode=$SAE_BARCODE"
echo ""
echo "Paths:"
echo "  SAE checkpoint: $SAE_CHECKPOINT"
echo "  Output directory: $OUTPUT_DIR"
echo "  Log files: ${LOG_DIR}/${SAE_BARCODE}.{out,err}"
echo ""

# Run semantic grounding analysis
python sae/scripts/semantic_grounding.py \
    --exp_dir "$EXP_DIR" \
    --sae_checkpoint "$SAE_CHECKPOINT" \
    --test_videos_dir "$TEST_VIDEOS" \
    --test_captions_dir "$TEST_CAPTIONS" \
    --split_file "$SPLIT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 4 \
    --num_workers 4 \
    --device cuda

echo ""
echo "=== Semantic Grounding Analysis Complete ==="
echo "End: $(date)"
