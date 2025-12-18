#!/bin/bash
#SBATCH --account=def-kmoran
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=yolo_seg
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=spencer.bialek@gmail.com

# ============================================================================
# YOLOv11 Segmentation Training on DRAC Cluster
# ============================================================================
# 
# Usage:
#   sbatch slurm/train.sh                              # Use defaults
#   sbatch slurm/train.sh --mode binary                # Binary segmentation
#   sbatch slurm/train.sh --mode top_n --top_n 100     # Top 100 categories
#   sbatch slurm/train.sh --repo /path/to/repo --data /path/to/data.tar.gz
#
# Key arguments:
#   --repo DIR          Path to yolo_segmentation repo (default: ~/yolo_segmentation)
#   --data FILE         Path to data tarball (default: REPO/data/mbari_raw.tar.gz)
#   --mode MODE         Conversion mode: binary, top_n, all (default: top_n)
#   --top_n N           Number of top categories for top_n mode (default: 100)
#   --val_ratio R       Validation split ratio (default: 0.2)
#
# Data format:
#   The script expects raw data (COCO JSON + images) and will convert
#   to YOLO format at job start in SLURM_TMPDIR for fast I/O.
#
# Before first run:
#   1. Run: bash slurm/setup_env.sh          # Create virtual environment
#   2. Create data tarball with raw COCO data:
#      bash slurm/prepare_data.sh
# ============================================================================

set -e  # Exit on error

# Default paths (adjust for your DRAC setup)
DEFAULT_REPO="${HOME}/yolo_segmentation"
REPO_DIR=""
DATA_TARBALL=""

# Default conversion options
CONVERT_MODE="top_n"
CONVERT_TOP_N="100"
VAL_RATIO="0.2"

# Parse command line arguments
TRAIN_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --repo)
            REPO_DIR="$2"
            shift 2
            ;;
        --data)
            DATA_TARBALL="$2"
            shift 2
            ;;
        --mode)
            CONVERT_MODE="$2"
            shift 2
            ;;
        --top_n)
            CONVERT_TOP_N="$2"
            shift 2
            ;;
        --val_ratio)
            VAL_RATIO="$2"
            shift 2
            ;;
        *)
            TRAIN_ARGS+=("$1")
            shift
            ;;
    esac
done

# Set defaults if not provided
if [ -z "${REPO_DIR}" ]; then
    # Try to find repo: first check if we're in it, then use default
    if [ -f "scripts/train.py" ]; then
        REPO_DIR="$(pwd)"
    elif [ -d "${DEFAULT_REPO}" ]; then
        REPO_DIR="${DEFAULT_REPO}"
    else
        echo "ERROR: Could not find repo. Use --repo /path/to/yolo_segmentation"
        exit 1
    fi
fi

if [ -z "${DATA_TARBALL}" ]; then
    DATA_TARBALL="${REPO_DIR}/data/mbari_raw.tar.gz"
fi

VENV_PATH="${REPO_DIR}/.venv"

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_PER_NODE"
echo "Started: $(date)"
echo "============================================"
echo "Repo: ${REPO_DIR}"
echo "Data: ${DATA_TARBALL}"
echo "Mode: ${CONVERT_MODE}"
echo "============================================"

# Create logs directory
mkdir -p "${REPO_DIR}/logs"

# ============================================================================
# 1. Setup Python Environment
# ============================================================================
echo ""
echo "[1/5] Setting up Python environment..."

# Load modules - use saved modules file if available, otherwise use defaults
if [ -f "${REPO_DIR}/slurm/.modules" ]; then
    echo "Loading modules from ${REPO_DIR}/slurm/.modules..."
    source "${REPO_DIR}/slurm/.modules"
else
    echo "Loading default modules..."
    module load StdEnv/2023
    module load python/3.11 cuda cudnn
    module load opencv/4.8.1
    module load scipy-stack
fi

echo ""
echo "Loaded modules:"
module list

# Activate virtual environment
if [ -d "${VENV_PATH}" ]; then
    echo ""
    echo "Activating virtual environment..."
    source "${VENV_PATH}/bin/activate"
else
    echo "ERROR: Virtual environment not found at ${VENV_PATH}"
    echo "Run 'bash slurm/setup_env.sh' first to create it."
    exit 1
fi

# Verify key packages
python -c "from ultralytics import YOLO; from pycocotools import mask; import cv2; print('Dependencies OK')"

# ============================================================================
# 2. Extract Raw Data to SLURM_TMPDIR
# ============================================================================
echo ""
echo "[2/5] Extracting raw data to local SSD ($SLURM_TMPDIR)..."

LOCAL_DATA_DIR="${SLURM_TMPDIR}/data"
mkdir -p "${LOCAL_DATA_DIR}"

if [ -f "${DATA_TARBALL}" ]; then
    echo "Extracting ${DATA_TARBALL}..."
    tar -xzf "${DATA_TARBALL}" -C "${LOCAL_DATA_DIR}"
    echo "Extraction complete."
    echo "Contents:"
    ls -la "${LOCAL_DATA_DIR}"
    
    # Find COCO JSON (look for *.json files)
    COCO_JSON=$(find "${LOCAL_DATA_DIR}" -name "*.json" -type f | head -1)
    
    # Find images directory (look for directory containing images)
    IMAGE_DIR=$(find "${LOCAL_DATA_DIR}" -type d -name "images" | head -1)
    if [ -z "${IMAGE_DIR}" ]; then
        # If no 'images' dir, look for 'train' dir
        IMAGE_DIR=$(find "${LOCAL_DATA_DIR}" -type d -name "train" | head -1)
    fi
    if [ -z "${IMAGE_DIR}" ]; then
        # Last resort: find a directory with image files
        IMAGE_DIR=$(find "${LOCAL_DATA_DIR}" -type f \( -name "*.jpg" -o -name "*.png" \) | head -1 | xargs dirname)
    fi
    
    if [ -z "${COCO_JSON}" ] || [ -z "${IMAGE_DIR}" ]; then
        echo "ERROR: Could not find JSON or images in extracted data"
        find "${LOCAL_DATA_DIR}" -type f | head -20
        exit 1
    fi
    
    echo "COCO JSON: ${COCO_JSON}"
    echo "Images: ${IMAGE_DIR}"
else
    echo "ERROR: Data tarball not found at ${DATA_TARBALL}"
    echo "Run 'bash slurm/prepare_data.sh' first."
    exit 1
fi

# ============================================================================
# 3. Convert COCO RLE to YOLO Format
# ============================================================================
echo ""
echo "[3/5] Converting COCO RLE to YOLO polygon format..."
echo "      Mode: ${CONVERT_MODE}, Top N: ${CONVERT_TOP_N}, Val ratio: ${VAL_RATIO}"

YOLO_DATASET="${SLURM_TMPDIR}/yolo_dataset"

cd "${REPO_DIR}"
python scripts/convert_coco_to_yolo.py \
    --coco_json "${COCO_JSON}" \
    --output_dir "${YOLO_DATASET}" \
    --image_dir "${IMAGE_DIR}" \
    --val_ratio "${VAL_RATIO}" \
    --mode "${CONVERT_MODE}" \
    --top_n "${CONVERT_TOP_N}" \
    --min_annotations 0 \
    --workers 8

# Verify conversion
if [ ! -f "${YOLO_DATASET}/dataset.yaml" ]; then
    echo "ERROR: Conversion failed - dataset.yaml not found"
    exit 1
fi

echo "Conversion complete!"
ls -la "${YOLO_DATASET}"

# ============================================================================
# 4. Run Training
# ============================================================================
echo ""
echo "[4/5] Starting training..."
echo ""

DATASET_CONFIG="${YOLO_DATASET}/dataset.yaml"

python scripts/train.py \
    --data "${DATASET_CONFIG}" \
    --model yolo11m-seg.pt \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 \
    --workers 8 \
    --project "${REPO_DIR}/runs/segment" \
    --name "drac_${SLURM_JOB_ID}" \
    "${TRAIN_ARGS[@]}"

# ============================================================================
# 5. Cleanup and Summary
# ============================================================================
echo ""
echo "[5/5] Training complete!"
echo ""

RESULTS_DIR="${REPO_DIR}/runs/segment/drac_${SLURM_JOB_ID}"
if [ -d "${RESULTS_DIR}" ]; then
    echo "============================================"
    echo "Results saved to: ${RESULTS_DIR}"
    echo "Best weights: ${RESULTS_DIR}/weights/best.pt"
    echo "Finished: $(date)"
    echo "============================================"
fi
