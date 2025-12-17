#!/bin/bash
# ============================================================================
# Setup Python Environment for DRAC Cluster
# ============================================================================
#
# Run this script ONCE before submitting training jobs.
# This creates a persistent virtual environment in the project directory.
#
# Usage:
#   bash slurm/setup_env.sh
#
# ============================================================================

set -e

echo "============================================"
echo "Setting up Python Environment for DRAC"
echo "============================================"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${PROJECT_DIR}/.venv"

echo "Project directory: ${PROJECT_DIR}"
echo "Virtual env path: ${VENV_PATH}"

# ============================================================================
# 1. Load Required Modules
# ============================================================================
echo ""
echo "[1/4] Loading modules..."

module load python/3.11 cuda/12.2 cudnn/9.2

python --version
which python

# ============================================================================
# 2. Create Virtual Environment
# ============================================================================
echo ""
echo "[2/4] Creating virtual environment..."

if [ -d "${VENV_PATH}" ]; then
    echo "Virtual environment already exists. Removing old one..."
    rm -rf "${VENV_PATH}"
fi

python -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

echo "Virtual environment created and activated."
which python
python --version

# ============================================================================
# 3. Install Dependencies
# ============================================================================
echo ""
echo "[3/4] Installing dependencies..."

# Upgrade pip
pip install --upgrade pip

# Install from requirements (with --no-index to use wheelhouse)
# If packages aren't in wheelhouse, they'll be downloaded
pip install --no-index ultralytics || pip install ultralytics
pip install --no-index pycocotools || pip install pycocotools
pip install --no-index opencv-python-headless || pip install opencv-python-headless
pip install --no-index tqdm || pip install tqdm
pip install --no-index pyyaml || pip install pyyaml

# Verify installation
echo ""
echo "Installed packages:"
pip list | grep -E "ultralytics|torch|opencv|pycocotools"

# ============================================================================
# 4. Test Installation
# ============================================================================
echo ""
echo "[4/4] Testing installation..."

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')

from ultralytics import YOLO
print('ultralytics: OK')

import cv2
print(f'OpenCV: {cv2.__version__}')

from pycocotools import mask
print('pycocotools: OK')
"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "Virtual environment: ${VENV_PATH}"
echo ""
echo "Next steps:"
echo "  1. Prepare your data tarball:"
echo "     tar -czf data/mbari_dataset.tar.gz -C data yolo_dataset"
echo ""
echo "  2. Submit a training job:"
echo "     sbatch slurm/train.sh"
echo ""
