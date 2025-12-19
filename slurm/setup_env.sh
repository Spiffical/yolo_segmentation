#!/bin/bash
# ============================================================================
# Setup Python Environment for DRAC Cluster
# ============================================================================
#
# Run this script ONCE before submitting training jobs.
# This creates a persistent virtual environment with proper DRAC modules.
#
# Best practices followed:
#   - Use StdEnv for consistent software environment
#   - Load opencv via module (not pip) for optimized binaries
#   - Use --no-index for pip to prefer Alliance wheels
#   - Load scipy-stack for optimized numpy/scipy
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

# Load standard environment first
module load StdEnv/2023

# Load Python and GPU support
module load python/3.11 cuda cudnn

# Load scipy-stack for optimized numpy, scipy (must come before opencv)
module load scipy-stack

# Load OpenCV via module (provides optimized opencv_python bindings)
module load opencv/4.8.1

echo ""
echo "Loaded modules:"
module list

echo ""
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

# Use --no-download to use only system packages initially
virtualenv --no-download "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

echo "Virtual environment created and activated."
which python
python --version

# ============================================================================
# 3. Install Dependencies
# ============================================================================
echo ""
echo "[3/4] Installing dependencies..."

# Upgrade pip using Alliance wheel
pip install --no-index --upgrade pip

# Check what wheels are available for our packages
echo ""
echo "Checking available wheels..."
avail_wheels ultralytics pycocotools tqdm pyyaml wandb || true

# Install packages - prefer Alliance wheels (--no-index), fallback to PyPI
echo ""
echo "Installing packages..."

# Core packages from Alliance wheels
pip install --no-index ultralytics || pip install ultralytics
pip install --no-index pycocotools || pip install pycocotools
pip install --no-index tqdm || pip install tqdm
pip install --no-index pyyaml || pip install pyyaml

# Weights & Biases for experiment tracking
pip install --no-index wandb || pip install wandb

# Note: OpenCV is already available from the opencv module
# Note: numpy/scipy are from scipy-stack module

# Show installed packages
echo ""
echo "Installed packages:"
pip list | grep -iE "ultralytics|torch|opencv|pycocotools|numpy|wandb" || true

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
    print(f'GPU count: {torch.cuda.device_count()}')

from ultralytics import YOLO
print('ultralytics: OK')

import cv2
print(f'OpenCV: {cv2.__version__}')

from pycocotools import mask
print('pycocotools: OK')

import numpy as np
print(f'NumPy: {np.__version__}')
"

# ============================================================================
# 5. Save Module Load Commands
# ============================================================================
# Save the module load commands so train.sh can use them
cat > "${PROJECT_DIR}/slurm/.modules" << 'EOF'
module load StdEnv/2023
module load python/3.11 cuda cudnn
module load opencv/4.8.1
module load scipy-stack
EOF

echo ""
echo "Module load commands saved to slurm/.modules"

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
echo "Required modules (load these in your job script):"
echo "  module load StdEnv/2023"
echo "  module load python/3.11 cuda cudnn"
echo "  module load opencv/4.8.1"
echo "  module load scipy-stack"
echo ""
echo "Next steps:"
echo "  1. Prepare your data tarball:"
echo "     bash slurm/prepare_data.sh"
echo ""
echo "  2. Submit a training job:"
echo "     sbatch slurm/train.sh --mode binary"
echo ""
