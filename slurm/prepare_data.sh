#!/bin/bash
# ============================================================================
# Prepare Raw Data Tarball for DRAC Cluster
# ============================================================================
#
# This script packages the RAW COCO data (JSON + images) for cluster transfer.
# The conversion to YOLO format happens on the cluster at job start.
#
# Usage:
#   bash slurm/prepare_data.sh
#
# Expected input structure:
#   data/seg_masks/train.json     # COCO annotations with RLE masks
#   data/images/train/            # OR a mounted drive with images
#
# Output:
#   data/mbari_raw.tar.gz         # Tarball for cluster transfer
#
# ============================================================================

set -e

# Configuration
COCO_JSON="${1:-data/seg_masks/train.json}"
IMAGE_DIR="${2:-/mnt/z/yolo/data/images/train}"
OUTPUT_TARBALL="${3:-data/mbari_raw.tar.gz}"
STAGING_DIR="data/.staging_raw"

echo "============================================"
echo "Preparing Raw Data Tarball for DRAC"
echo "============================================"
echo ""
echo "COCO JSON: ${COCO_JSON}"
echo "Image directory: ${IMAGE_DIR}"
echo "Output: ${OUTPUT_TARBALL}"

# Verify inputs exist
if [ ! -f "${COCO_JSON}" ]; then
    echo "ERROR: COCO JSON not found: ${COCO_JSON}"
    exit 1
fi

if [ ! -d "${IMAGE_DIR}" ]; then
    echo "ERROR: Image directory not found: ${IMAGE_DIR}"
    echo "Make sure the drive is mounted or provide correct path."
    exit 1
fi

# Create staging directory
echo ""
echo "Creating staging directory..."
rm -rf "${STAGING_DIR}"
mkdir -p "${STAGING_DIR}/images"

# Copy COCO JSON
echo "Copying COCO JSON..."
cp "${COCO_JSON}" "${STAGING_DIR}/train.json"

# Get list of images from JSON and copy them
echo "Extracting image list from JSON..."
python3 -c "
import json
import os
import shutil
from tqdm import tqdm

with open('${COCO_JSON}', 'r') as f:
    data = json.load(f)

images = data['images']
image_dir = '${IMAGE_DIR}'
staging_dir = '${STAGING_DIR}/images'

print(f'Copying {len(images)} images...')
copied = 0
missing = 0

for img in tqdm(images):
    src = os.path.join(image_dir, img['file_name'])
    dst = os.path.join(staging_dir, img['file_name'])
    if os.path.exists(src):
        shutil.copy2(src, dst)
        copied += 1
    else:
        missing += 1

print(f'Copied: {copied}, Missing: {missing}')
"

# Get size info
echo ""
echo "Staging directory contents:"
TOTAL_IMAGES=$(ls "${STAGING_DIR}/images" | wc -l)
TOTAL_SIZE=$(du -sh "${STAGING_DIR}" | cut -f1)
echo "  - Images: ${TOTAL_IMAGES}"
echo "  - Total size: ${TOTAL_SIZE}"

# Create tarball
echo ""
echo "Creating tarball (this may take a while for large datasets)..."
tar -czf "${OUTPUT_TARBALL}" -C "${STAGING_DIR}" .

# Cleanup staging
echo "Cleaning up staging directory..."
rm -rf "${STAGING_DIR}"

# Show result
TARBALL_SIZE=$(du -sh "${OUTPUT_TARBALL}" | cut -f1)
echo ""
echo "============================================"
echo "Tarball Created Successfully!"
echo "============================================"
echo ""
echo "Output: ${OUTPUT_TARBALL}"
echo "Size: ${TARBALL_SIZE}"
echo ""
echo "To upload to DRAC (use Globus for large files):"
echo "  scp ${OUTPUT_TARBALL} <username>@narval.computecanada.ca:~/projects/def-kmoran/<username>/yolo_segmentation/data/"
echo ""
echo "Then submit training job:"
echo "  sbatch slurm/train.sh --mode top_n --top_n 100"
