#!/bin/bash
# ============================================================================
# Prepare Raw Data Tarball for DRAC Cluster
# ============================================================================
#
# This script packages the RAW COCO data (JSON + images) for cluster transfer.
# The conversion to YOLO format happens on the cluster at job start.
#
# Usage:
#   bash slurm/prepare_data.sh [coco_json] [image_dir] [output]
#
# Examples:
#   bash slurm/prepare_data.sh  # Use defaults
#   bash slurm/prepare_data.sh data/seg_masks/train.json /mnt/z/yolo/data/images/train
#
# ============================================================================

set -e

# Configuration
COCO_JSON="${1:-data/seg_masks/train.json}"
IMAGE_DIR="${2:-/mnt/z/yolo/data/images/train}"
OUTPUT_TARBALL="${3:-data/mbari_raw.tar.gz}"

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

# Count images
echo ""
echo "Analyzing dataset..."
python3 << EOF
import json
import os

with open('${COCO_JSON}', 'r') as f:
    data = json.load(f)

image_dir = '${IMAGE_DIR}'
total = len(data['images'])
found = sum(1 for img in data['images'] if os.path.exists(os.path.join(image_dir, img['file_name'])))

print(f"Images in JSON: {total}")
print(f"Images found: {found}")
print(f"Annotations: {len(data['annotations'])}")
print(f"Categories: {len(data['categories'])}")

if found < total:
    print(f"WARNING: {total - found} images are missing!")
EOF

# Create tarball
# We use a simple approach: tar the JSON and the image directory
echo ""
echo "Creating tarball..."
echo "This may take a while for large datasets..."

# Get absolute paths
COCO_JSON_ABS=$(realpath "${COCO_JSON}")
IMAGE_DIR_ABS=$(realpath "${IMAGE_DIR}")

# Create tar archive
# Structure: train.json + images/
tar -czhf "${OUTPUT_TARBALL}" \
    -C "$(dirname "${COCO_JSON_ABS}")" "$(basename "${COCO_JSON_ABS}")" \
    -C "$(dirname "${IMAGE_DIR_ABS}")" "$(basename "${IMAGE_DIR_ABS}")"

# Rename entries to expected structure (train.json + images/)
# Actually tar doesn't easily allow renaming. Let's check what we created:
echo ""
echo "Tarball contents (first 10 entries):"
tar -tzf "${OUTPUT_TARBALL}" | head -10

TARBALL_SIZE=$(du -sh "${OUTPUT_TARBALL}" | cut -f1)
echo ""
echo "============================================"
echo "Tarball Created!"
echo "============================================"
echo ""
echo "Output: ${OUTPUT_TARBALL}"
echo "Size: ${TARBALL_SIZE}"
echo ""
echo "Expected structure inside tar:"
echo "  - $(basename "${COCO_JSON_ABS}")  (COCO annotations)"
echo "  - $(basename "${IMAGE_DIR_ABS}")/  (images)"
echo ""
echo "Upload to DRAC (use Globus for large files):"
echo "  scp ${OUTPUT_TARBALL} <user>@narval.computecanada.ca:~/projects/def-kmoran/<user>/yolo_segmentation/data/"
echo ""
echo "Then submit job:"
echo "  sbatch slurm/train.sh --mode top_n --top_n 100"
