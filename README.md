
# YOLO Segmentation Training for MBARI Underwater Images

This repository contains tools for training YOLOv11 instance segmentation models on MBARI/FathomNet underwater imagery.

## Features

- **COCO RLE to YOLO polygon conversion** with multiple category modes
- **Automatic train/val splitting** with stratification
- **Local training scripts** with full CLI control
- **DRAC cluster support** with SLURM submission scripts

## Quick Start

### 1. Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Convert COCO Dataset to YOLO Format

```bash
# Full dataset (all 1897 categories)
python scripts/convert_coco_to_yolo.py \
    --coco_json data/seg_masks/train.json \
    --output_dir data/yolo_dataset \
    --image_dir /mnt/z/yolo/data/images/train \
    --mode all

# Top 50 categories only
python scripts/convert_coco_to_yolo.py \
    --mode top_n --top_n 50

# Binary segmentation (object vs background)
python scripts/convert_coco_to_yolo.py \
    --mode binary
```

### 3. Validate Conversion (Optional)

```bash
# Visualize random samples to verify masks
python scripts/validate.py --n_samples 5
```

### 4. Train Locally

```bash
# Quick test with small model
python scripts/train.py \
    --model yolo11n-seg.pt \
    --epochs 10 \
    --batch 8

# Full training with medium model
python scripts/train.py \
    --model yolo11m-seg.pt \
    --epochs 100 \
    --batch 16
```

## Project Structure

```
yolo_segmentation/
├── configs/              # Configuration files
├── scripts/              # Main scripts
│   ├── convert_coco_to_yolo.py  # Data conversion
│   ├── train.py                  # Training
│   ├── validate.py               # Visualization
│   └── prepare_subset.py         # Create test subsets
├── slurm/                # Cluster submission scripts
│   ├── train.sh          # SLURM job script
│   ├── setup_env.sh      # Environment setup
│   └── prepare_data.sh   # Data tarball creation
├── src/                  # Shared utilities
│   └── data_utils.py     # Data manipulation functions
├── data/                 # Data directory (gitignored)
│   ├── seg_masks/        # COCO JSON annotations
│   └── yolo_dataset/     # Converted YOLO format
└── runs/                 # Training outputs (gitignored)
```

## DRAC Cluster Deployment

The cluster workflow uses **raw COCO data** - conversion to YOLO format happens on the cluster at job start for faster I/O from `SLURM_TMPDIR`.

### 1. Setup Environment (run once on cluster)

```bash
# On the cluster login node
bash slurm/setup_env.sh
```

### 2. Prepare Data Tarball (locally)

```bash
# Package raw COCO JSON + images
bash slurm/prepare_data.sh

# This creates data/mbari_raw.tar.gz with:
#   - train.json (COCO annotations with RLE masks)
#   - images/ (all training images)
```

### 3. Upload to Cluster

```bash
# For large datasets, use Globus instead of scp
scp data/mbari_raw.tar.gz <user>@narval.computecanada.ca:~/projects/def-kmoran/<user>/yolo_segmentation/data/
```

### 4. Submit Training Job

```bash
# Binary segmentation (object vs background)
sbatch slurm/train.sh --mode binary

# Top 100 categories
sbatch slurm/train.sh --mode top_n --top_n 100

# All categories (1897 classes)
sbatch slurm/train.sh --mode all
```

The job will:
1. Extract raw data to `SLURM_TMPDIR` (fast local SSD)
2. Convert COCO RLE → YOLO polygons
3. Train YOLOv11 segmentation
4. Save results to `runs/segment/`

## Category Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `all` | All 1897 categories | Full multi-class segmentation |
| `top_n` | Top N categories by count | Focus on common species |
| `binary` | Object vs background | Species-agnostic detection |

## Model Sizes

| Model | Parameters | Speed | Accuracy |
|-------|------------|-------|----------|
| yolo11n-seg | 2.6M | Fastest | Good |
| yolo11s-seg | 11.2M | Fast | Better |
| yolo11m-seg | 25.3M | Medium | Best local |
| yolo11l-seg | 43.7M | Slow | Better |
| yolo11x-seg | 68.7M | Slowest | Best |

For local testing, start with `yolo11n-seg`. For cluster training, use `yolo11m-seg` or larger.

## License

MIT License
