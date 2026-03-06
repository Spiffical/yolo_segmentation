
# YOLO Detection and Segmentation Training for MBARI Underwater Images

This repository contains tools for training Ultralytics YOLO detection and
instance segmentation models on MBARI/FathomNet underwater imagery.

## Features

- **COCO to YOLO conversion** for detection boxes and segmentation polygons
- **Curated coarse label plans** for MBARI/FathomNet long-tail labels
- **Automatic train/val splitting** with stratification
- **Local training scripts** with full CLI control
- **DRAC cluster support** with SLURM submission scripts
- **Sweep submission with reproducible manifests**
- **Single-file benchmark reports across many checkpoints**
- **Local video inference with per-video summary CSV**

## Quick Start

### 1. Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Convert COCO Dataset to YOLO Format

```bash
# Binary detection from native COCO boxes
python scripts/convert_coco_to_yolo.py \
    --coco_json data/seg_masks/train.json \
    --output_dir data/yolo_detect_binary \
    --image_dir /mnt/z/yolo/data/images/train \
    --task detect \
    --mode binary

# Coarse detection using the curated bio8 label plan
python scripts/convert_coco_to_yolo.py \
    --coco_json data/seg_masks/train.json \
    --output_dir data/yolo_detect_coarse \
    --image_dir /mnt/z/yolo/data/images/train \
    --task detect \
    --mode coarse \
    --label_plan coarse_v1_bio8

# Binary segmentation from masks
python scripts/convert_coco_to_yolo.py \
    --coco_json data/seg_masks/train.json \
    --output_dir data/yolo_segment_binary \
    --image_dir /mnt/z/yolo/data/images/train \
    --task segment \
    --mode binary
```

### 3. Validate Conversion (Optional)

```bash
# Visualize random samples to verify segmentation masks
python scripts/validate.py --dataset data/yolo_segment_binary --n_samples 5
```

### 4. Train Locally

```bash
# Quick binary detection smoke test
python scripts/train.py \
    --data data/yolo_detect_binary/dataset.yaml \
    --model yolo11n.pt \
    --epochs 5 \
    --batch 8

# Quick segmentation smoke test
python scripts/train.py \
    --data data/yolo_segment_binary/dataset.yaml \
    --model yolo11n-seg.pt \
    --epochs 5 \
    --batch 4

# Full coarse detection training
python scripts/train.py \
    --data data/yolo_detect_coarse/dataset.yaml \
    --model yolo11l.pt \
    --epochs 100 \
    --batch 16

# Full segmentation training
python scripts/train.py \
    --data data/yolo_segment_binary/dataset.yaml \
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
│   ├── submit_sweep.py           # Sweep submission helper
│   ├── benchmark_models.py       # Checkpoint benchmarking report
│   ├── predict_videos.py         # Local video inference
│   └── prepare_subset.py         # Create test subsets
├── slurm/                # Cluster submission scripts
│   ├── train.sh          # SLURM job script
│   ├── evaluate_models.sh # Evaluate many checkpoints + report
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
# Binary detection (object vs background)
sbatch slurm/train.sh --task detect --mode binary

# Coarse biological detection
sbatch slurm/train.sh \
    --task detect \
    --mode coarse \
    --label-plan coarse_v1_bio8

# Top 100 segmentation classes
sbatch slurm/train.sh --task segment --mode top_n --top_n 100

# Full segmentation dataset
sbatch slurm/train.sh --task segment --mode all

# If code is on /project but venv is under /home:
sbatch slurm/train.sh \
    --repo /project/def-kmoran/<user>/yolo_segmentation \
    --venv /home/<user>/yolo_segmentation/.venv \
    --task detect \
    --mode binary
```

The job will:
1. Extract raw data to `SLURM_TMPDIR` (fast local SSD)
2. Convert COCO annotations → YOLO detection or segmentation labels
3. Train the requested YOLO task
4. Save results under `$SCRATCH/yolo_seg/...`

`slurm/train.sh` automatically enables multi-GPU DDP when more than one GPU is
allocated (for example, `sbatch --gpus-per-node=h100:4 ...`).

## Sweep Workflow (Nibi + W&B)

Use the example sweep configs:

```bash
# Binary detection
python scripts/submit_sweep.py \
    --config configs/nibi_detect_binary.yaml

# Coarse detection
python scripts/submit_sweep.py \
    --config configs/nibi_detect_coarse_v1.yaml

# Binary segmentation
python scripts/submit_sweep.py \
    --config configs/nibi_sweep_binary.yaml
```

Sweep YAML also supports optional Slurm resource keys in `global` or per
experiment:
- `gpus_per_node` (e.g. `h100:1`)
- `cpus_per_task` (e.g. `6`)
- `mem` (e.g. `32000M`)
- `time` (e.g. `24:00:00`)

For a FathomNet-like segmentation regime (`imgsz=640`, `optimizer=auto`,
`patience=5`, `lr0=0.01`), use:

```bash
python scripts/submit_sweep.py \
    --config configs/nibi_sweep_binary_fathomnet_like.yaml
```

For segmentation multiclass experiments:

```bash
# Top-N multiclass (recommended first)
python scripts/submit_sweep.py \
    --config configs/nibi_sweep_multiclass_topn.yaml

# Full multiclass (all classes)
python scripts/submit_sweep.py \
    --config configs/nibi_sweep_multiclass_all.yaml
```

This dry-run writes commands and a manifest under `runs/sweeps/...`.

To actually submit a sweep:

```bash
python scripts/submit_sweep.py \
    --config configs/nibi_detect_binary.yaml \
    --submit
```

Each run gets a deterministic W&B name/group and a saved command manifest for
later traceability.

Practical queue-friendly default on Nibi:
- `gpus_per_node: h100:1`
- `cpus_per_task: 6`
- `mem: 32000M`
- Batch sizes chosen to keep per-GPU load similar to your earlier 2/4-GPU runs.

## Benchmark All Checkpoints Into One Report

After training finishes, run one evaluation job that benchmarks all discovered
`weights/best.pt` checkpoints and writes ranked report files:

```bash
sbatch slurm/evaluate_models.sh \
    --checkpoints-root /scratch/$USER/yolo_seg/2026-02-13 \
    --data /project/def-kmoran/merileo/yolo_segmentation/data/mbari_raw.tar.gz \
    --task detect \
    --mode binary \
    --splits val,test \
    --split val
```

Outputs:
- `benchmark_report.md` (human-readable ranking)
- `benchmark_report.csv` (analysis-friendly table)
- `benchmark_report.json` (machine-readable summary + metrics)
- `eval_runs/.../confusion_matrix.png` for each model/split
- `eval_runs/.../confusion_matrix_normalized.png` for each model/split
- `train_results_plot` paths in report rows (points to each run's `results.png`)

## Profile Labels

Inspect class distribution from the COCO JSON before choosing multiclass settings:

```bash
python scripts/profile_labels.py \
    --coco-json data/seg_masks/train.json \
    --output-dir runs/label_profile
```

Outputs:
- `runs/label_profile/label_profile.md`
- `runs/label_profile/category_counts.csv`

## Coarse Label Plans

The MBARI/FathomNet taxonomy is long-tailed and mixed-resolution, so the repo
includes curated coarse label plans in `configs/label_plans.yaml`.

- `coarse_v1_bio8`: `echinoderm`, `coral_anemone`, `gelatinous`, `sponge`,
  `fish`, `crustacean`, `other_invertebrate`, `cephalopod`
- `coarse_v1_bio8_plus_gear`: same as above plus `gear_debris`

The design rationale and approximate coverage are documented in
[docs/coarse_label_plan.md](/home/sbialek/ONC/yolo_segmentation/docs/coarse_label_plan.md).

## Run Best Model On Local ONC Videos

```bash
python scripts/predict_videos.py \
    --model /path/to/best.pt \
    --source /path/to/onc/videos \
    --recursive \
    --project runs/predict \
    --name onc_validation
```

Outputs include annotated videos and `inference_summary.csv`.

## Category Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `all` | All native categories | Full multi-class training |
| `top_n` | Top N categories by count | Focus on common native labels |
| `binary` | Object vs background | Species-agnostic detection or segmentation |
| `coarse` | Named plan from `configs/label_plans.yaml` | Stable biological grouping |

## Model Sizes

| Model | Task | Parameters | Speed | Accuracy |
|-------|------|------------|-------|----------|
| yolo11n.pt | detect | small | Fastest | Good |
| yolo11l.pt | detect | large | Slow | Better |
| yolo11n-seg.pt | segment | 2.6M | Fastest | Good |
| yolo11m-seg.pt | segment | 25.3M | Medium | Best local |
| yolo11x-seg.pt | segment | 68.7M | Slowest | Best |

For local detection testing, start with `yolo11n.pt`. For local segmentation
testing, start with `yolo11n-seg.pt`. On cluster, use `yolo11l.pt`,
`yolo11m-seg.pt`, or larger models as resources allow.

## License

MIT License
