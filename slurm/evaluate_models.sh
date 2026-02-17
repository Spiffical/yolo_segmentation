#!/bin/bash
#SBATCH --account=def-kmoran
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=08:00:00
#SBATCH --job-name=yolo_eval
#SBATCH --output=/home/%u/yolo_segmentation/logs/%x-%j.out
#SBATCH --error=/home/%u/yolo_segmentation/logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=spencer.bialek@gmail.com

# ============================================================================
# Benchmark multiple training checkpoints and generate one ranked report.
# ============================================================================
#
# Example:
#   sbatch slurm/evaluate_models.sh \
#       --checkpoints-root /scratch/$USER/yolo_seg/2026-02-13 \
#       --data /project/def-kmoran/merileo/yolo_segmentation/data/mbari_test_raw.tar.gz \
#       --mode binary \
#       --splits val,test \
#       --split val
#
# You can skip --data and pass --dataset-yaml if you already prepared a YOLO
# dataset config:
#   sbatch slurm/evaluate_models.sh \
#       --checkpoints-root /scratch/$USER/yolo_seg/2026-02-13 \
#       --dataset-yaml /scratch/$USER/mbari_eval/yolo_dataset/dataset.yaml \
#       --splits val,test
# ============================================================================

set -euo pipefail

DEFAULT_REPO="${HOME}/yolo_segmentation"
REPO_DIR=""
VENV_PATH=""
CHECKPOINTS_ROOT=""
DATASET_YAML=""
DATA_TARBALL=""
CONVERT_MODE="binary"
CONVERT_TOP_N="100"
VAL_RATIO="0.2"
SPLIT_SEED="42"
EVAL_SPLIT="val"
EVAL_SPLITS="val,test"
EVAL_IMGSZ="640"
EVAL_BATCH="16"
EVAL_WORKERS="8"
EVAL_DEVICE="auto"
PRIMARY_METRIC="metrics/mAP50-95(M)"
CHECKPOINT_PATTERN="weights/best.pt"
MAX_MODELS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --repo)
            REPO_DIR="$2"
            shift 2
            ;;
        --checkpoints-root)
            CHECKPOINTS_ROOT="$2"
            shift 2
            ;;
        --venv)
            VENV_PATH="$2"
            shift 2
            ;;
        --dataset-yaml)
            DATASET_YAML="$2"
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
        --top_n|--top-n)
            CONVERT_TOP_N="$2"
            shift 2
            ;;
        --val_ratio|--val-ratio)
            VAL_RATIO="$2"
            shift 2
            ;;
        --split-seed)
            SPLIT_SEED="$2"
            shift 2
            ;;
        --split)
            EVAL_SPLIT="$2"
            shift 2
            ;;
        --splits)
            EVAL_SPLITS="$2"
            shift 2
            ;;
        --imgsz)
            EVAL_IMGSZ="$2"
            shift 2
            ;;
        --batch)
            EVAL_BATCH="$2"
            shift 2
            ;;
        --workers)
            EVAL_WORKERS="$2"
            shift 2
            ;;
        --device)
            EVAL_DEVICE="$2"
            shift 2
            ;;
        --primary-metric)
            PRIMARY_METRIC="$2"
            shift 2
            ;;
        --pattern)
            CHECKPOINT_PATTERN="$2"
            shift 2
            ;;
        --max-models)
            MAX_MODELS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ -z "${REPO_DIR}" ]; then
    if [ -f "scripts/benchmark_models.py" ]; then
        REPO_DIR="$(pwd)"
    elif [ -d "${DEFAULT_REPO}" ]; then
        REPO_DIR="${DEFAULT_REPO}"
    else
        echo "ERROR: Could not find repo. Use --repo /path/to/yolo_segmentation"
        exit 1
    fi
fi

if [ -z "${CHECKPOINTS_ROOT}" ]; then
    echo "ERROR: --checkpoints-root is required"
    exit 1
fi

if [ -z "${VENV_PATH}" ]; then
    for candidate in \
        "${REPO_DIR}/.venv" \
        "${DEFAULT_REPO}/.venv" \
        "${HOME}/.venv"
    do
        if [ -d "${candidate}" ]; then
            VENV_PATH="${candidate}"
            break
        fi
    done
fi

OUTPUT_BASE="${SCRATCH}/yolo_seg_eval"
RUN_DATE=$(date +%Y-%m-%d)
RUN_NAME="eval_${SLURM_JOB_ID}"
OUTPUT_DIR="${OUTPUT_BASE}/${RUN_DATE}/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${REPO_DIR}/logs"

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started: $(date)"
echo "Repo: ${REPO_DIR}"
echo "Checkpoints root: ${CHECKPOINTS_ROOT}"
echo "Dataset YAML: ${DATASET_YAML:-<generated from raw data>}"
echo "Data tarball: ${DATA_TARBALL:-<not provided>}"
echo "Venv: ${VENV_PATH:-not set}"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"

cat > "${OUTPUT_DIR}/eval_config.yaml" << EOF
job_id: ${SLURM_JOB_ID}
node: ${SLURM_NODELIST}
date: $(date -Iseconds)
repo_dir: ${REPO_DIR}
venv_path: ${VENV_PATH}
checkpoints_root: ${CHECKPOINTS_ROOT}
dataset_yaml: ${DATASET_YAML}
data_tarball: ${DATA_TARBALL}
mode: ${CONVERT_MODE}
top_n: ${CONVERT_TOP_N}
val_ratio: ${VAL_RATIO}
split_seed: ${SPLIT_SEED}
split: ${EVAL_SPLIT}
splits: ${EVAL_SPLITS}
imgsz: ${EVAL_IMGSZ}
batch: ${EVAL_BATCH}
workers: ${EVAL_WORKERS}
device: ${EVAL_DEVICE}
primary_metric: ${PRIMARY_METRIC}
pattern: ${CHECKPOINT_PATTERN}
max_models: ${MAX_MODELS}
EOF

echo "[1/4] Loading environment..."
if [ -f "${REPO_DIR}/slurm/.modules" ]; then
    source "${REPO_DIR}/slurm/.modules"
else
    module load StdEnv/2023
    module load python/3.11 cuda cudnn
    module load scipy-stack
    module load opencv/4.12.0
fi

if [ -d "${VENV_PATH}" ]; then
    source "${VENV_PATH}/bin/activate"
else
    echo "ERROR: Missing venv at ${VENV_PATH:-<empty>}"
    echo "Provide --venv /path/to/.venv"
    exit 1
fi

python -c "from ultralytics import YOLO; import yaml; print('Dependencies OK')"

cd "${REPO_DIR}"

if [ -z "${DATASET_YAML}" ]; then
    if [ -z "${DATA_TARBALL}" ]; then
        echo "ERROR: Provide either --dataset-yaml or --data"
        exit 1
    fi

    echo "[2/4] Preparing evaluation dataset from raw tarball..."
    LOCAL_DATA_DIR="${SLURM_TMPDIR}/eval_raw_data"
    YOLO_DATASET="${SLURM_TMPDIR}/eval_yolo_dataset"
    mkdir -p "${LOCAL_DATA_DIR}"
    tar -xzf "${DATA_TARBALL}" -C "${LOCAL_DATA_DIR}"

    COCO_JSON=$(find "${LOCAL_DATA_DIR}" -name "*.json" -type f | head -1)
    IMAGE_DIR=$(find "${LOCAL_DATA_DIR}" -type d -name "images" | head -1)
    if [ -z "${IMAGE_DIR}" ]; then
        IMAGE_DIR=$(find "${LOCAL_DATA_DIR}" -type d -name "train" | head -1)
    fi
    if [ -z "${IMAGE_DIR}" ]; then
        IMAGE_DIR=$(find "${LOCAL_DATA_DIR}" -type f \( -name "*.jpg" -o -name "*.png" \) | head -1 | xargs dirname)
    fi

    if [ -z "${COCO_JSON}" ] || [ -z "${IMAGE_DIR}" ]; then
        echo "ERROR: Could not infer COCO JSON and image directory from ${DATA_TARBALL}"
        exit 1
    fi

    python scripts/convert_coco_to_yolo.py \
        --coco_json "${COCO_JSON}" \
        --output_dir "${YOLO_DATASET}" \
        --image_dir "${IMAGE_DIR}" \
        --val_ratio "${VAL_RATIO}" \
        --seed "${SPLIT_SEED}" \
        --mode "${CONVERT_MODE}" \
        --top_n "${CONVERT_TOP_N}" \
        --min_annotations 0 \
        --workers "${EVAL_WORKERS}"

    DATASET_YAML="${YOLO_DATASET}/dataset.yaml"
else
    echo "[2/4] Using provided dataset YAML."
fi

if [ ! -f "${DATASET_YAML}" ]; then
    echo "ERROR: Dataset YAML not found: ${DATASET_YAML}"
    exit 1
fi

echo "[3/4] Running benchmark across checkpoints..."
BENCHMARK_CMD=(
    python scripts/benchmark_models.py
    --dataset "${DATASET_YAML}"
    --checkpoints-root "${CHECKPOINTS_ROOT}"
    --pattern "${CHECKPOINT_PATTERN}"
    --splits ${EVAL_SPLITS//,/ }
    --rank-split "${EVAL_SPLIT}"
    --imgsz "${EVAL_IMGSZ}"
    --batch "${EVAL_BATCH}"
    --workers "${EVAL_WORKERS}"
    --device "${EVAL_DEVICE}"
    --primary-metric "${PRIMARY_METRIC}"
    --project "${OUTPUT_DIR}"
    --name "benchmark"
)

if [ -n "${MAX_MODELS}" ]; then
    BENCHMARK_CMD+=(--max-models "${MAX_MODELS}")
fi

"${BENCHMARK_CMD[@]}"

echo "[4/4] Finished."
echo "Report directory: ${OUTPUT_DIR}/benchmark"
echo "Markdown report: ${OUTPUT_DIR}/benchmark/benchmark_report.md"
echo "CSV report: ${OUTPUT_DIR}/benchmark/benchmark_report.csv"
echo "JSON report: ${OUTPUT_DIR}/benchmark/benchmark_report.json"
