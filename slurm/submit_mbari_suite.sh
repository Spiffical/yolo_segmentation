#!/bin/bash
# Submit a full MBARI experiment suite to Nibi as separate SLURM jobs.
#
# Usage:
#   bash slurm/submit_mbari_suite.sh
#   DRY_RUN=1 bash slurm/submit_mbari_suite.sh
#   INCLUDE_MIDWATER=1 bash slurm/submit_mbari_suite.sh
#
# CLI examples:
#   bash slurm/submit_mbari_suite.sh \
#     --data /project/rpp-kmoran/merileo/data/mbari_raw.tar.gz \
#     --checkpoint-root /project/rpp-kmoran/merileo/yolo_data/models/fathomnet
#   bash slurm/submit_mbari_suite.sh --dry-run --include-midwater
#
# Environment-variable overrides also work:
#   REPO_DIR=/home/merileo/yolo_segmentation
#   VENV_PATH=/home/merileo/yolo_segmentation/.venv
#   DATA_TARBALL=/project/rpp-kmoran/merileo/data/mbari_raw.tar.gz
#   CHECKPOINT_ROOT=/project/rpp-kmoran/merileo/yolo_data/models/fathomnet
#   SUITE_TAG=20260305_a
#   DRY_RUN=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

usage() {
    cat <<'EOF'
Usage:
  bash slurm/submit_mbari_suite.sh [options]

Path overrides:
  --repo DIR                 Repo root containing slurm/train.sh
  --venv DIR                 Python virtualenv path
  --data FILE                MBARI raw tarball path
  --checkpoint-root DIR      Root containing fathomnet/{megalodon,mbari_315k_yolov8,midwater_2025}
  --label-plans-file FILE    Coarse label YAML
  --manifest-dir DIR         Submission manifest output directory
  --megalodon-pt FILE        Override discovered Megalodon checkpoint path
  --mbari315k-pt FILE        Override discovered MBARI-315k checkpoint path
  --midwater-pt FILE         Override discovered Midwater checkpoint path

Suite behavior:
  --suite-tag TAG            W&B / manifest suite tag
  --dry-run                  Print and record sbatch commands without submitting
  --include-midwater         Include the midwater detector when found
  --include-gear-variant     Include coarse_v1_bio8_plus_gear runs

Resources:
  --gpus-per-node SPEC       Default: h100:1
  --cpus-per-task N          Default: 6
  --mem SIZE                 Default: 32000M
  --time HH:MM:SS            Default: 24:00:00

Toggles:
  --no-binary-detect
  --no-coarse-detect
  --no-binary-segment
  --no-coarse-segment

Examples:
  bash slurm/submit_mbari_suite.sh \
    --data /project/rpp-kmoran/merileo/data/mbari_raw.tar.gz \
    --checkpoint-root /project/rpp-kmoran/merileo/yolo_data/models/fathomnet \
    --dry-run
EOF
}

DEFAULT_DATA_CANDIDATES=(
    "/project/rpp-kmoran/merileo/data/mbari_raw.tar.gz"
    "/project/rpp-kmoran/merileo/yolo_segmentation/data/mbari_raw.tar.gz"
    "${REPO_DIR}/data/mbari_raw.tar.gz"
)

find_first_pt() {
    local search_dir="$1"
    if [[ ! -d "${search_dir}" ]]; then
        return 1
    fi
    find "${search_dir}" -maxdepth 2 -type f -name "*.pt" | sort | head -n 1
}

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/project/rpp-kmoran/merileo/yolo_data/models/fathomnet}"
SUITE_TAG="${SUITE_TAG:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
INCLUDE_MIDWATER="${INCLUDE_MIDWATER:-0}"
INCLUDE_GEAR_VARIANT="${INCLUDE_GEAR_VARIANT:-0}"
RUN_BINARY_DETECT="${RUN_BINARY_DETECT:-1}"
RUN_COARSE_DETECT="${RUN_COARSE_DETECT:-1}"
RUN_BINARY_SEGMENT="${RUN_BINARY_SEGMENT:-1}"
RUN_COARSE_SEGMENT="${RUN_COARSE_SEGMENT:-1}"

GPUS_PER_NODE="${GPUS_PER_NODE:-h100:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-6}"
MEM="${MEM:-32000M}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"

DETECT_EPOCHS="${DETECT_EPOCHS:-100}"
SEG_EPOCHS="${SEG_EPOCHS:-100}"
DETECT_IMGSZ="${DETECT_IMGSZ:-960}"
SEG_IMGSZ="${SEG_IMGSZ:-640}"

WANDB_PROJECT_DET_BIN="${WANDB_PROJECT_DET_BIN:-yolo-detection-mbari-binary}"
WANDB_PROJECT_DET_COARSE="${WANDB_PROJECT_DET_COARSE:-yolo-detection-mbari-coarse}"
WANDB_PROJECT_SEG_BIN="${WANDB_PROJECT_SEG_BIN:-yolo-segmentation-mbari-binary}"
WANDB_PROJECT_SEG_COARSE="${WANDB_PROJECT_SEG_COARSE:-yolo-segmentation-mbari-coarse}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)
            REPO_DIR="$2"
            shift 2
            ;;
        --venv)
            VENV_PATH="$2"
            shift 2
            ;;
        --data)
            DATA_TARBALL="$2"
            shift 2
            ;;
        --checkpoint-root)
            CHECKPOINT_ROOT="$2"
            shift 2
            ;;
        --label-plans-file)
            LABEL_PLANS_FILE="$2"
            shift 2
            ;;
        --manifest-dir)
            MANIFEST_DIR="$2"
            shift 2
            ;;
        --suite-tag)
            SUITE_TAG="$2"
            shift 2
            ;;
        --megalodon-pt)
            MEGALODON_PT="$2"
            shift 2
            ;;
        --mbari315k-pt)
            MBARI315K_PT="$2"
            shift 2
            ;;
        --midwater-pt)
            MIDWATER_PT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --include-midwater)
            INCLUDE_MIDWATER=1
            shift
            ;;
        --include-gear-variant)
            INCLUDE_GEAR_VARIANT=1
            shift
            ;;
        --no-binary-detect)
            RUN_BINARY_DETECT=0
            shift
            ;;
        --no-coarse-detect)
            RUN_COARSE_DETECT=0
            shift
            ;;
        --no-binary-segment)
            RUN_BINARY_SEGMENT=0
            shift
            ;;
        --no-coarse-segment)
            RUN_COARSE_SEGMENT=0
            shift
            ;;
        --gpus-per-node)
            GPUS_PER_NODE="$2"
            shift 2
            ;;
        --cpus-per-task)
            CPUS_PER_TASK="$2"
            shift 2
            ;;
        --mem)
            MEM="$2"
            shift 2
            ;;
        --time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

TRAIN_SCRIPT="${REPO_DIR}/slurm/train.sh"
LABEL_PLANS_FILE="${LABEL_PLANS_FILE:-${REPO_DIR}/configs/label_plans.yaml}"
WANDB_GROUP_PREFIX="${WANDB_GROUP_PREFIX:-mbari_suite_${SUITE_TAG}}"

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "ERROR: train script not found: ${TRAIN_SCRIPT}" >&2
    exit 1
fi

if [[ ! -f "${LABEL_PLANS_FILE}" ]]; then
    echo "ERROR: label plan file not found: ${LABEL_PLANS_FILE}" >&2
    exit 1
fi

if [[ -z "${DATA_TARBALL:-}" ]]; then
    for candidate in "${DEFAULT_DATA_CANDIDATES[@]}"; do
        if [[ -f "${candidate}" ]]; then
            DATA_TARBALL="${candidate}"
            break
        fi
    done
fi

if [[ -z "${DATA_TARBALL:-}" || ! -f "${DATA_TARBALL}" ]]; then
    echo "ERROR: data tarball not found: ${DATA_TARBALL:-<unset>}" >&2
    echo "Tried default candidates:" >&2
    for candidate in "${DEFAULT_DATA_CANDIDATES[@]}"; do
        echo "  - ${candidate}" >&2
    done
    echo "Pass --data /path/to/mbari_raw.tar.gz" >&2
    exit 1
fi

if [[ -z "${VENV_PATH:-}" ]]; then
    for candidate in \
        "${REPO_DIR}/.venv" \
        "/home/merileo/yolo_segmentation/.venv" \
        "${HOME}/yolo_segmentation/.venv"
    do
        if [[ -d "${candidate}" ]]; then
            VENV_PATH="${candidate}"
            break
        fi
    done
fi

if [[ -z "${VENV_PATH:-}" || ! -d "${VENV_PATH}" ]]; then
    echo "ERROR: virtual environment not found: ${VENV_PATH:-<unset>}" >&2
    echo "Pass --venv /path/to/.venv" >&2
    exit 1
fi

if [[ -z "${MANIFEST_DIR:-}" ]]; then
    if [[ -n "${SCRATCH:-}" ]]; then
        MANIFEST_DIR="${SCRATCH}/yolo_seg/submissions"
    else
        MANIFEST_DIR="${REPO_DIR}/runs/submissions"
    fi
fi
mkdir -p "${MANIFEST_DIR}"
MANIFEST_PATH="${MANIFEST_DIR}/${SUITE_TAG}_mbari_suite.tsv"
printf "status\tjob_id\tname\tproject\tgroup\tmodel\tcommand\n" > "${MANIFEST_PATH}"

MEGALODON_PT="${MEGALODON_PT:-$(find_first_pt "${CHECKPOINT_ROOT}/megalodon" || true)}"
MBARI315K_PT="${MBARI315K_PT:-$(find_first_pt "${CHECKPOINT_ROOT}/mbari_315k_yolov8" || true)}"
MIDWATER_PT="${MIDWATER_PT:-$(find_first_pt "${CHECKPOINT_ROOT}/midwater_2025" || true)}"

echo "============================================"
echo "MBARI suite tag: ${SUITE_TAG}"
echo "Repo: ${REPO_DIR}"
echo "Venv: ${VENV_PATH}"
echo "Data: ${DATA_TARBALL}"
echo "Checkpoint root: ${CHECKPOINT_ROOT}"
echo "Dry run: ${DRY_RUN}"
echo "Manifest: ${MANIFEST_PATH}"
echo "--------------------------------------------"
echo "Megalodon checkpoint: ${MEGALODON_PT:-<missing>}"
echo "MBARI-315k checkpoint: ${MBARI315K_PT:-<missing>}"
echo "Midwater checkpoint: ${MIDWATER_PT:-<missing>}"
echo "============================================"

submitted_jobs=0
skipped_jobs=0

append_manifest() {
    local status="$1"
    local job_id="$2"
    local name="$3"
    local project="$4"
    local group="$5"
    local model="$6"
    local command="$7"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${status}" "${job_id}" "${name}" "${project}" "${group}" "${model}" "${command}" \
        >> "${MANIFEST_PATH}"
}

submit_job() {
    local name="$1"
    local wandb_project="$2"
    local wandb_group="$3"
    local wandb_tags="$4"
    local task="$5"
    local mode="$6"
    local label_plan="$7"
    local model="$8"
    local batch="$9"
    local epochs="${10}"
    local seed="${11}"
    local imgsz="${12}"
    local optimizer="${13}"
    local patience="${14}"
    local lrf="${15}"
    local save_period="${16}"
    local lr0="${17}"

    if [[ "${model}" == /* && ! -f "${model}" ]]; then
        echo "SKIP ${name}: missing checkpoint ${model}"
        append_manifest "skipped" "" "${name}" "${wandb_project}" "${wandb_group}" "${model}" "missing checkpoint"
        skipped_jobs=$((skipped_jobs + 1))
        return 0
    fi

    local -a cmd=(
        sbatch
        --parsable
        "--gpus-per-node=${GPUS_PER_NODE}"
        "--cpus-per-task=${CPUS_PER_TASK}"
        "--mem=${MEM}"
        "--time=${TIME_LIMIT}"
        "${TRAIN_SCRIPT}"
        --repo "${REPO_DIR}"
        --venv "${VENV_PATH}"
        --data "${DATA_TARBALL}"
        --task "${task}"
        --mode "${mode}"
        --top_n 100
        --split-seed "${seed}"
        --model "${model}"
        --epochs "${epochs}"
        --batch "${batch}"
        --wandb-project "${wandb_project}"
        --wandb-name "${name}"
        --wandb-group "${wandb_group}"
        --wandb-tags "${wandb_tags}"
        --imgsz "${imgsz}"
        --optimizer "${optimizer}"
        --patience "${patience}"
        --lrf "${lrf}"
        --save-period "${save_period}"
        --seed "${seed}"
        --lr0 "${lr0}"
    )

    if [[ -n "${label_plan}" ]]; then
        cmd+=(--label-plan "${label_plan}" --label-plans-file "${LABEL_PLANS_FILE}")
    fi

    local cmd_str
    cmd_str="$(printf '%q ' "${cmd[@]}")"

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "DRY RUN: ${cmd_str}"
        append_manifest "planned" "" "${name}" "${wandb_project}" "${wandb_group}" "${model}" "${cmd_str}"
        return 0
    fi

    local job_id
    job_id="$("${cmd[@]}")"
    echo "SUBMITTED ${name} -> job ${job_id}"
    append_manifest "submitted" "${job_id}" "${name}" "${wandb_project}" "${wandb_group}" "${model}" "${cmd_str}"
    submitted_jobs=$((submitted_jobs + 1))
}

submit_detect_binary() {
    local name="$1"
    local model="$2"
    local batch="$3"
    local lr0="$4"
    local seed="$5"
    local checkpoint_tag="$6"
    submit_job \
        "${name}" \
        "${WANDB_PROJECT_DET_BIN}" \
        "${WANDB_GROUP_PREFIX}_det_bin" \
        "mbari,detection,binary,nibi,suite_${SUITE_TAG},${checkpoint_tag}" \
        detect \
        binary \
        "" \
        "${model}" \
        "${batch}" \
        "${DETECT_EPOCHS}" \
        "${seed}" \
        "${DETECT_IMGSZ}" \
        AdamW \
        30 \
        0.01 \
        20 \
        "${lr0}"
}

submit_detect_coarse() {
    local name="$1"
    local model="$2"
    local batch="$3"
    local lr0="$4"
    local seed="$5"
    local label_plan="$6"
    local checkpoint_tag="$7"
    submit_job \
        "${name}" \
        "${WANDB_PROJECT_DET_COARSE}" \
        "${WANDB_GROUP_PREFIX}_det_${label_plan}" \
        "mbari,detection,coarse,${label_plan},nibi,suite_${SUITE_TAG},${checkpoint_tag}" \
        detect \
        coarse \
        "${label_plan}" \
        "${model}" \
        "${batch}" \
        "${DETECT_EPOCHS}" \
        "${seed}" \
        "${DETECT_IMGSZ}" \
        auto \
        20 \
        0.01 \
        -1 \
        "${lr0}"
}

submit_segment_binary() {
    local name="$1"
    local model="$2"
    local batch="$3"
    local lr0="$4"
    local seed="$5"
    submit_job \
        "${name}" \
        "${WANDB_PROJECT_SEG_BIN}" \
        "${WANDB_GROUP_PREFIX}_seg_bin" \
        "mbari,segmentation,binary,nibi,suite_${SUITE_TAG}" \
        segment \
        binary \
        "" \
        "${model}" \
        "${batch}" \
        "${SEG_EPOCHS}" \
        "${seed}" \
        "${SEG_IMGSZ}" \
        auto \
        30 \
        0.01 \
        20 \
        "${lr0}"
}

submit_segment_coarse() {
    local name="$1"
    local model="$2"
    local batch="$3"
    local lr0="$4"
    local seed="$5"
    local label_plan="$6"
    submit_job \
        "${name}" \
        "${WANDB_PROJECT_SEG_COARSE}" \
        "${WANDB_GROUP_PREFIX}_seg_${label_plan}" \
        "mbari,segmentation,coarse,${label_plan},nibi,suite_${SUITE_TAG}" \
        segment \
        coarse \
        "${label_plan}" \
        "${model}" \
        "${batch}" \
        "${SEG_EPOCHS}" \
        "${seed}" \
        "${SEG_IMGSZ}" \
        auto \
        20 \
        0.01 \
        -1 \
        "${lr0}"
}

if [[ "${RUN_BINARY_DETECT}" == "1" ]]; then
    [[ -n "${MEGALODON_PT}" ]] && submit_detect_binary \
        "det_bin_megalodon_b8_lr5e4_s42" "${MEGALODON_PT}" 8 0.0005 42 "checkpoint_megalodon"
    [[ -n "${MBARI315K_PT}" ]] && submit_detect_binary \
        "det_bin_mbari315k_b8_lr5e4_s42" "${MBARI315K_PT}" 8 0.0005 42 "checkpoint_mbari315k"
    submit_detect_binary "det_bin_yolo11m_b16_lr1e3_s42" "yolo11m.pt" 16 0.001 42 "checkpoint_yolo11m"
    submit_detect_binary "det_bin_yolo11l_b12_lr5e4_s42" "yolo11l.pt" 12 0.0005 42 "checkpoint_yolo11l"
fi

if [[ "${RUN_COARSE_DETECT}" == "1" ]]; then
    [[ -n "${MBARI315K_PT}" ]] && submit_detect_coarse \
        "det_coarse8_mbari315k_b8_lr5e4_s0" "${MBARI315K_PT}" 8 0.0005 0 "coarse_v1_bio8" "checkpoint_mbari315k"
    submit_detect_coarse \
        "det_coarse8_yolo11m_b16_lr1e3_s0" "yolo11m.pt" 16 0.001 0 "coarse_v1_bio8" "checkpoint_yolo11m"
    submit_detect_coarse \
        "det_coarse8_yolo11l_b12_lr5e4_s0" "yolo11l.pt" 12 0.0005 0 "coarse_v1_bio8" "checkpoint_yolo11l"

    if [[ "${INCLUDE_GEAR_VARIANT}" == "1" ]]; then
        [[ -n "${MBARI315K_PT}" ]] && submit_detect_coarse \
            "det_coarse8g_mbari315k_b8_lr5e4_s0" "${MBARI315K_PT}" 8 0.0005 0 "coarse_v1_bio8_plus_gear" "checkpoint_mbari315k"
        submit_detect_coarse \
            "det_coarse8g_yolo11m_b16_lr1e3_s0" "yolo11m.pt" 16 0.001 0 "coarse_v1_bio8_plus_gear" "checkpoint_yolo11m"
    fi

    if [[ "${INCLUDE_MIDWATER}" == "1" && -n "${MIDWATER_PT}" ]]; then
        submit_detect_coarse \
            "det_coarse8_midwater_b8_lr5e4_s0" "${MIDWATER_PT}" 8 0.0005 0 "coarse_v1_bio8" "checkpoint_midwater2025"
    fi
fi

if [[ "${RUN_BINARY_SEGMENT}" == "1" ]]; then
    submit_segment_binary "seg_bin_yolo11m_b8_lr1e3_s42" "yolo11m-seg.pt" 8 0.001 42
    submit_segment_binary "seg_bin_yolo11l_b8_lr5e4_s42" "yolo11l-seg.pt" 8 0.0005 42
fi

if [[ "${RUN_COARSE_SEGMENT}" == "1" ]]; then
    submit_segment_coarse "seg_coarse8_yolo11m_b8_lr1e3_s0" "yolo11m-seg.pt" 8 0.001 0 "coarse_v1_bio8"
    submit_segment_coarse "seg_coarse8_yolo11l_b8_lr5e4_s0" "yolo11l-seg.pt" 8 0.0005 0 "coarse_v1_bio8"

    if [[ "${INCLUDE_GEAR_VARIANT}" == "1" ]]; then
        submit_segment_coarse "seg_coarse8g_yolo11m_b8_lr1e3_s0" "yolo11m-seg.pt" 8 0.001 0 "coarse_v1_bio8_plus_gear"
    fi
fi

echo "============================================"
echo "Suite complete."
echo "Submitted jobs: ${submitted_jobs}"
echo "Skipped jobs: ${skipped_jobs}"
echo "Manifest: ${MANIFEST_PATH}"
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Mode: dry-run only (no jobs submitted)"
fi
echo "============================================"
