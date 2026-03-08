#!/bin/bash
# Resume one or more timed-out MBARI training runs from existing last.pt checkpoints.
#
# Examples:
#   bash slurm/resume_mbari_runs.sh \
#     --run-dir /scratch/$USER/yolo_seg/2026-03-07/detect_coarse_12345/det_coarse8_mbari315k_b8_lr5e4_s0 \
#     --time 48:00:00 \
#     --dry-run
#
#   bash slurm/resume_mbari_runs.sh \
#     --last-pt /scratch/$USER/yolo_seg/2026-03-07/detect_binary_12346/det_bin_megalodon_b8_lr5e4_s42/weights/last.pt \
#     --data /project/rpp-kmoran/merileo/data/mbari_raw.tar.gz

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
TRAIN_SCRIPT="${REPO_DIR}/slurm/train.sh"

usage() {
    cat <<'EOF'
Usage:
  bash slurm/resume_mbari_runs.sh [options]

Run selectors:
  --run-dir DIR               Existing train run dir containing weights/last.pt
  --output-dir DIR            Existing job output dir containing run_config.yaml and a single weights/last.pt
  --last-pt FILE              Existing checkpoint path

Path overrides:
  --repo DIR                  Repo root containing slurm/train.sh
  --venv DIR                  Python virtualenv path override
  --data FILE                 Data tarball override
  --label-plans-file FILE     Coarse label plan YAML override

W&B options:
  --wandb-run-id ID           Override detected W&B run ID (single-run use only)
  --wandb-resume MODE         W&B resume mode (default: allow)
  --allow-missing-wandb-id    Resume even if the original W&B run ID cannot be detected

Resources:
  --gpus-per-node SPEC        Default: h100:1
  --cpus-per-task N           Default: 6
  --mem SIZE                  Default: 32000M
  --time HH:MM:SS             Default: 24:00:00

Behavior:
  --dry-run                   Print sbatch commands without submitting

Notes:
  - Full-state resume requires weights/last.pt, not best.pt.
  - Ultralytics resume continues toward the original total epoch target saved in last.pt.
EOF
}

RUN_DIRS=()
OUTPUT_DIRS=()
LAST_PTS=()

VENV_PATH=""
DATA_TARBALL=""
LABEL_PLANS_FILE=""
REPO_OVERRIDE_SET=0
LABEL_PLANS_OVERRIDE_SET=0
WANDB_RUN_ID_OVERRIDE=""
WANDB_RESUME_MODE="${WANDB_RESUME_MODE:-allow}"
ALLOW_MISSING_WANDB_ID=0
DRY_RUN="${DRY_RUN:-0}"

GPUS_PER_NODE="${GPUS_PER_NODE:-h100:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-6}"
MEM="${MEM:-32000M}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir)
            RUN_DIRS+=("$2")
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIRS+=("$2")
            shift 2
            ;;
        --last-pt)
            LAST_PTS+=("$2")
            shift 2
            ;;
        --repo)
            REPO_DIR="$2"
            TRAIN_SCRIPT="${REPO_DIR}/slurm/train.sh"
            REPO_OVERRIDE_SET=1
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
        --label-plans-file)
            LABEL_PLANS_FILE="$2"
            LABEL_PLANS_OVERRIDE_SET=1
            shift 2
            ;;
        --wandb-run-id)
            WANDB_RUN_ID_OVERRIDE="$2"
            shift 2
            ;;
        --wandb-resume)
            WANDB_RESUME_MODE="$2"
            shift 2
            ;;
        --allow-missing-wandb-id)
            ALLOW_MISSING_WANDB_ID=1
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
        --dry-run)
            DRY_RUN=1
            shift
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

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "ERROR: train script not found: ${TRAIN_SCRIPT}" >&2
    exit 1
fi

if [[ ${#RUN_DIRS[@]} -eq 0 && ${#OUTPUT_DIRS[@]} -eq 0 && ${#LAST_PTS[@]} -eq 0 ]]; then
    echo "ERROR: provide at least one of --run-dir, --output-dir, or --last-pt" >&2
    usage >&2
    exit 1
fi

total_specs=$(( ${#RUN_DIRS[@]} + ${#OUTPUT_DIRS[@]} + ${#LAST_PTS[@]} ))
if [[ -n "${WANDB_RUN_ID_OVERRIDE}" && "${total_specs}" -ne 1 ]]; then
    echo "ERROR: --wandb-run-id can only be used when resuming exactly one run" >&2
    exit 1
fi

yaml_get() {
    local key="$1"
    local file="$2"
    sed -n "s/^${key}: //p" "${file}" | head -n 1
}

strip_outer_quotes() {
    local value="$1"
    value="${value%\"}"
    value="${value#\"}"
    printf '%s' "${value}"
}

json_get() {
    local key="$1"
    local file="$2"
    sed -n "s/.*\"${key}\": \"\\([^\"]*\\)\".*/\\1/p" "${file}" | head -n 1
}

extract_wandb_id_from_dirname() {
    local name="$1"
    sed -n 's/^.*-\([A-Za-z0-9]\{6,\}\)$/\1/p' <<<"${name}" | head -n 1
}

find_wandb_run_id() {
    local output_dir="$1"
    local train_dir="$2"
    local meta_file="${train_dir}/wandb_run.json"
    if [[ -f "${meta_file}" ]]; then
        json_get id "${meta_file}"
        return 0
    fi

    local runtime_root="${output_dir}/runtime"
    local wandb_root="${runtime_root}/wandb"
    local candidate=""
    local latest_link=""

    latest_link="$(find "${runtime_root}" -type l -name latest-run 2>/dev/null | sort | tail -n 1 || true)"
    if [[ -n "${latest_link}" && -e "${latest_link}" ]]; then
        candidate="$(basename "$(readlink -f "${latest_link}")")"
    elif [[ -d "${wandb_root}" || -d "${runtime_root}" ]]; then
        candidate="$(
            find "${runtime_root}" -type d \( -name 'run-*' -o -name 'offline-run-*' \) 2>/dev/null \
                | sort \
                | tail -n 1 \
                | xargs -r basename
        )"
    fi

    if [[ -n "${candidate}" ]]; then
        extract_wandb_id_from_dirname "${candidate}"
    fi
}

append_extra_args() {
    local raw="$1"
    local -n out_ref=$2
    if [[ -z "${raw}" ]]; then
        return 0
    fi

    while IFS= read -r arg; do
        [[ -n "${arg}" ]] && out_ref+=("${arg}")
    done < <(
        python3 - <<'PY' "${raw}"
import shlex
import sys

for token in shlex.split(sys.argv[1]):
    print(token)
PY
    )
}

resolve_from_output_dir() {
    local output_dir="$1"
    if [[ ! -d "${output_dir}" ]]; then
        echo "ERROR: output dir not found: ${output_dir}" >&2
        return 1
    fi
    local checkpoint
    checkpoint="$(find "${output_dir}" -maxdepth 3 -type f -path '*/weights/last.pt' | sort | head -n 1)"
    if [[ -z "${checkpoint}" ]]; then
        echo "ERROR: could not find weights/last.pt under ${output_dir}" >&2
        return 1
    fi
    printf '%s\n%s\n%s\n' "$(dirname "$(dirname "${checkpoint}")")" "${output_dir}" "${checkpoint}"
}

resolve_from_run_dir() {
    local train_dir="$1"
    local checkpoint="${train_dir}/weights/last.pt"
    if [[ ! -f "${checkpoint}" ]]; then
        echo "ERROR: last checkpoint not found: ${checkpoint}" >&2
        return 1
    fi
    printf '%s\n%s\n%s\n' "${train_dir}" "$(dirname "${train_dir}")" "${checkpoint}"
}

resolve_from_last_pt() {
    local checkpoint="$1"
    if [[ ! -f "${checkpoint}" ]]; then
        echo "ERROR: checkpoint not found: ${checkpoint}" >&2
        return 1
    fi
    local train_dir
    train_dir="$(dirname "$(dirname "${checkpoint}")")"
    printf '%s\n%s\n%s\n' "${train_dir}" "$(dirname "${train_dir}")" "${checkpoint}"
}

submit_resume() {
    local train_dir="$1"
    local output_dir="$2"
    local checkpoint="$3"

    local config_file="${output_dir}/run_config.yaml"
    if [[ ! -f "${config_file}" ]]; then
        echo "ERROR: run config not found: ${config_file}" >&2
        return 1
    fi

    local repo_dir="${REPO_DIR}"
    if [[ "${REPO_OVERRIDE_SET}" != "1" ]]; then
        local config_repo_dir
        config_repo_dir="$(strip_outer_quotes "$(yaml_get repo_dir "${config_file}")")"
        if [[ -n "${config_repo_dir}" ]]; then
            repo_dir="${config_repo_dir}"
        fi
    fi
    local venv_path="${VENV_PATH:-$(strip_outer_quotes "$(yaml_get venv_path "${config_file}")")}"
    local data_tarball="${DATA_TARBALL:-$(strip_outer_quotes "$(yaml_get data_tarball "${config_file}")")}"
    local task="$(strip_outer_quotes "$(yaml_get task "${config_file}")")"
    local mode="$(strip_outer_quotes "$(yaml_get mode "${config_file}")")"
    local label_plan="$(strip_outer_quotes "$(yaml_get label_plan "${config_file}")")"
    local label_plans_file=""
    if [[ "${LABEL_PLANS_OVERRIDE_SET}" == "1" ]]; then
        label_plans_file="${LABEL_PLANS_FILE}"
    else
        label_plans_file="$(strip_outer_quotes "$(yaml_get label_plans_file "${config_file}")")"
    fi
    local top_n="$(strip_outer_quotes "$(yaml_get top_n "${config_file}")")"
    local val_ratio="$(strip_outer_quotes "$(yaml_get val_ratio "${config_file}")")"
    local split_seed="$(strip_outer_quotes "$(yaml_get split_seed "${config_file}")")"
    local model="$(strip_outer_quotes "$(yaml_get model "${config_file}")")"
    local epochs="$(strip_outer_quotes "$(yaml_get epochs "${config_file}")")"
    local batch="$(strip_outer_quotes "$(yaml_get batch "${config_file}")")"
    local wandb_project="$(strip_outer_quotes "$(yaml_get wandb_project "${config_file}")")"
    local wandb_name="$(strip_outer_quotes "$(yaml_get wandb_run_name "${config_file}")")"
    local wandb_group="$(strip_outer_quotes "$(yaml_get wandb_group "${config_file}")")"
    local wandb_tags="$(strip_outer_quotes "$(yaml_get wandb_tags "${config_file}")")"
    local extra_args="$(strip_outer_quotes "$(yaml_get extra_args "${config_file}")")"

    if [[ -z "${venv_path}" ]]; then
        echo "ERROR: venv path missing in ${config_file}; pass --venv explicitly" >&2
        return 1
    fi
    if [[ -z "${data_tarball}" ]]; then
        echo "ERROR: data tarball missing in ${config_file}; pass --data explicitly" >&2
        return 1
    fi
    if [[ -z "${label_plans_file}" ]]; then
        label_plans_file="${repo_dir}/configs/label_plans.yaml"
    fi

    local wandb_run_id="${WANDB_RUN_ID_OVERRIDE}"
    if [[ -z "${wandb_run_id}" ]]; then
        wandb_run_id="$(find_wandb_run_id "${output_dir}" "${train_dir}")"
    fi

    if [[ -n "${wandb_project}" && -z "${wandb_run_id}" && "${ALLOW_MISSING_WANDB_ID}" != "1" ]]; then
        echo "ERROR: could not detect W&B run ID for ${train_dir}" >&2
        echo "       Looked in ${train_dir}/wandb_run.json and ${output_dir}/runtime/wandb" >&2
        echo "       Pass --wandb-run-id <id> or rerun with --allow-missing-wandb-id" >&2
        return 1
    fi

    local -a cmd=(
        sbatch
        --parsable
        "--gpus-per-node=${GPUS_PER_NODE}"
        "--cpus-per-task=${CPUS_PER_TASK}"
        "--mem=${MEM}"
        "--time=${TIME_LIMIT}"
        "${TRAIN_SCRIPT}"
        --repo "${repo_dir}"
        --venv "${venv_path}"
        --data "${data_tarball}"
        --task "${task}"
        --mode "${mode}"
        --top_n "${top_n:-100}"
        --val_ratio "${val_ratio:-0.2}"
        --split-seed "${split_seed:-42}"
        --model "${model}"
        --epochs "${epochs}"
        --batch "${batch}"
        --resume "${checkpoint}"
    )

    if [[ -n "${label_plan}" && "${label_plan}" != "<none>" ]]; then
        cmd+=(--label-plan "${label_plan}" --label-plans-file "${label_plans_file}")
    fi
    if [[ -n "${wandb_project}" ]]; then
        cmd+=(--wandb-project "${wandb_project}")
    fi
    if [[ -n "${wandb_name}" ]]; then
        cmd+=(--wandb-name "${wandb_name}")
    fi
    if [[ -n "${wandb_group}" ]]; then
        cmd+=(--wandb-group "${wandb_group}")
    fi
    if [[ -n "${wandb_tags}" ]]; then
        cmd+=(--wandb-tags "${wandb_tags}")
    fi
    if [[ -n "${wandb_run_id}" ]]; then
        cmd+=(--wandb-run-id "${wandb_run_id}" --wandb-resume "${WANDB_RESUME_MODE}")
    fi

    append_extra_args "${extra_args}" cmd

    local cmd_str
    cmd_str="$(printf '%q ' "${cmd[@]}")"
    echo "--------------------------------------------"
    echo "Run dir: ${train_dir}"
    echo "Output dir: ${output_dir}"
    echo "Resume checkpoint: ${checkpoint}"
    if [[ -n "${wandb_run_id}" ]]; then
        echo "W&B run ID: ${wandb_run_id} (resume=${WANDB_RESUME_MODE})"
    else
        echo "W&B run ID: <not found>"
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "DRY RUN: ${cmd_str}"
        return 0
    fi

    local job_id
    job_id="$("${cmd[@]}")"
    echo "SUBMITTED -> job ${job_id}"
}

for run_dir in "${RUN_DIRS[@]}"; do
    mapfile -t resolved < <(resolve_from_run_dir "$(readlink -f "${run_dir}")")
    submit_resume "${resolved[0]}" "${resolved[1]}" "${resolved[2]}"
done

for output_dir in "${OUTPUT_DIRS[@]}"; do
    mapfile -t resolved < <(resolve_from_output_dir "$(readlink -f "${output_dir}")")
    submit_resume "${resolved[0]}" "${resolved[1]}" "${resolved[2]}"
done

for checkpoint in "${LAST_PTS[@]}"; do
    mapfile -t resolved < <(resolve_from_last_pt "$(readlink -f "${checkpoint}")")
    submit_resume "${resolved[0]}" "${resolved[1]}" "${resolved[2]}"
done
