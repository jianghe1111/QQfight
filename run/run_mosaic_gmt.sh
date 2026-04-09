#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN=python
    else
        PYTHON_BIN=python3
    fi
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[ERROR] Python launcher not found: ${PYTHON_BIN}" >&2
    exit 1
fi

detect_gpu_count() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        local visible="${CUDA_VISIBLE_DEVICES// /}"
        if [[ -z "${visible}" || "${visible}" == "-1" ]]; then
            echo 0
            return
        fi
        IFS=',' read -r -a gpu_ids <<< "${visible}"
        echo "${#gpu_ids[@]}"
        return
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index --format=csv,noheader | wc -l
    else
        echo 0
    fi
}

GPU_COUNT=$(detect_gpu_count)
if [[ "${GPU_COUNT}" -lt 1 ]]; then
    echo "[ERROR] No visible NVIDIA GPU found. Set CUDA_VISIBLE_DEVICES or run on a GPU machine." >&2
    exit 1
fi

TASK="${TASK:-General-Tracking-Flat-G1-Wo-State-Estimation-v0-World-Coordinate-Reward}"
MOTION_PATH="${MOTION_PATH:-${REPO_ROOT}/data/lafan1_retargeting/npz/g1}"
LOG_PROJECT_NAME="${LOG_PROJECT_NAME:-GMT_MOSAIC_RL}"
RUN_NAME="${RUN_NAME:-GMT_MOSAIC_GMT}"
HEADLESS_FLAG="${HEADLESS_FLAG:---headless}"
LOGGER="${LOGGER:-wandb}"

if [[ ! -d "${MOTION_PATH}" && ! -f "${MOTION_PATH}" ]]; then
    echo "[ERROR] Motion path does not exist: ${MOTION_PATH}" >&2
    exit 1
fi

if [[ "${GPU_COUNT}" -eq 1 ]]; then
    NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
    NUM_ENVS="${NUM_ENVS:-19000}"
else
    NPROC_PER_NODE="${NPROC_PER_NODE:-${GPU_COUNT}}"
    NUM_ENVS_PER_GPU="${NUM_ENVS_PER_GPU:-3000}"
    NUM_ENVS="${NUM_ENVS:-$((NPROC_PER_NODE * NUM_ENVS_PER_GPU))}"
fi

if [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
    echo "[ERROR] NPROC_PER_NODE must be >= 1, got ${NPROC_PER_NODE}" >&2
    exit 1
fi

if [[ "${NPROC_PER_NODE}" -gt "${GPU_COUNT}" ]]; then
    echo "[ERROR] NPROC_PER_NODE=${NPROC_PER_NODE} exceeds visible GPU count ${GPU_COUNT}" >&2
    exit 1
fi

echo "[INFO] MOSAIC GMT launch settings"
echo "[INFO]   python          : ${PYTHON_BIN}"
echo "[INFO]   visible_gpus    : ${GPU_COUNT}"
echo "[INFO]   nproc_per_node  : ${NPROC_PER_NODE}"
echo "[INFO]   num_envs        : ${NUM_ENVS}"
echo "[INFO]   task            : ${TASK}"
echo "[INFO]   motion_path     : ${MOTION_PATH}"
echo "[INFO]   log_project     : ${LOG_PROJECT_NAME}"
echo "[INFO]   run_name        : ${RUN_NAME}"

TRAIN_ARGS=(
    scripts/rsl_rl/train.py
    --task="${TASK}"
    --num_envs="${NUM_ENVS}"
    --motion "${MOTION_PATH}"
    --logger "${LOGGER}"
    --log_project_name "${LOG_PROJECT_NAME}"
    --run_name "${RUN_NAME}"
)

if [[ -n "${HEADLESS_FLAG}" ]]; then
    TRAIN_ARGS+=("${HEADLESS_FLAG}")
fi

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    HYDRA_FULL_ERROR=1 "${PYTHON_BIN}" -m torch.distributed.run \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="${NPROC_PER_NODE}" \
        "${TRAIN_ARGS[@]}" \
        --distributed \
        "$@"
else
    HYDRA_FULL_ERROR=1 "${PYTHON_BIN}" "${TRAIN_ARGS[@]}" "$@"
fi
