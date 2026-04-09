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

TASK="${TASK:-General-Tracking-Flat-G1-v0}"
MOTION_PATH="${MOTION_PATH:-${REPO_ROOT}/data/lafan1_retargeting/npz/g1}"
LOG_PROJECT_NAME="${LOG_PROJECT_NAME:-QQfight}"
RUN_NAME="${RUN_NAME:-lafan1_g1_general_tracking}"
LOGGER="${LOGGER:-wandb}"
HEADLESS_FLAG="${HEADLESS_FLAG:---headless}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

if [[ ! -d "${MOTION_PATH}" && ! -f "${MOTION_PATH}" ]]; then
    echo "[ERROR] Motion path does not exist: ${MOTION_PATH}" >&2
    exit 1
fi

if ! "${PYTHON_BIN}" - <<'PY'
import importlib.util
import sys

required = ("isaaclab", "whole_body_tracking", "rsl_rl")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print(
        "[ERROR] Missing required Python packages: "
        + ", ".join(missing)
        + ". Activate your Isaac Lab environment and install "
        + "'source/whole_body_tracking' and 'source/rsl_rl' in editable mode first.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
then
    exit 1
fi

TRAIN_ARGS=(
    scripts/rsl_rl/train.py
    --task="${TASK}"
    --motion "${MOTION_PATH}"
    --logger "${LOGGER}"
    --log_project_name "${LOG_PROJECT_NAME}"
    --run_name "${RUN_NAME}"
)

if [[ -n "${HEADLESS_FLAG}" ]]; then
    TRAIN_ARGS+=("${HEADLESS_FLAG}")
fi

if [[ -n "${NUM_ENVS:-}" ]]; then
    TRAIN_ARGS+=("--num_envs=${NUM_ENVS}")
fi

if [[ -n "${MAX_ITERATIONS:-}" ]]; then
    TRAIN_ARGS+=("--max_iterations=${MAX_ITERATIONS}")
fi

echo "[INFO] Launching official general training pipeline for LAFAN1 G1"
echo "[INFO]   python          : ${PYTHON_BIN}"
echo "[INFO]   nproc_per_node  : ${NPROC_PER_NODE}"
echo "[INFO]   task            : ${TASK}"
echo "[INFO]   motion_path     : ${MOTION_PATH}"
echo "[INFO]   log_project     : ${LOG_PROJECT_NAME}"
echo "[INFO]   run_name        : ${RUN_NAME}"

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
