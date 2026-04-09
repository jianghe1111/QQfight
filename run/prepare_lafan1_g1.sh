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

INPUT_DIR="${INPUT_DIR:-${REPO_ROOT}/data/lafan1_retargeting/hf_repo/g1}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/data/lafan1_retargeting/npz/g1}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-lafan1_g1}"
INPUT_FPS="${INPUT_FPS:-30}"
HEADLESS_FLAG="${HEADLESS_FLAG:---headless}"

if [[ ! -d "${INPUT_DIR}" ]]; then
    echo "[ERROR] Input CSV directory does not exist: ${INPUT_DIR}" >&2
    exit 1
fi

if ! "${PYTHON_BIN}" - <<'PY'
import importlib.util
import sys

required = ("isaaclab", "whole_body_tracking")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print(
        "[ERROR] Missing required Python packages: "
        + ", ".join(missing)
        + ". Activate your Isaac Lab environment and install "
        + "'source/whole_body_tracking' in editable mode first.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
then
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "[INFO] Preparing LAFAN1 G1 motions"
echo "[INFO]   input_dir      : ${INPUT_DIR}"
echo "[INFO]   output_dir     : ${OUTPUT_DIR}"
echo "[INFO]   output_prefix  : ${OUTPUT_PREFIX}"
echo "[INFO]   input_fps      : ${INPUT_FPS}"

ARGS=(
    scripts/batch_csv_to_npz.py
    --input_dir "${INPUT_DIR}"
    --input_fps "${INPUT_FPS}"
    --output_prefix "${OUTPUT_PREFIX}"
    --output_dir "${OUTPUT_DIR}"
)

if [[ -n "${HEADLESS_FLAG}" ]]; then
    ARGS+=("${HEADLESS_FLAG}")
fi

"${PYTHON_BIN}" "${ARGS[@]}" "$@"
