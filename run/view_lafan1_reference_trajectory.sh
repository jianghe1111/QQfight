#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

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

VIS_ROOT="${VIS_ROOT:-${REPO_ROOT}/data/lafan1_retargeting/hf_visualizer}"
ROBOT_TYPE="${ROBOT_TYPE:-g1}"
FILE_NAME="${FILE_NAME:-walk1_subject1}"
CSV_DIR="${CSV_DIR:-${REPO_ROOT}/data/lafan1_retargeting/hf_repo/${ROBOT_TYPE}}"
CSV_LINK_NAME="${CSV_LINK_NAME:-${ROBOT_TYPE}}"

if [[ ! -f "${VIS_ROOT}/rerun_visualize.py" ]]; then
    echo "[ERROR] Missing dataset visualizer script: ${VIS_ROOT}/rerun_visualize.py" >&2
    exit 1
fi

if [[ ! -d "${CSV_DIR}" ]]; then
    echo "[ERROR] Missing CSV directory for robot '${ROBOT_TYPE}': ${CSV_DIR}" >&2
    exit 1
fi

if [[ ! -f "${CSV_DIR}/${FILE_NAME}.csv" ]]; then
    echo "[ERROR] Missing motion file: ${CSV_DIR}/${FILE_NAME}.csv" >&2
    exit 1
fi

if ! "${PYTHON_BIN}" - <<'PY'
import importlib.util
import sys

required = ("pinocchio", "rerun", "trimesh", "numpy")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print(
        "[ERROR] Missing visualization dependencies: "
        + ", ".join(missing)
        + ". Install the dataset viewer requirements first.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
then
    exit 1
fi

mkdir -p "${VIS_ROOT}"
cd "${VIS_ROOT}"

if [[ -L "${CSV_LINK_NAME}" ]]; then
    rm -f "${CSV_LINK_NAME}"
fi

if [[ ! -e "${CSV_LINK_NAME}" ]]; then
    ln -s "${CSV_DIR}" "${CSV_LINK_NAME}"
fi

echo "[INFO] Launching HF trajectory viewer"
echo "[INFO]   visualizer_root : ${VIS_ROOT}"
echo "[INFO]   robot_type      : ${ROBOT_TYPE}"
echo "[INFO]   file_name       : ${FILE_NAME}"
echo "[INFO]   csv_dir         : ${CSV_DIR}"

exec "${PYTHON_BIN}" rerun_visualize.py --file_name "${FILE_NAME}" --robot_type "${ROBOT_TYPE}"
