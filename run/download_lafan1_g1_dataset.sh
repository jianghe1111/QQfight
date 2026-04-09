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

DEST_DIR="${DEST_DIR:-${REPO_ROOT}/data/lafan1_retargeting/hf_repo}"
ALLOW_PATTERNS="${ALLOW_PATTERNS:-g1/*}"
HF_DATASET_ID="${HF_DATASET_ID:-lvhaidong/LAFAN1_Retargeting_Dataset}"
export DEST_DIR
export ALLOW_PATTERNS
export HF_DATASET_ID

echo "[INFO] Downloading dataset subset from Hugging Face"
echo "[INFO]   dataset_id      : ${HF_DATASET_ID}"
echo "[INFO]   allow_patterns  : ${ALLOW_PATTERNS}"
echo "[INFO]   dest_dir        : ${DEST_DIR}"

"${PYTHON_BIN}" - <<'PY'
import os
import sys

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:
    print("[ERROR] huggingface_hub is required. Install it with: pip install huggingface_hub", file=sys.stderr)
    raise SystemExit(1) from exc

snapshot_download(
    repo_id=os.environ["HF_DATASET_ID"],
    repo_type="dataset",
    allow_patterns=os.environ["ALLOW_PATTERNS"],
    local_dir=os.environ["DEST_DIR"],
    max_workers=4,
)
PY

FILE_COUNT=$(find "${DEST_DIR}/g1" -maxdepth 1 -type f 2>/dev/null | wc -l | tr -d ' ')
echo "[INFO] Download complete. Files in ${DEST_DIR}/g1: ${FILE_COUNT}"
