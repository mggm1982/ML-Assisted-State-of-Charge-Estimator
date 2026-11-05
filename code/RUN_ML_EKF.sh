#!/usr/bin/env bash
set -euo pipefail
CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$CODE_DIR")"
if [ "$#" -gt 1 ]; then
  echo 'Usage: bash code/RUN_ML_EKF.sh [extracted-dataset-folder]'
  exit 2
fi
if [ -z "${PYTHON_BIN:-}" ]; then
  if [ ! -x "$CODE_DIR/.venv/bin/python" ]; then
    python3 -m venv "$CODE_DIR/.venv"
  fi
  PYTHON_BIN="$CODE_DIR/.venv/bin/python"
  "$PYTHON_BIN" -m pip install -r "$PROJECT_DIR/libraries/requirements.txt"
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR="$CODE_DIR/.mpl-cache"
export PYTHONDONTWRITEBYTECODE=1
if [ "$#" -eq 1 ]; then
  "$PYTHON_BIN" "$CODE_DIR/run_submission.py" --data "$1"
else
  "$PYTHON_BIN" "$CODE_DIR/run_submission.py"
fi
