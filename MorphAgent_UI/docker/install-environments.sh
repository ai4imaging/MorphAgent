#!/usr/bin/env bash
# Build-time-only installation. Do not reuse the host setup scripts here.
set -euo pipefail

UI_ENV="${MORPHAGENT_ENV_NAME:-morphagent}"
SANDBOX_ENV="${MORPHAGENT_SANDBOX_ENV_NAME:-morphagent_sandbox}"
ALLEN_ENV="${MORPHAGENT_ALLEN_ENV_NAME:-morphagent_allen}"
INSTALL_ALLEN="${INSTALL_ALLEN:-1}"
ROOT="/opt/MorphAgent_UI"

case "${INSTALL_ALLEN}" in
  0|1) ;;
  *)
    echo "ERROR: INSTALL_ALLEN must be 0 or 1, got: ${INSTALL_ALLEN}" >&2
    exit 2
    ;;
esac

echo "[Docker build] creating UI environment: ${UI_ENV}"
conda create -y -n "${UI_ENV}" -c conda-forge --strict-channel-priority \
  "python=3.10" "pip>=24" setuptools wheel \
  "pyqt=5" pyqt5-sip "qtpy>=2.4"
conda run --no-capture-output -n "${UI_ENV}" \
  python -m pip install --no-cache-dir -r "${ROOT}/dependencies/requirements-demo-ui.txt"

echo "[Docker build] creating generated-code sandbox: ${SANDBOX_ENV}"
conda create -y -n "${SANDBOX_ENV}" -c conda-forge --strict-channel-priority \
  "python=3.10" "pip>=24" setuptools wheel
conda run --no-capture-output -n "${SANDBOX_ENV}" \
  python -m pip install --no-cache-dir -r "${ROOT}/dependencies/requirements-sandbox.txt"
conda run --no-capture-output -n "${SANDBOX_ENV}" python - <<'PY'
import cv2
import mahotas
import numpy
import pandas
import scipy
import skimage
import tifffile

print(
    "[Docker build] sandbox ready:",
    "numpy", numpy.__version__,
    "skimage", skimage.__version__,
)
PY

if [[ "${INSTALL_ALLEN}" == "1" ]]; then
  echo "[Docker build] creating Allen environment: ${ALLEN_ENV}"
  # `conda create -y` is supported by old Conda releases. This deliberately
  # avoids the separate `conda env create` parser that caused the original
  # cross-machine `unrecognized arguments: -y` failure.
  conda create -y -n "${ALLEN_ENV}" -c conda-forge --strict-channel-priority \
    "python=3.6" pip
  conda run --no-capture-output -n "${ALLEN_ENV}" \
    python -m pip install --upgrade 'pip<22' 'setuptools<59' wheel
  conda run --no-capture-output -n "${ALLEN_ENV}" \
    python -m pip install --no-cache-dir \
      -r "${ROOT}/MorphAgent/envs/requirements-allen.txt"
  conda run --no-capture-output -n "${ALLEN_ENV}" \
    python -m pip install --no-cache-dir --no-deps \
      "${ROOT}/MorphAgent/segmentation_allen"
  conda run --no-capture-output -n "${ALLEN_ENV}" \
    python "${ROOT}/MorphAgent/segmentation_allen/check_installation.py"
else
  echo "[Docker build] Allen environment disabled by INSTALL_ALLEN=0"
fi
