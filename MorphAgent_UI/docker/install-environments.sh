#!/usr/bin/env bash
# Build-time-only installation. Do not reuse the host setup scripts here.
set -euo pipefail

UI_ENV="${MORPHAGENT_ENV_NAME:-morphagent}"
SANDBOX_ENV="${MORPHAGENT_SANDBOX_ENV_NAME:-morphagent_sandbox}"
ALLEN_ENV="${MORPHAGENT_ALLEN_ENV_NAME:-morphagent_allen}"
INSTALL_ALLEN="${INSTALL_ALLEN:-1}"
INSTALL_COMPONENT="${INSTALL_COMPONENT:-all}"
ROOT="/opt/MorphAgent_UI"

case "${INSTALL_ALLEN}" in
  0|1) ;;
  *)
    echo "ERROR: INSTALL_ALLEN must be 0 or 1, got: ${INSTALL_ALLEN}" >&2
    exit 2
    ;;
esac

case "${INSTALL_COMPONENT}" in
  all|ui|sandbox|allen) ;;
  *)
    echo "ERROR: INSTALL_COMPONENT must be all, ui, sandbox, or allen; got: ${INSTALL_COMPONENT}" >&2
    exit 2
    ;;
esac

install_ui() {
  echo "[Docker build] creating UI environment: ${UI_ENV}"
  conda create -y -n "${UI_ENV}" -c conda-forge --strict-channel-priority \
    "python=3.10" "pip>=24" setuptools wheel \
    "pyqt=5" pyqt5-sip "qtpy>=2.4"
  conda run --no-capture-output -n "${UI_ENV}" \
    python -m pip install --no-cache-dir -r "${ROOT}/dependencies/requirements-demo-ui.txt"
}

install_sandbox() {
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
}

install_allen() {
  if [[ "${INSTALL_ALLEN}" != "1" ]]; then
    echo "[Docker build] Allen environment disabled by INSTALL_ALLEN=0"
    return
  fi

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
}

case "${INSTALL_COMPONENT}" in
  ui) install_ui ;;
  sandbox) install_sandbox ;;
  allen) install_allen ;;
  all)
    install_ui
    install_sandbox
    install_allen
    ;;
esac

# Remove package archives and indexes in the same Docker layer that created
# them. Deleting them in a later layer would not reduce the final image size.
conda clean -afy
