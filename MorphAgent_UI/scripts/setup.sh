#!/usr/bin/env bash
# MorphAgent UI setup (macOS / Linux).
# Creates `morphagent` with Qt + scientific stack, then optional `morphagent_allen`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HANDOFF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPOSITORY="${HANDOFF_ROOT}/MorphAgent"
ENV_NAME="${MORPHAGENT_ENV_NAME:-morphagent}"
ALLEN_ENV_NAME="${MORPHAGENT_ALLEN_ENV_NAME:-morphagent_allen}"
ALLEN_YML="${REPOSITORY}/envs/environment_allen.yml"
ALLEN_YML_ALT="${HANDOFF_ROOT}/dependencies/environment-allen-optional.yml"
ALLEN_REQ="${REPOSITORY}/envs/requirements-allen.txt"
REQ_FILE="${HANDOFF_ROOT}/dependencies/requirements-demo-ui.txt"
INSTALL_ALLEN="${MORPHAGENT_INSTALL_ALLEN:-1}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda was not found. Install Miniforge/Miniconda/Anaconda first." >&2
  exit 1
fi

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "ERROR: missing requirements file: ${REQ_FILE}" >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

env_exists() {
  conda env list | awk '{print $1}' | grep -Fxq "$1"
}

# Core GUI + numeric stack via conda-forge (reliable Qt binaries; pip pins follow).
install_core_conda_packages() {
  local env="$1"
  echo "Installing core conda packages into ${env} (PyQt5 / numpy / scipy / …)…"
  conda install -y -n "${env}" -c conda-forge \
    "python=3.10" \
    "pip>=24" \
    "numpy>=1.26,<3" \
    "scipy>=1.11" \
    "pandas>=2.0" \
    "pyqt=5" \
    "qtpy>=2.4" \
    "pillow>=10" \
    "matplotlib>=3.8" \
    "scikit-image>=0.22" \
    "scikit-learn>=1.3" \
    "tifffile>=2023" \
    "pyyaml>=6" \
    "tqdm" \
    "h5py" \
    "networkx" \
    "imageio" \
    "requests" \
    "lxml"
}

if env_exists "${ENV_NAME}"; then
  echo "Using existing conda environment: ${ENV_NAME}"
else
  echo "Creating conda environment: ${ENV_NAME}"
  conda create -y -n "${ENV_NAME}" -c conda-forge python=3.10 pip
fi

install_core_conda_packages "${ENV_NAME}"

echo "Installing pip packages from requirements-demo-ui.txt…"
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install -r "${REQ_FILE}"
conda run -n "${ENV_NAME}" python -m pip install -e "${REPOSITORY}"

# Ensure PyQt is importable even if pip/conda competed on the binding name.
if ! conda run -n "${ENV_NAME}" python -c "import PyQt5, qtpy, numpy" >/dev/null 2>&1; then
  echo "[WARN] PyQt5/numpy import failed after pip; reinstalling pyqt/numpy from conda-forge…" >&2
  install_core_conda_packages "${ENV_NAME}"
  conda run -n "${ENV_NAME}" python -c "import PyQt5, qtpy, numpy; print('[OK]', 'PyQt5', PyQt5.QtCore.PYQT_VERSION_STR, 'numpy', numpy.__version__)"
fi

if [[ ! -f "${REPOSITORY}/.env" ]]; then
  if [[ -f "${REPOSITORY}/.env.example" ]]; then
    cp "${REPOSITORY}/.env.example" "${REPOSITORY}/.env"
    chmod 600 "${REPOSITORY}/.env" || true
  else
    echo "[WARN] ${REPOSITORY}/.env.example missing; skipped creating .env" >&2
  fi
fi

# --- Allen classic segmentation (optional; soft-fail with WARNING) ------------
# Apple Silicon: Python 3.6 only via CONDA_SUBDIR=osx-64 (Rosetta).
# TIFF driver uses tifffile + aicssegmentation (no aicsimageio / aicspylibczi).
setup_allen_env() {
  local yml="${ALLEN_YML}"
  if [[ ! -f "${yml}" ]]; then
    yml="${ALLEN_YML_ALT}"
  fi
  if [[ ! -f "${yml}" ]]; then
    echo "[WARN] Allen environment.yml not found; skipping Allen setup" >&2
    return 1
  fi
  if [[ ! -f "${ALLEN_REQ}" ]]; then
    echo "[WARN] Allen requirements not found: ${ALLEN_REQ}; skipping Allen setup" >&2
    return 1
  fi

  local arch
  arch="$(uname -m)"

  if env_exists "${ALLEN_ENV_NAME}"; then
    echo "Using existing conda environment: ${ALLEN_ENV_NAME}"
  else
    echo "Creating Allen segmentation environment: ${ALLEN_ENV_NAME}"
    if [[ "${arch}" == "arm64" ]]; then
      echo "Apple Silicon detected: using CONDA_SUBDIR=osx-64 (Rosetta)"
      CONDA_SUBDIR=osx-64 conda env create -y -n "${ALLEN_ENV_NAME}" -f "${yml}"
    else
      conda env create -y -n "${ALLEN_ENV_NAME}" -f "${yml}"
    fi
  fi

  if [[ "${arch}" == "arm64" ]]; then
    conda activate "${ALLEN_ENV_NAME}"
    conda env config vars set CONDA_SUBDIR=osx-64 || true
    conda deactivate
  fi

  echo "Installing Allen scientific stack into ${ALLEN_ENV_NAME}..."
  conda run -n "${ALLEN_ENV_NAME}" python -m pip install --upgrade 'pip<22' 'setuptools<59' 'wheel'
  conda run -n "${ALLEN_ENV_NAME}" python -m pip install -r "${ALLEN_REQ}"
  echo "Installing vendored aicssegmentation (no-deps)..."
  conda run -n "${ALLEN_ENV_NAME}" python -m pip install -e "${REPOSITORY}/segmentation_allen" --no-deps
  echo "Verifying Allen installation..."
  conda run -n "${ALLEN_ENV_NAME}" python "${REPOSITORY}/segmentation_allen/check_installation.py"
  echo "[OK] Allen environment ${ALLEN_ENV_NAME} is ready"
}

if [[ "${INSTALL_ALLEN}" == "1" ]]; then
  set +e
  setup_allen_env
  allen_rc=$?
  set -e
  if [[ "${allen_rc}" -ne 0 ]]; then
    echo "[WARN] Allen environment setup failed (exit ${allen_rc})." >&2
    echo "[WARN] MorphAgent UI install continues. Custom data without masks will skip" >&2
    echo "[WARN] auto-segmentation and still finish the run (code/VLM without masks)." >&2
  fi
else
  echo "[INFO] Skipping Allen env (MORPHAGENT_INSTALL_ALLEN=0)."
fi

echo "Running install verification (UI smoke, offscreen)…"
QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}" \
  conda run -n "${ENV_NAME}" python "${HANDOFF_ROOT}/scripts/verify_install.py" --ui-smoke

echo
echo "Installation verified. Start MorphAgent with:"
echo "  MORPHAGENT_ENV_NAME=${ENV_NAME} bash scripts/start_ui.sh"
echo "Allen segmentation env (optional): ${ALLEN_ENV_NAME}"
