#!/usr/bin/env bash
# MorphAgent UI setup (macOS / Linux only — Windows uses setup_windows.ps1).
# Creates `morphagent` with scientific stack + pip PyQt5, then optional `morphagent_allen`.
#
# Qt (Unix): pip PyQt5 only. Do not mirror Windows' conda pyqt path here — on
# macOS, mixing conda `pyqt` with pip `PyQt5` loads two QtCore copies and aborts.
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

# Scientific stack via conda-forge (no pyqt — Qt comes from pip PyQt5 only).
install_core_conda_packages() {
  local env="$1"
  echo "Installing core conda packages into ${env} (numpy / scipy / …; Qt via pip)…"
  # Drop any previously mixed conda Qt stack before continuing.
  conda remove -y -n "${env}" --force pyqt pyqt5-sip qt-main 2>/dev/null || true
  conda install -y -n "${env}" -c conda-forge \
    "python=3.10" \
    "pip>=24" \
    "numpy>=1.26,<3" \
    "scipy>=1.11" \
    "pandas>=2.0" \
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

ensure_pip_pyqt() {
  local env="$1"
  echo "Ensuring a single pip PyQt5 stack in ${env}…"
  conda run -n "${env}" python -m pip uninstall -y PyQt5 PyQt5-Qt5 PyQt5-sip 2>/dev/null || true
  # Clean broken partial installs that leave no RECORD/METADATA.
  conda run -n "${env}" python - <<'PY'
from pathlib import Path
import site, shutil
for base in map(Path, site.getsitepackages()):
    for pattern in ("PyQt5", "PyQt5-*.dist-info", "PyQt5_sip*", "pyqt5_qt5*", "pyqt5_sip*"):
        for path in base.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.is_file():
                path.unlink(missing_ok=True)
print("[OK] cleared stale PyQt5 paths")
PY
  conda run -n "${env}" python -m pip install --ignore-installed --no-cache-dir "PyQt5==5.15.11" "qtpy==2.4.3"
  conda run -n "${env}" python - <<'PY'
from pathlib import Path
from PyQt5 import QtCore
import qtpy, numpy, PyQt5
print(f"[OK] pip PyQt5={QtCore.PYQT_VERSION_STR}, qtpy={qtpy.__version__}, numpy={numpy.__version__}")
print(f"[OK] QtCore={QtCore.__file__}")
# Conda Qt dylibs must not remain beside pip PyQt5.
lib = Path(PyQt5.__file__).resolve().parents[3] / "lib"
conda_qt = list(lib.glob("libQt5Core*")) if lib.is_dir() else []
if conda_qt:
    raise SystemExit(f"conda Qt dylibs still present: {conda_qt[:3]} — remove conda pyqt/qt-main")
PY
}

if env_exists "${ENV_NAME}"; then
  echo "Using existing conda environment: ${ENV_NAME}"
else
  echo "Creating conda environment: ${ENV_NAME}"
  conda create -y -n "${ENV_NAME}" -c conda-forge python=3.10 pip
fi

install_core_conda_packages "${ENV_NAME}"

echo "Installing pip packages from requirements-demo-ui.txt (PyQt5 deferred)…"
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
# Install non-Qt pins first so ensure_pip_pyqt owns a single clean PyQt5 tree.
REQ_NO_PYQT="$(mktemp)"
grep -vE '^[[:space:]]*PyQt5([=<>!].*)?([[:space:]]*#.*)?$' "${REQ_FILE}" > "${REQ_NO_PYQT}" || true
conda run -n "${ENV_NAME}" python -m pip install -r "${REQ_NO_PYQT}"
rm -f "${REQ_NO_PYQT}"
conda run -n "${ENV_NAME}" python -m pip install -e "${REPOSITORY}"
ensure_pip_pyqt "${ENV_NAME}"

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
