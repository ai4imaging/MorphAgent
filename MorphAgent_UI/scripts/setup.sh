#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HANDOFF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPOSITORY="${HANDOFF_ROOT}/MorphAgent"
ENV_NAME="${MORPHAGENT_ENV_NAME:-morphagent}"
ALLEN_ENV_NAME="${MORPHAGENT_ALLEN_ENV_NAME:-morphagent_allen}"
ALLEN_YML="${REPOSITORY}/envs/environment_allen.yml"
ALLEN_YML_ALT="${HANDOFF_ROOT}/dependencies/environment-allen-optional.yml"
ALLEN_REQ="${REPOSITORY}/envs/requirements-allen.txt"
INSTALL_ALLEN="${MORPHAGENT_INSTALL_ALLEN:-1}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda was not found. Install Miniforge/Miniconda/Anaconda first." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "Using existing conda environment: ${ENV_NAME}"
else
  conda create -y -n "${ENV_NAME}" python=3.10 pip
fi

conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install -r "${HANDOFF_ROOT}/dependencies/requirements-demo-ui.txt"
conda run -n "${ENV_NAME}" python -m pip install -e "${REPOSITORY}"
if [[ ! -f "${REPOSITORY}/.env" ]]; then
  cp "${REPOSITORY}/.env.example" "${REPOSITORY}/.env"
  chmod 600 "${REPOSITORY}/.env"
fi

# --- Allen classic segmentation (optional; soft-fail with WARNING) ------------
# Apple Silicon: Python 3.6 only via CONDA_SUBDIR=osx-64 (Rosetta).
# TIFF driver uses tifffile + aicssegmentation (no aicsimageio / aicspylibczi).
# Failure must NOT abort the main morphagent install.
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

  if conda env list | awk '{print $1}' | grep -Fxq "${ALLEN_ENV_NAME}"; then
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

conda run -n "${ENV_NAME}" python "${HANDOFF_ROOT}/scripts/verify_install.py" --ui-smoke

echo
echo "Installation verified. Start MorphAgent with:"
echo "  MORPHAGENT_ENV_NAME=${ENV_NAME} bash scripts/start_ui.sh"
echo "Allen segmentation env (optional): ${ALLEN_ENV_NAME}"
