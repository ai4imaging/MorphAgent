#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HANDOFF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPOSITORY="${HANDOFF_ROOT}/MorphAgent"
ENV_NAME="${MORPHAGENT_ENV_NAME:-morphagent}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda was not found. Install Miniforge/Miniconda/Anaconda first." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
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
conda run -n "${ENV_NAME}" python "${HANDOFF_ROOT}/scripts/verify_install.py" --ui-smoke

echo
echo "Installation verified. Start MorphAgent with:"
echo "  MORPHAGENT_ENV_NAME=${ENV_NAME} bash scripts/start_ui.sh"
