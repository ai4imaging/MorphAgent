#!/usr/bin/env bash
# MorphAgent UI Lite — single-env setup (macOS / Linux).
# Creates conda env morphagent_lite (Python + pip only), then pip-installs everything else.
#
# Avoids classic+conda-forge mega-solves that crash old conda on large indexes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HANDOFF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${MORPHAGENT_ENV_NAME:-morphagent_lite}"
REQ_FILE="${HANDOFF_ROOT}/dependencies/requirements-lite.txt"
ENV_FILE="${HANDOFF_ROOT}/MorphAgent/.env"
ENV_EXAMPLE="${HANDOFF_ROOT}/MorphAgent/.env.example"

export CONDA_REPORT_ERRORS="${CONDA_REPORT_ERRORS:-false}"
# Do not force CONDA_NO_PLUGINS / CONDA_SOLVER=classic by default (keeps libmamba when available).
# Classic fallback is applied only around a tiny python+pip create if needed.

echo "============================================================"
echo " MorphAgent UI Lite setup"
echo " Scope: Tau demo + Code/VLM; knowledge via precomputed txt"
echo "        (skips PDF parse / PubMed / auto deep-research / Allen)"
echo " Root: ${HANDOFF_ROOT}"
echo " Env:  ${ENV_NAME} (python+pip via conda; science stack via pip)"
echo " Live PDF/literature/Allen: use MorphAgent_UI_Docker/"
echo "============================================================"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda was not found. Install Miniconda/Anaconda first." >&2
  exit 1
fi
if [[ ! -f "${REQ_FILE}" ]]; then
  echo "ERROR: missing ${REQ_FILE}" >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

accept_anaconda_tos_best_effort() {
  # Best-effort; ignore failures on Miniforge / old conda without tos plugin.
  local ch
  for ch in \
    "https://repo.anaconda.com/pkgs/main" \
    "https://repo.anaconda.com/pkgs/r" \
    "https://repo.anaconda.com/pkgs/msys2"
  do
    conda tos accept --override-channels --channel "${ch}" >/dev/null 2>&1 \
      || conda tos accept -c "${ch}" >/dev/null 2>&1 \
      || true
  done
  echo "[OK] Best-effort Anaconda ToS accept"
}

create_lite_env() {
  # Tiny solve only — never conda-install numpy/scipy/pyqt/etc.
  echo "[..] Creating ${ENV_NAME} (python=3.10 + pip only; science stack via pip)"
  unset CONDA_NO_PLUGINS CONDA_SOLVER || true
  if conda create -n "${ENV_NAME}" python=3.10 pip -y -c defaults --override-channels; then
    return 0
  fi
  echo "[WARN] conda create failed; retry with classic fallback (still python+pip only)"
  export CONDA_NO_PLUGINS=true
  export CONDA_SOLVER=classic
  conda create -n "${ENV_NAME}" python=3.10 pip -y -c defaults --override-channels
  unset CONDA_NO_PLUGINS CONDA_SOLVER || true
}

accept_anaconda_tos_best_effort

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  if [[ "${MORPHAGENT_RECREATE_ENVS:-0}" == "1" ]]; then
    echo "[..] Removing existing env ${ENV_NAME}"
    conda env remove -n "${ENV_NAME}" -y
  else
    echo "[OK] Env ${ENV_NAME} already exists (set MORPHAGENT_RECREATE_ENVS=1 to recreate)"
  fi
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  create_lite_env
fi

echo "[..] Upgrading pip"
conda run --no-capture-output -n "${ENV_NAME}" python -m pip install -U pip setuptools wheel

echo "[..] pip install -r dependencies/requirements-lite.txt (includes PyQt5; no conda-forge science solve)"
if ! conda run --no-capture-output -n "${ENV_NAME}" python -m pip install -r "${REQ_FILE}" --retries 5; then
  echo "[WARN] Full pip install failed; retrying PyQt5 via pip only, then requirements again"
  conda run --no-capture-output -n "${ENV_NAME}" python -m pip install "PyQt5>=5.15,<5.16" qtpy --retries 5 || true
  conda run --no-capture-output -n "${ENV_NAME}" python -m pip install -r "${REQ_FILE}" --retries 5
fi

echo "[..] pip install -e MorphAgent"
conda run --no-capture-output -n "${ENV_NAME}" python -m pip install -e "${HANDOFF_ROOT}/MorphAgent"

# Seed .env for Lite defaults without clobbering existing API keys.
if [[ ! -f "${ENV_FILE}" && -f "${ENV_EXAMPLE}" ]]; then
  cp "${ENV_EXAMPLE}" "${ENV_FILE}"
fi
if [[ -f "${ENV_FILE}" ]]; then
  ENV_FILE="${ENV_FILE}" ENV_NAME="${ENV_NAME}" python3 - <<'PY'
import os
from pathlib import Path
path = Path(os.environ["ENV_FILE"])
env_name = os.environ["ENV_NAME"]
text = path.read_text(encoding="utf-8") if path.is_file() else ""
lines = text.splitlines()
wanted = {
    "CONDA_ENV": env_name,
    "SEGMENTATION_BACKEND": "none",
}
out = []
seen = set()
for line in lines:
    if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
        out.append(line)
        continue
    key = line.split("=", 1)[0].strip()
    if key in wanted:
        out.append(f"{key}={wanted[key]}")
        seen.add(key)
    else:
        out.append(line)
for key, value in wanted.items():
    if key not in seen:
        out.append(f"{key}={value}")
filtered = [line for line in out if not line.startswith("SEGMENTATION_CONDA_ENV=")]
path.write_text("\n".join(filtered) + "\n", encoding="utf-8")
print(f"[OK] Updated {path}")
PY
fi

echo "[..] verify_install.py"
conda run --no-capture-output -n "${ENV_NAME}" python "${SCRIPT_DIR}/verify_install.py"

echo
echo "[OK] Lite setup complete (Tau demo trial)."
echo "     Launch: bash scripts/start_ui.sh"
echo "     Knowledge: demo/precomputed/*.txt injected into prompts"
echo "============================================================"
