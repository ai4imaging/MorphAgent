#!/usr/bin/env bash
# MorphAgent UI setup (macOS / Linux).
# Creates:
#   - `morphagent`          — Qt UI + agent (LangChain / OpenAI client)
#   - `morphagent_sandbox`  — frozen scientific stack for agent-generated extract()
#   - optional `morphagent_allen` — Allen segmentation
#
# Design rules (keep this script boring and idempotent):
#   1) pip is owned by conda — NEVER `pip install --upgrade pip` (breaks mixed installs).
#   2) Qt is owned by conda `pyqt=5` — NEVER pip-install PyQt5 / PyQt5-Qt5.
#   3) Feature code uses morphagent_sandbox; UI/agent uses morphagent.
#
# Optional: MORPHAGENT_RECREATE_ENVS=1  → delete + recreate morphagent + sandbox first.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HANDOFF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPOSITORY="${HANDOFF_ROOT}/MorphAgent"
ENV_NAME="${MORPHAGENT_ENV_NAME:-morphagent}"
SANDBOX_ENV_NAME="${MORPHAGENT_SANDBOX_ENV_NAME:-morphagent_sandbox}"
ALLEN_ENV_NAME="${MORPHAGENT_ALLEN_ENV_NAME:-morphagent_allen}"
ALLEN_YML="${REPOSITORY}/envs/environment_allen.yml"
ALLEN_YML_ALT="${HANDOFF_ROOT}/dependencies/environment-allen-optional.yml"
ALLEN_REQ="${REPOSITORY}/envs/requirements-allen.txt"
REQ_FILE="${HANDOFF_ROOT}/dependencies/requirements-demo-ui.txt"
SANDBOX_REQ_FILE="${HANDOFF_ROOT}/dependencies/requirements-sandbox.txt"
INSTALL_ALLEN="${MORPHAGENT_INSTALL_ALLEN:-1}"
RECREATE_ENVS="${MORPHAGENT_RECREATE_ENVS:-0}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda was not found. Install Miniforge/Miniconda/Anaconda first." >&2
  exit 1
fi

# Miniconda 24+ may prompt for Anaconda ToS via conda-anaconda-tos and hang/EOF
# in non-interactive setup. Official workaround; we install from conda-forge.
ensure_conda_noninteractive() {
  export CONDA_NO_PLUGINS=true
  export CONDA_REPORT_ERRORS=false
}
ensure_conda_noninteractive
echo "[OK] CONDA_NO_PLUGINS=${CONDA_NO_PLUGINS} (avoids Anaconda ToS interactive prompt)"

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "ERROR: missing requirements file: ${REQ_FILE}" >&2
  exit 1
fi
if [[ ! -f "${SANDBOX_REQ_FILE}" ]]; then
  echo "ERROR: missing sandbox requirements file: ${SANDBOX_REQ_FILE}" >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

env_exists() {
  conda env list | awk '{print $1}' | grep -Fxq "$1"
}

# Older conda (e.g. 23.1) rejects `conda env create -y` / `conda env remove -y`
# with: conda-env: error: unrecognized arguments: -y
# Try common variants until one works.
conda_env_remove() {
  local name="$1"
  local rc=1
  set +e
  set +o pipefail

  echo "  try: conda env remove -y -n ${name}"
  conda env remove -y -n "${name}"
  rc=$?
  if [[ "${rc}" -eq 0 ]]; then set -o pipefail; set -e; return 0; fi

  echo "  try: conda env remove --yes -n ${name}"
  conda env remove --yes -n "${name}"
  rc=$?
  if [[ "${rc}" -eq 0 ]]; then set -o pipefail; set -e; return 0; fi

  echo "  try: printf y | conda env remove -n ${name}"
  printf 'y\ny\ny\n' | conda env remove -n "${name}"
  rc=$?
  if [[ "${rc}" -eq 0 ]]; then set -o pipefail; set -e; return 0; fi

  echo "  try: conda remove -y -n ${name} --all"
  conda remove -y -n "${name}" --all
  rc=$?
  set -o pipefail
  set -e
  return "${rc}"
}

conda_env_create_from_yaml() {
  # Usage: [CONDA_SUBDIR=...] conda_env_create_from_yaml NAME FILE.yml
  # Inherits CONDA_SUBDIR from the caller when set (Apple Silicon osx-64).
  local name="$1"
  local yml="$2"
  local rc=1

  _cleanup_partial_env() {
    if env_exists "${name}"; then
      echo "  cleaning partial env ${name} before next create attempt…"
      conda_env_remove "${name}" || true
    fi
  }

  set +e
  set +o pipefail

  echo "  try: conda env create -y -n ${name} -f ${yml}"
  conda env create -y -n "${name}" -f "${yml}"
  rc=$?
  if [[ "${rc}" -eq 0 ]] && env_exists "${name}"; then set -o pipefail; set -e; return 0; fi
  _cleanup_partial_env

  echo "  try: conda env create --yes -n ${name} -f ${yml}"
  conda env create --yes -n "${name}" -f "${yml}"
  rc=$?
  if [[ "${rc}" -eq 0 ]] && env_exists "${name}"; then set -o pipefail; set -e; return 0; fi
  _cleanup_partial_env

  echo "  try: conda env create -n ${name} -f ${yml}  (no -y; older conda-env)"
  conda env create -n "${name}" -f "${yml}"
  rc=$?
  if [[ "${rc}" -eq 0 ]] && env_exists "${name}"; then set -o pipefail; set -e; return 0; fi
  _cleanup_partial_env

  echo "  try: printf y | conda env create -n ${name} -f ${yml}"
  printf 'y\ny\ny\n' | conda env create -n "${name}" -f "${yml}"
  rc=$?
  if [[ "${rc}" -eq 0 ]] && env_exists "${name}"; then set -o pipefail; set -e; return 0; fi
  _cleanup_partial_env

  echo "  try: conda create -y -n ${name} -f ${yml}"
  conda create -y -n "${name}" -f "${yml}"
  rc=$?
  if [[ "${rc}" -eq 0 ]] && env_exists "${name}"; then set -o pipefail; set -e; return 0; fi
  _cleanup_partial_env

  set -o pipefail
  set -e
  echo "ERROR: all conda env create variants failed for ${name} from ${yml}" >&2
  echo "ERROR: conda version: $(conda --version 2>/dev/null || echo unknown)" >&2
  return 1
}

ensure_env() {
  local env="$1"
  ensure_conda_noninteractive
  if [[ "${RECREATE_ENVS}" == "1" ]] && env_exists "${env}"; then
    echo "MORPHAGENT_RECREATE_ENVS=1 → removing conda env ${env}"
    conda_env_remove "${env}"
  fi
  if env_exists "${env}"; then
    echo "Using existing conda environment: ${env}"
  else
    echo "Creating conda environment: ${env}"
    # Prefer --override-channels so default anaconda.com channels (ToS plugin) are skipped.
    # `conda create -y` is widely supported; keep a no -y fallback just in case.
    set +e
    set +o pipefail
    conda create -y -n "${env}" -c conda-forge --override-channels "python=3.10" "pip>=24"
    local rc=$?
    if [[ "${rc}" -ne 0 ]]; then
      echo "  retry: conda create --yes -n ${env} ..."
      conda create --yes -n "${env}" -c conda-forge --override-channels "python=3.10" "pip>=24"
      rc=$?
    fi
    if [[ "${rc}" -ne 0 ]]; then
      echo "  retry without override-channels..."
      conda create -y -n "${env}" -c conda-forge "python=3.10" "pip>=24"
      rc=$?
    fi
    if [[ "${rc}" -ne 0 ]]; then
      echo "  retry without -y: conda create -n ${env} ..."
      printf 'a\na\na\ny\ny\ny\n' | conda create -n "${env}" -c conda-forge "python=3.10" "pip>=24"
      rc=$?
    fi
    set -o pipefail
    set -e
    if [[ "${rc}" -ne 0 ]] || ! env_exists "${env}"; then
      echo "ERROR: failed to create conda env ${env}" >&2
      exit 1
    fi
  fi
}

# Always heal pip from conda. Do not self-upgrade pip with pip.
# Wipe leftover mixed pip trees (e.g. pip-26.1 + pip-26.2 dist-info / build_env
# as both .py and package), which cause:
#   ImportError: cannot import name 'get_runnable_pip'
repair_pip() {
  local env="$1"
  ensure_conda_noninteractive
  echo "Repairing pip in ${env} via conda (no pip self-upgrade)…"
  local sp
  sp="$(conda run -n "${env}" python -c 'import site; print(site.getsitepackages()[0])')"
  echo "  cleaning pip leftovers under ${sp}"
  rm -rf "${sp}/pip"
  # Remove any pip-*.dist-info / egg-info without relying on shell globs.
  find "${sp}" -maxdepth 1 \( -name 'pip-*.dist-info' -o -name 'pip-*.egg-info' \) -print0 \
    | xargs -0 rm -rf 2>/dev/null || true
  conda install -y -n "${env}" -c conda-forge --override-channels --force-reinstall "pip>=24" "setuptools" "wheel"
  conda run -n "${env}" python -m pip --version
}

pip_install() {
  local env="$1"
  shift
  repair_pip "${env}"
  conda run -n "${env}" python -m pip install "$@"
}

write_filtered_ui_requirements() {
  local dst="$1"
  python3 - <<PY
from pathlib import Path
import re
src = Path("${REQ_FILE}")
dst = Path("${dst}")
# Drop PyQt5 and any PyQt5-* lines; conda pyqt=5 owns Qt.
pat = re.compile(r"^\s*PyQt5([=<>!\.].*)?\s*(#.*)?$", re.IGNORECASE)
lines = []
for ln in src.read_text(encoding="utf-8").splitlines():
    if pat.match(ln):
        continue
    if ln.strip().lower().startswith("pyqt5"):
        continue
    lines.append(ln)
dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[OK] filtered UI requirements → {dst} ({len(lines)} lines; PyQt5 skipped)")
PY
}

install_ui_conda_stack() {
  local env="$1"
  ensure_conda_noninteractive
  echo "Installing UI conda stack into ${env} (conda pyqt=5 + science libs)…"
  conda install -y -n "${env}" -c conda-forge --override-channels \
    "python=3.10" \
    "pip>=24" \
    "setuptools" \
    "wheel" \
    "numpy>=1.26,<3" \
    "scipy>=1.11" \
    "pandas>=2.0" \
    "pyqt=5" \
    "pyqt5-sip" \
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

install_sandbox_conda_stack() {
  local env="$1"
  ensure_conda_noninteractive
  echo "Installing sandbox conda stack into ${env} (science only, no Qt)…"
  conda install -y -n "${env}" -c conda-forge --override-channels \
    "python=3.10" \
    "pip>=24" \
    "setuptools" \
    "wheel" \
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

ensure_single_conda_qt() {
  local env="$1"
  ensure_conda_noninteractive
  echo "Ensuring single conda Qt stack in ${env}…"
  # Remove only the pip-bundled Qt runtime if someone installed it earlier.
  conda run -n "${env}" python -m pip uninstall -y PyQt5-Qt5 2>/dev/null || true
  conda install -y -n "${env}" -c conda-forge --override-channels --force-reinstall "pyqt=5" "pyqt5-sip" "qtpy>=2.4"
  conda run -n "${env}" python - <<'PY'
import PyQt5
from PyQt5 import QtCore
from pathlib import Path
import qtpy, numpy
print("[OK] single Qt stack: PyQt5=%s, qtpy=%s, numpy=%s" % (
    QtCore.PYQT_VERSION_STR, qtpy.__version__, numpy.__version__))
pip_qt = Path(PyQt5.__file__).resolve().parent / "Qt5" / "lib"
if pip_qt.exists():
    raise SystemExit(
        "Pip PyQt5 Qt binaries still present under site-packages/PyQt5/Qt5; "
        "remove PyQt5-Qt5 and reinstall conda pyqt=5"
    )
PY
}

setup_ui_env() {
  ensure_env "${ENV_NAME}"
  install_ui_conda_stack "${ENV_NAME}"

  local filtered_req
  filtered_req="$(mktemp "${TMPDIR:-/tmp}/morphagent-req-no-pyqt.XXXXXX")"
  write_filtered_ui_requirements "${filtered_req}"

  echo "Installing UI pip packages (PyQt5 skipped)…"
  pip_install "${ENV_NAME}" -r "${filtered_req}"
  rm -f "${filtered_req}"
  pip_install "${ENV_NAME}" -e "${REPOSITORY}"

  ensure_single_conda_qt "${ENV_NAME}"
}

setup_sandbox_env() {
  ensure_env "${SANDBOX_ENV_NAME}"
  install_sandbox_conda_stack "${SANDBOX_ENV_NAME}"

  echo "Installing frozen sandbox pip stack…"
  pip_install "${SANDBOX_ENV_NAME}" -r "${SANDBOX_REQ_FILE}"

  conda run -n "${SANDBOX_ENV_NAME}" python - <<'PY'
import importlib
need = ["numpy", "scipy", "pandas", "skimage", "sklearn", "cv2", "mahotas", "tifffile"]
missing = []
for name in need:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append((name, repr(exc)))
import numpy, skimage
print("[OK] sandbox", "numpy", numpy.__version__, "skimage", skimage.__version__, "MISSING", missing or "none")
assert not missing, missing
PY
}

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
    echo "conda: $(conda --version 2>/dev/null || echo unknown)"
    if [[ "${arch}" == "arm64" ]]; then
      echo "Apple Silicon detected: using CONDA_SUBDIR=osx-64 (Rosetta)"
      CONDA_SUBDIR=osx-64 conda_env_create_from_yaml "${ALLEN_ENV_NAME}" "${yml}"
    else
      conda_env_create_from_yaml "${ALLEN_ENV_NAME}" "${yml}"
    fi
  fi

  if [[ "${arch}" == "arm64" ]]; then
    conda activate "${ALLEN_ENV_NAME}"
    conda env config vars set CONDA_SUBDIR=osx-64 || true
    conda deactivate
  fi

  echo "Installing Allen scientific stack into ${ALLEN_ENV_NAME}..."
  # Allen is py3.6 — keep its own pip bootstrap; do not touch morphagent pip.
  conda run -n "${ALLEN_ENV_NAME}" python -m pip install --upgrade 'pip<22' 'setuptools<59' 'wheel'
  conda run -n "${ALLEN_ENV_NAME}" python -m pip install -r "${ALLEN_REQ}"
  echo "Installing vendored aicssegmentation (no-deps)..."
  conda run -n "${ALLEN_ENV_NAME}" python -m pip install -e "${REPOSITORY}/segmentation_allen" --no-deps
  echo "Verifying Allen installation..."
  conda run -n "${ALLEN_ENV_NAME}" python "${REPOSITORY}/segmentation_allen/check_installation.py"
  echo "[OK] Allen environment ${ALLEN_ENV_NAME} is ready"
}

echo "============================================================"
echo " MorphAgent UI setup"
echo " handoff: ${HANDOFF_ROOT}"
echo " UI env: ${ENV_NAME}"
echo " sandbox env: ${SANDBOX_ENV_NAME}"
echo " recreate envs: ${RECREATE_ENVS}"
echo "============================================================"

setup_ui_env
setup_sandbox_env

if [[ ! -f "${REPOSITORY}/.env" ]]; then
  if [[ -f "${REPOSITORY}/.env.example" ]]; then
    cp "${REPOSITORY}/.env.example" "${REPOSITORY}/.env"
    chmod 600 "${REPOSITORY}/.env" || true
  else
    echo "[WARN] ${REPOSITORY}/.env.example missing; skipped creating .env" >&2
  fi
fi

if [[ -f "${REPOSITORY}/.env" ]]; then
  if grep -q '^CONDA_ENV=' "${REPOSITORY}/.env"; then
    sed -i.bak 's/^CONDA_ENV=.*/CONDA_ENV=morphagent_sandbox/' "${REPOSITORY}/.env" && rm -f "${REPOSITORY}/.env.bak"
  else
    printf '\nCONDA_ENV=morphagent_sandbox\n' >> "${REPOSITORY}/.env"
  fi
fi

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
echo "  bash scripts/start_ui.sh"
echo "UI / agent env: ${ENV_NAME}"
echo "Code sandbox env (feature extract): ${SANDBOX_ENV_NAME}"
echo "Allen segmentation env (optional): ${ALLEN_ENV_NAME}"
echo
echo "If an old env is badly corrupted, recreate cleanly with:"
echo "  MORPHAGENT_RECREATE_ENVS=1 bash scripts/setup.sh"
