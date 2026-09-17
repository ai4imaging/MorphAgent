#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HANDOFF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${MORPHAGENT_ENV_NAME:-morphagent_lite}"

export CONDA_REPORT_ERRORS="${CONDA_REPORT_ERRORS:-false}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda was not found." >&2
  exit 1
fi

exec conda run --no-capture-output -n "${ENV_NAME}" \
  python "${HANDOFF_ROOT}/MorphAgent/launch_ui.py" "$@"
