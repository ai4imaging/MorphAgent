#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${MORPHAGENT_ENV_NAME:-morphagent_lite}"
exec conda run --no-capture-output -n "${ENV_NAME}" python "${REPO_ROOT}/launch_web_ui.py" "$@"
