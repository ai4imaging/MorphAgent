#!/usr/bin/env bash
# Start Xvfb + fluxbox + x11vnc + noVNC, then launch MorphAgent UI.
set -euo pipefail

HANDOFF_ROOT="${HANDOFF_ROOT:-/opt/MorphAgent_UI}"
ENV_NAME="${MORPHAGENT_ENV_NAME:-morphagent}"
DISPLAY_NUM="${DISPLAY_NUM:-1}"
export DISPLAY=":${DISPLAY_NUM}"
RESOLUTION="${RESOLUTION:-1920x1080x24}"
VNC_PORT="${VNC_PORT:-5900}"
NO_VNC_PORT="${NO_VNC_PORT:-6080}"
DATA_ROOT="${MORPHAGENT_DATA:-/data}"

mkdir -p "${DATA_ROOT}/results" "${DATA_ROOT}/workspace"

# Optional host-mounted .env overrides the image default.
if [[ -f "${DATA_ROOT}/.env" ]]; then
  cp "${DATA_ROOT}/.env" "${HANDOFF_ROOT}/MorphAgent/.env"
  chmod 600 "${HANDOFF_ROOT}/MorphAgent/.env" || true
elif [[ ! -f "${HANDOFF_ROOT}/MorphAgent/.env" && -f "${HANDOFF_ROOT}/MorphAgent/.env.example" ]]; then
  cp "${HANDOFF_ROOT}/MorphAgent/.env.example" "${HANDOFF_ROOT}/MorphAgent/.env"
  chmod 600 "${HANDOFF_ROOT}/MorphAgent/.env" || true
fi

# Keep sandbox env pointer consistent for code-route extract().
if [[ -f "${HANDOFF_ROOT}/MorphAgent/.env" ]]; then
  if grep -q '^CONDA_ENV=' "${HANDOFF_ROOT}/MorphAgent/.env"; then
    sed -i 's/^CONDA_ENV=.*/CONDA_ENV=morphagent_sandbox/' "${HANDOFF_ROOT}/MorphAgent/.env"
  else
    printf '\nCONDA_ENV=morphagent_sandbox\n' >> "${HANDOFF_ROOT}/MorphAgent/.env"
  fi
fi

cleanup() {
  jobs -p | xargs -r kill 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[MorphAgent] starting virtual display ${DISPLAY} (${RESOLUTION})"
Xvfb "${DISPLAY}" -screen 0 "${RESOLUTION}" -ac +extension GLX +render -noreset &
sleep 1

fluxbox >/tmp/fluxbox.log 2>&1 &
sleep 1

echo "[MorphAgent] starting VNC on :${VNC_PORT}"
x11vnc -display "${DISPLAY}" -forever -shared -rfbport "${VNC_PORT}" -nopw -xkb -quiet \
  >/tmp/x11vnc.log 2>&1 &

echo "[MorphAgent] starting noVNC on :${NO_VNC_PORT}"
websockify --web=/usr/share/novnc "${NO_VNC_PORT}" "localhost:${VNC_PORT}" \
  >/tmp/novnc.log 2>&1 &

# Give noVNC a moment before healthchecks / browser open.
sleep 2

echo "[MorphAgent] UI ready → open http://localhost:${NO_VNC_PORT}/vnc.html?autoconnect=true&resize=remote"
echo "[MorphAgent] Tip: Load previous run → MorphAgent/demo/data/results/completed_demo_run (no API needed)"

# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

cd "${HANDOFF_ROOT}"
exec python "${HANDOFF_ROOT}/MorphAgent/launch_ui.py" "$@"
