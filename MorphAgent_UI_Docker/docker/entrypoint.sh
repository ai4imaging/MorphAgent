#!/usr/bin/env bash
# Start the virtual desktop and the unchanged MorphAgent PyQt application.
set -euo pipefail

HANDOFF_ROOT="${HANDOFF_ROOT:-/opt/MorphAgent_UI_Docker}"
APP_ROOT="${HANDOFF_ROOT}/MorphAgent"
UI_ENV="${MORPHAGENT_ENV_NAME:-morphagent}"
DATA_ROOT="${MORPHAGENT_DATA_ROOT:-/data}"
WORKSPACE_ROOT="${MORPHAGENT_WORKSPACE_ROOT:-/workspace}"
CONFIG_FILE="${DATA_ROOT}/config/.env"
APP_ENV_FILE="${APP_ROOT}/.env"
PERSISTENT_RESULTS="${DATA_ROOT}/demo-results"
APP_RESULTS="${APP_ROOT}/demo/data/results"
SEED_RUN="/opt/morphagent-seed/completed_demo_run"
LOG_DIR="${DATA_ROOT}/logs"
DISPLAY_NUM="${DISPLAY_NUM:-1}"
RESOLUTION="${RESOLUTION:-1920x1080x24}"
VNC_PORT="${VNC_PORT:-5900}"
NO_VNC_PORT="${NO_VNC_PORT:-6080}"

if [[ ! "${RESOLUTION}" =~ ^[0-9]+x[0-9]+x(16|24|32)$ ]]; then
  echo "ERROR: RESOLUTION must look like 1920x1080x24, got: ${RESOLUTION}" >&2
  exit 2
fi
for port_name in VNC_PORT NO_VNC_PORT; do
  port_value="${!port_name}"
  if [[ ! "${port_value}" =~ ^[0-9]+$ ]] || ((port_value < 1024 || port_value > 65535)); then
    echo "ERROR: ${port_name} must be an integer from 1024 to 65535" >&2
    exit 2
  fi
done

export DISPLAY=":${DISPLAY_NUM}"
export HOME="${DATA_ROOT}/home"
export XDG_CONFIG_HOME="${DATA_ROOT}/xdg/config"
export XDG_CACHE_HOME="${DATA_ROOT}/xdg/cache"
export XDG_RUNTIME_DIR="/tmp/morphagent-runtime"

mkdir -p \
  "$(dirname "${CONFIG_FILE}")" \
  "${PERSISTENT_RESULTS}" \
  "${LOG_DIR}" \
  "${WORKSPACE_ROOT}" \
  "${HOME}" \
  "${XDG_CONFIG_HOME}" \
  "${XDG_CACHE_HOME}" \
  "${XDG_RUNTIME_DIR}"
chmod 0700 "${XDG_RUNTIME_DIR}"

upsert_env() {
  local key="$1"
  local value="$2"
  if grep -q "^${key}=" "${CONFIG_FILE}"; then
    sed -i "s|^${key}=.*|${key}=${value}|" "${CONFIG_FILE}"
  else
    printf '%s=%s\n' "${key}" "${value}" >>"${CONFIG_FILE}"
  fi
}

if [[ ! -f "${CONFIG_FILE}" ]]; then
  cp "${APP_ROOT}/.env.example" "${CONFIG_FILE}"
fi
chmod 0600 "${CONFIG_FILE}"
upsert_env CONDA_BASE_PATH /opt/conda
upsert_env CONDA_ENV morphagent_sandbox
upsert_env SEGMENTATION_BACKEND allen
upsert_env SEGMENTATION_CONDA_ENV morphagent_allen

# The UI writes its API settings to APP_ENV_FILE. A symlink makes every save
# land in the persistent bind mount without exposing credentials in the image.
rm -f "${APP_ENV_FILE}"
ln -s "${CONFIG_FILE}" "${APP_ENV_FILE}"

# The repository demo path is hard-coded by the existing UI. Redirect only its
# results directory into persistent state and seed the standard completed run.
if [[ ! -d "${PERSISTENT_RESULTS}/completed_demo_run" ]]; then
  cp -a "${SEED_RUN}" "${PERSISTENT_RESULTS}/completed_demo_run"
fi
rm -rf "${APP_RESULTS}"
ln -s "${PERSISTENT_RESULTS}" "${APP_RESULTS}"

PIDS=()
cleanup() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[MorphAgent] starting virtual display ${DISPLAY} (${RESOLUTION})"
Xvfb "${DISPLAY}" -screen 0 "${RESOLUTION}" -ac +extension GLX +render -noreset \
  >"${LOG_DIR}/xvfb.log" 2>&1 &
PIDS+=("$!")

for _ in $(seq 1 50); do
  [[ -S "/tmp/.X11-unix/X${DISPLAY_NUM}" ]] && break
  sleep 0.1
done
if [[ ! -S "/tmp/.X11-unix/X${DISPLAY_NUM}" ]]; then
  echo "ERROR: Xvfb did not create display ${DISPLAY}" >&2
  exit 1
fi

fluxbox >"${LOG_DIR}/fluxbox.log" 2>&1 &
PIDS+=("$!")

VNC_AUTH=(-nopw)
if [[ -n "${VNC_PASSWORD:-}" ]]; then
  PASS_FILE="${DATA_ROOT}/config/x11vnc.pass"
  x11vnc -storepasswd "${VNC_PASSWORD}" "${PASS_FILE}" >/dev/null
  chmod 0600 "${PASS_FILE}"
  VNC_AUTH=(-rfbauth "${PASS_FILE}")
fi

# Raw VNC is reachable only from inside the container. The Compose file also
# binds the browser endpoint to 127.0.0.1 on the host.
x11vnc -display "${DISPLAY}" -localhost -forever -shared -xkb -quiet \
  -rfbport "${VNC_PORT}" "${VNC_AUTH[@]}" \
  >"${LOG_DIR}/x11vnc.log" 2>&1 &
PIDS+=("$!")

websockify --web=/usr/share/novnc "${NO_VNC_PORT}" "127.0.0.1:${VNC_PORT}" \
  >"${LOG_DIR}/novnc.log" 2>&1 &
PIDS+=("$!")

echo "[MorphAgent] browser UI: http://127.0.0.1:${NO_VNC_PORT}/vnc.html?autoconnect=true&resize=scale"
echo "[MorphAgent] custom data inside the container: ${WORKSPACE_ROOT}"

cd "${HANDOFF_ROOT}"
conda run --no-capture-output -n "${UI_ENV}" \
  python "${APP_ROOT}/launch_ui.py" "$@" \
  >"${LOG_DIR}/ui.log" 2>&1 &
UI_PID="$!"
PIDS+=("${UI_PID}")
wait "${UI_PID}"
