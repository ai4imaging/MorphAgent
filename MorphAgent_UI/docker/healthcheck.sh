#!/usr/bin/env bash
set -euo pipefail

NO_VNC_PORT="${NO_VNC_PORT:-6080}"
DISPLAY_NUM="${DISPLAY_NUM:-1}"

curl --fail --silent --show-error "http://127.0.0.1:${NO_VNC_PORT}/" >/dev/null
test -S "/tmp/.X11-unix/X${DISPLAY_NUM}"
pgrep -f 'websockify.*6080' >/dev/null
pgrep -f 'launch_ui.py' >/dev/null
