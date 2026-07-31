#!/bin/bash
# MorphAgent UI — macOS one-click launcher (double-click in Finder)
set -euo pipefail

cd "$(cd "$(dirname "$0")" && pwd)"

ROOT="$(pwd)"
# Packaged zip layout: files live next to this script.
# Repo layout: this script lives in docker/mac/ → handoff root is ../..
if [[ -f "${ROOT}/docker-compose.yml" && -f "${ROOT}/Dockerfile" ]]; then
  COMPOSE_DIR="${ROOT}"
  export COMPOSE_FILE="${ROOT}/docker-compose.yml"
elif [[ -f "${ROOT}/../docker-compose.yml" ]]; then
  COMPOSE_DIR="$(cd "${ROOT}/.." && pwd)"
  export COMPOSE_FILE="${COMPOSE_DIR}/docker-compose.yml"
else
  osascript -e 'display alert "MorphAgent UI" message "Cannot find docker-compose.yml next to this launcher." as critical' || true
  exit 1
fi

UI_URL="http://localhost:6080/vnc.html?autoconnect=true&resize=remote"

if ! command -v docker >/dev/null 2>&1; then
  osascript -e 'display alert "MorphAgent UI" message "Docker is not installed.\n\nInstall Docker Desktop for Mac, start it, then double-click again:\nhttps://www.docker.com/products/docker-desktop/" as critical' || true
  open "https://www.docker.com/products/docker-desktop/" || true
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  osascript -e 'display alert "MorphAgent UI" message "Docker Desktop is not running.\n\nPlease start Docker Desktop, wait until it is ready, then double-click again." as critical' || true
  open -a Docker 2>/dev/null || true
  exit 1
fi

ARCH="$(uname -m)"
if [[ "${ARCH}" == "arm64" ]]; then
  export DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/arm64}"
else
  export DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
fi

echo "============================================================"
echo " MorphAgent UI (Docker)"
echo " platform: ${DOCKER_PLATFORM}"
echo " compose:  ${COMPOSE_FILE}"
echo "============================================================"
echo "First launch builds the image (several minutes). Later starts are fast."
echo

cd "${COMPOSE_DIR}"
docker compose -f "${COMPOSE_FILE}" up -d --build

echo
echo "Waiting for noVNC…"
for _ in $(seq 1 90); do
  if curl -fsS "http://127.0.0.1:6080/" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

open "${UI_URL}" || true

osascript -e "display notification \"Browser opened at ${UI_URL}\" with title \"MorphAgent UI\"" || true

echo
echo "MorphAgent UI is running."
echo "  Browser: ${UI_URL}"
echo "  Stop:    docker compose -f \"${COMPOSE_FILE}\" down"
echo
echo "Tip: Home → Load a previous run → completed_demo_run (no API key needed)."
echo "Press Enter to close this window…"
read -r _ || true
