#!/usr/bin/env bash
# MorphAgent UI — Linux one-click launcher
set -euo pipefail

cd "$(cd "$(dirname "$0")" && pwd)"
ROOT="$(pwd)"

if [[ -f "${ROOT}/docker-compose.yml" && -f "${ROOT}/Dockerfile" ]]; then
  COMPOSE_DIR="${ROOT}"
  COMPOSE_FILE="${ROOT}/docker-compose.yml"
elif [[ -f "${ROOT}/../docker-compose.yml" ]]; then
  COMPOSE_DIR="$(cd "${ROOT}/.." && pwd)"
  COMPOSE_FILE="${COMPOSE_DIR}/docker-compose.yml"
else
  echo "ERROR: Cannot find docker-compose.yml next to this launcher." >&2
  exit 1
fi

UI_URL="http://localhost:6080/vnc.html?autoconnect=true&resize=remote"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed." >&2
  echo "Install Docker Engine, then re-run this script:" >&2
  echo "  https://docs.docker.com/engine/install/" >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not running, or your user cannot access it." >&2
  echo "Try: sudo systemctl start docker" >&2
  echo "And ensure your user is in the docker group." >&2
  exit 1
fi

ARCH="$(uname -m)"
case "${ARCH}" in
  aarch64|arm64) export DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/arm64}" ;;
  *)             export DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}" ;;
esac

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

if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "${UI_URL}" >/dev/null 2>&1 || true
elif command -v sensible-browser >/dev/null 2>&1; then
  sensible-browser "${UI_URL}" >/dev/null 2>&1 || true
fi

echo
echo "MorphAgent UI is running."
echo "  Browser: ${UI_URL}"
echo "  Stop:    docker compose -f \"${COMPOSE_FILE}\" down"
echo
echo "Tip: Home → Load a previous run → completed_demo_run (no API key needed)."
