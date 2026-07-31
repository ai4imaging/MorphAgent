#!/usr/bin/env bash
# Build three platform “click-to-run” Docker zip packages (local only; does not push).
#
# Output:
#   docker/dist/MorphAgent-UI-Docker-macOS.zip
#   docker/dist/MorphAgent-UI-Docker-Windows.zip
#   docker/dist/MorphAgent-UI-Docker-Linux.zip
#
# Optional:
#   --with-image   also docker-build + docker-save into each zip (much larger)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HANDOFF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKER_DIR="${HANDOFF_ROOT}/docker"
DIST_DIR="${DOCKER_DIR}/dist"
STAGE_ROOT="${DIST_DIR}/_stage"
WITH_IMAGE=0
IMAGE_NAME="${MORPHAGENT_DOCKER_IMAGE:-morphagent-ui:latest}"
TAG_DATE="$(date +%Y%m%d)"

for arg in "$@"; do
  case "${arg}" in
    --with-image) WITH_IMAGE=1 ;;
    -h|--help)
      echo "Usage: bash scripts/build_docker_packages.sh [--with-image]"
      exit 0
      ;;
    *)
      echo "Unknown argument: ${arg}" >&2
      exit 1
      ;;
  esac
done

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: missing required command: $1" >&2
    exit 1
  }
}

need zip
need rsync

rm -rf "${DIST_DIR}"
mkdir -p "${DIST_DIR}" "${STAGE_ROOT}"

write_package_dockerfile() {
  local dest="$1"
  cat > "${dest}" <<'EOF'
# MorphAgent UI — browser-accessible desktop (noVNC)
# Build from this package root:
#   docker compose up -d --build
FROM continuumio/miniconda3:24.9.2-0

ENV DEBIAN_FRONTEND=noninteractive \
    DISPLAY=:1 \
    QT_X11_NO_MITSHM=1 \
    LIBGL_ALWAYS_SOFTWARE=1 \
    QT_QPA_PLATFORM=xcb \
    MORPHAGENT_INSTALL_ALLEN=0 \
    MORPHAGENT_ENV_NAME=morphagent \
    MORPHAGENT_SANDBOX_ENV_NAME=morphagent_sandbox \
    NO_VNC_PORT=6080 \
    VNC_PORT=5900 \
    RESOLUTION=1920x1080x24

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        fluxbox \
        x11vnc \
        xvfb \
        novnc \
        websockify \
        libgl1 \
        libglib2.0-0 \
        libxkbcommon-x11-0 \
        libdbus-1-3 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-shape0 \
        libxcb-xinerama0 \
        libxcb-xfixes0 \
        libfontconfig1 \
        libxrender1 \
        libxi6 \
        libsm6 \
        libxext6 \
        libjpeg-dev \
        zlib1g-dev \
        fonts-dejavu-core \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/share/novnc/vnc.html /usr/share/novnc/index.html

WORKDIR /opt/MorphAgent_UI

COPY MorphAgent /opt/MorphAgent_UI/MorphAgent
COPY dependencies /opt/MorphAgent_UI/dependencies
COPY scripts /opt/MorphAgent_UI/scripts

RUN bash scripts/setup.sh \
    && conda clean -afy \
    && find /opt/conda -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

COPY entrypoint.sh /usr/local/bin/morphagent-entrypoint.sh
RUN chmod +x /usr/local/bin/morphagent-entrypoint.sh

VOLUME ["/data"]
EXPOSE 6080 5900

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=5 \
    CMD curl -fsS "http://127.0.0.1:${NO_VNC_PORT}/" >/dev/null || exit 1

ENTRYPOINT ["/usr/local/bin/morphagent-entrypoint.sh"]
EOF
}

write_readme() {
  local dest="$1"
  local platform="$2"
  local starter="$3"
  cat > "${dest}" <<EOF
MorphAgent UI — Docker one-click package (${platform})
=====================================================

Requirements
------------
- Docker Desktop (macOS / Windows) or Docker Engine (Linux)
- ~8 GB RAM, ~10 GB free disk for the first image build

How to start
------------
1. Install and start Docker.
2. Double-click / run: ${starter}
3. Wait for the first build (several minutes), then your browser opens to:
   http://localhost:6080/vnc.html?autoconnect=true&resize=remote

Quick review without an API key
-------------------------------
In the UI: Home → Load a previous run →
  MorphAgent/demo/data/results/completed_demo_run

Stop
----
  docker compose down

Persist API keys
----------------
Place a .env file into the named Docker volume mount path /data/.env
(or configure credentials in the UI Configure page).

Built: ${TAG_DATE}
EOF
}

stage_common() {
  local stage="$1"
  mkdir -p "${stage}"
  write_package_dockerfile "${stage}/Dockerfile"
  cp "${DOCKER_DIR}/entrypoint.sh" "${stage}/entrypoint.sh"
  cp "${DOCKER_DIR}/docker-compose.package.yml" "${stage}/docker-compose.yml"
  chmod +x "${stage}/entrypoint.sh"

  rsync -a --delete \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    --exclude '.env' \
    --exclude 'logs/' \
    "${HANDOFF_ROOT}/MorphAgent/" "${stage}/MorphAgent/"

  rsync -a --delete \
    --exclude '.DS_Store' \
    "${HANDOFF_ROOT}/dependencies/" "${stage}/dependencies/"

  mkdir -p "${stage}/scripts"
  cp "${HANDOFF_ROOT}/scripts/setup.sh" "${stage}/scripts/setup.sh"
  cp "${HANDOFF_ROOT}/scripts/verify_install.py" "${stage}/scripts/verify_install.py"
  chmod +x "${stage}/scripts/setup.sh"
}

IMAGE_TAR=""
if [[ "${WITH_IMAGE}" -eq 1 ]]; then
  need docker
  echo "Building Docker image ${IMAGE_NAME} (this can take a long time)…"
  docker build -f "${DOCKER_DIR}/Dockerfile" -t "${IMAGE_NAME}" "${HANDOFF_ROOT}"
  IMAGE_TAR="${DIST_DIR}/morphagent-ui-image.tar"
  echo "Saving image to ${IMAGE_TAR}…"
  docker save -o "${IMAGE_TAR}" "${IMAGE_NAME}"
fi

maybe_copy_image() {
  local stage="$1"
  if [[ -n "${IMAGE_TAR}" && -f "${IMAGE_TAR}" ]]; then
    cp "${IMAGE_TAR}" "${stage}/morphagent-ui-image.tar"
    cat > "${stage}/load-image.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"
docker load -i morphagent-ui-image.tar
echo "Loaded morphagent-ui image. You can now run the platform starter."
EOF
    cat > "${stage}/load-image.bat" <<'EOF'
@echo off
cd /d "%~dp0"
docker load -i morphagent-ui-image.tar
echo Loaded morphagent-ui image. You can now run MorphAgent-UI.bat
pause
EOF
    chmod +x "${stage}/load-image.sh"
  fi
}

pack_one() {
  local name="$1"
  local starter_src="$2"
  local starter_name="$3"
  local platform_label="$4"
  local stage="${STAGE_ROOT}/${name}"

  echo "Packaging ${name}…"
  rm -rf "${stage}"
  stage_common "${stage}"
  cp "${starter_src}" "${stage}/${starter_name}"
  chmod +x "${stage}/${starter_name}" 2>/dev/null || true
  write_readme "${stage}/README.txt" "${platform_label}" "${starter_name}"
  maybe_copy_image "${stage}"

  local zip_path="${DIST_DIR}/${name}.zip"
  rm -f "${zip_path}"
  (
    cd "${STAGE_ROOT}"
    zip -r -q "${zip_path}" "${name}"
  )
  echo "  → ${zip_path}"
}

pack_one "MorphAgent-UI-Docker-macOS" \
  "${DOCKER_DIR}/mac/MorphAgent-UI.command" "MorphAgent-UI.command" "macOS"

pack_one "MorphAgent-UI-Docker-Windows" \
  "${DOCKER_DIR}/win/MorphAgent-UI.bat" "MorphAgent-UI.bat" "Windows"

pack_one "MorphAgent-UI-Docker-Linux" \
  "${DOCKER_DIR}/linux/MorphAgent-UI.sh" "MorphAgent-UI.sh" "Linux"

# Keep staging for inspection; comment out to save disk:
rm -rf "${STAGE_ROOT}"
if [[ -n "${IMAGE_TAR}" && -f "${IMAGE_TAR}" ]]; then
  rm -f "${IMAGE_TAR}"
fi

echo
echo "Done. Packages written to: ${DIST_DIR}"
ls -lh "${DIST_DIR}"/*.zip
echo
echo "Upload these three zip files to a GitHub Release (do not git-push the zips),"
echo "then keep the download links in README_UI.md."
