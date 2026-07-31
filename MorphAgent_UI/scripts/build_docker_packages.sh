#!/usr/bin/env bash
# Build one MorphAgent UI Docker zip package (local only; does not push).
#
# Output:
#   docker/dist/MorphAgent-UI-Docker.zip
#
# Optional:
#   --with-image   also docker-build + docker-save into the zip (much larger)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HANDOFF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKER_DIR="${HANDOFF_ROOT}/docker"
DIST_DIR="${DOCKER_DIR}/dist"
STAGE_ROOT="${DIST_DIR}/_stage"
WITH_IMAGE=0
IMAGE_NAME="${MORPHAGENT_DOCKER_IMAGE:-morphagent-ui:latest}"
TAG_DATE="$(date +%Y%m%d)"
PKG_NAME="MorphAgent-UI-Docker"

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
  # Keep in sync with docker/Dockerfile, but COPY paths are package-root relative.
  cat > "${dest}" <<'EOF'
FROM continuumio/miniconda3:24.9.2-0

ENV DEBIAN_FRONTEND=noninteractive \
    DISPLAY=:1 \
    QT_X11_NO_MITSHM=1 \
    LIBGL_ALWAYS_SOFTWARE=1 \
    CONDA_SOLVER=libmamba \
    MORPHAGENT_INSTALL_ALLEN=0 \
    MORPHAGENT_ENV_NAME=morphagent \
    MORPHAGENT_SANDBOX_ENV_NAME=morphagent_sandbox \
    MORPHAGENT_DOCKER=1 \
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
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    conda install -y -n base -c conda-forge --override-channels conda-libmamba-solver; \
    conda config --set solver libmamba; \
    conda config --remove channels defaults 2>/dev/null || true; \
    conda config --add channels conda-forge; \
    conda config --set channel_priority strict

WORKDIR /opt/MorphAgent_UI

COPY MorphAgent /opt/MorphAgent_UI/MorphAgent
COPY dependencies /opt/MorphAgent_UI/dependencies
COPY scripts /opt/MorphAgent_UI/scripts

RUN set -eux; \
    QT_QPA_PLATFORM=offscreen bash scripts/setup.sh; \
    conda clean -afy; \
    find /opt/conda -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

COPY novnc /opt/morphagent-novnc
COPY fluxbox /opt/morphagent-fluxbox
COPY entrypoint.sh /usr/local/bin/morphagent-entrypoint.sh
RUN set -eux; \
    chmod +x /usr/local/bin/morphagent-entrypoint.sh; \
    cp /opt/morphagent-novnc/morphagent.html /usr/share/novnc/morphagent.html; \
    ln -sfn /usr/share/novnc/morphagent.html /usr/share/novnc/index.html

VOLUME ["/data"]
EXPOSE 6080 5900

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=5 \
    CMD curl -fsS "http://127.0.0.1:${NO_VNC_PORT}/morphagent.html" >/dev/null || exit 1

ENTRYPOINT ["/usr/local/bin/morphagent-entrypoint.sh"]
EOF
}

STAGE="${STAGE_ROOT}/${PKG_NAME}"
echo "Packaging ${PKG_NAME}…"
rm -rf "${STAGE}"
mkdir -p "${STAGE}"

write_package_dockerfile "${STAGE}/Dockerfile"
cp "${DOCKER_DIR}/entrypoint.sh" "${STAGE}/entrypoint.sh"
cp "${DOCKER_DIR}/docker-compose.package.yml" "${STAGE}/docker-compose.yml"
cp "${DOCKER_DIR}/start.sh" "${STAGE}/start.sh"
cp "${DOCKER_DIR}/start.command" "${STAGE}/start.command"
cp "${DOCKER_DIR}/start.bat" "${STAGE}/start.bat"
cp "${DOCKER_DIR}/README.txt" "${STAGE}/README.txt"
chmod +x "${STAGE}/entrypoint.sh" "${STAGE}/start.sh" "${STAGE}/start.command"

rsync -a --delete \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' --exclude '.env' --exclude 'logs/' \
  "${HANDOFF_ROOT}/MorphAgent/" "${STAGE}/MorphAgent/"

rsync -a --delete --exclude '.DS_Store' \
  "${HANDOFF_ROOT}/dependencies/" "${STAGE}/dependencies/"

rsync -a --delete --exclude '.DS_Store' \
  "${DOCKER_DIR}/novnc/" "${STAGE}/novnc/"
rsync -a --delete --exclude '.DS_Store' \
  "${DOCKER_DIR}/fluxbox/" "${STAGE}/fluxbox/"

mkdir -p "${STAGE}/scripts"
cp "${HANDOFF_ROOT}/scripts/setup.sh" "${STAGE}/scripts/setup.sh"
cp "${HANDOFF_ROOT}/scripts/verify_install.py" "${STAGE}/scripts/verify_install.py"
chmod +x "${STAGE}/scripts/setup.sh"

# Stamp build date into README
printf '\nBuilt: %s\n' "${TAG_DATE}" >> "${STAGE}/README.txt"

if [[ "${WITH_IMAGE}" -eq 1 ]]; then
  need docker
  echo "Building Docker image ${IMAGE_NAME}…"
  docker build -f "${DOCKER_DIR}/Dockerfile" -t "${IMAGE_NAME}" "${HANDOFF_ROOT}"
  docker save -o "${STAGE}/morphagent-ui-image.tar" "${IMAGE_NAME}"
  cat > "${STAGE}/load-image.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"
docker load -i morphagent-ui-image.tar
echo "Loaded. Run start.sh / start.command / start.bat"
EOF
  cat > "${STAGE}/load-image.bat" <<'EOF'
@echo off
cd /d "%~dp0"
docker load -i morphagent-ui-image.tar
echo Loaded. Run start.bat
pause
EOF
  chmod +x "${STAGE}/load-image.sh"
fi

ZIP_PATH="${DIST_DIR}/${PKG_NAME}.zip"
rm -f "${ZIP_PATH}"
(
  cd "${STAGE_ROOT}"
  zip -r -q "${ZIP_PATH}" "${PKG_NAME}"
)
rm -rf "${STAGE_ROOT}"

echo
echo "Done: ${ZIP_PATH}"
ls -lh "${ZIP_PATH}"
echo
echo "Upload this zip to a GitHub Release (do not git-push the zip)."
echo "Suggested asset name: MorphAgent-UI-Docker.zip"
echo "Suggested tag:        ui-docker-${TAG_DATE}"
