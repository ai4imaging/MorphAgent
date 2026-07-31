# MorphAgent UI: reproducible Linux desktop exposed only through local noVNC.
# Build context must be MorphAgent_UI/:
#   docker build --platform linux/amd64 -f docker/Dockerfile .
FROM continuumio/miniconda3:24.9.2-0

ARG INSTALL_ALLEN=1
ARG NOVNC_VERSION=1.5.0
ARG WEBSOCKIFY_VERSION=0.12.0

ENV DEBIAN_FRONTEND=noninteractive \
    DISPLAY=:1 \
    QT_X11_NO_MITSHM=1 \
    QT_QPA_PLATFORM=xcb \
    LIBGL_ALWAYS_SOFTWARE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MORPHAGENT_ENV_NAME=morphagent \
    MORPHAGENT_SANDBOX_ENV_NAME=morphagent_sandbox \
    MORPHAGENT_ALLEN_ENV_NAME=morphagent_allen \
    CONDA_BASE_PATH=/opt/conda \
    CONDA_ENV=morphagent_sandbox \
    SEGMENTATION_CONDA_ENV=morphagent_allen \
    SEGMENTATION_BACKEND=allen \
    NO_VNC_PORT=6080 \
    VNC_PORT=5900 \
    RESOLUTION=1920x1080x24

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        dbus-x11 \
        fluxbox \
        fonts-dejavu-core \
        fonts-liberation \
        libdbus-1-3 \
        libfontconfig1 \
        libgl1 \
        libglib2.0-0 \
        libjpeg-dev \
        libsm6 \
        libx11-xcb1 \
        libxcb-cursor0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-shape0 \
        libxcb-xfixes0 \
        libxcb-xinerama0 \
        libxext6 \
        libxi6 \
        libxkbcommon-x11-0 \
        libxrender1 \
        procps \
        tini \
        x11vnc \
        xauth \
        xvfb \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /usr/share/novnc \
    && curl --fail --location --retry 5 --retry-all-errors \
        "https://github.com/novnc/noVNC/archive/refs/tags/v${NOVNC_VERSION}.tar.gz" \
        | tar -xz --strip-components=1 -C /usr/share/novnc \
    && python -m pip install --no-cache-dir "websockify==${WEBSOCKIFY_VERSION}" \
    && ln -sfn /usr/share/novnc/vnc.html /usr/share/novnc/index.html

WORKDIR /opt/MorphAgent_UI

# Dependency manifests are copied first so application-only changes reuse the
# expensive environment layer.
COPY dependencies/ /opt/MorphAgent_UI/dependencies/
COPY MorphAgent/envs/ /opt/MorphAgent_UI/MorphAgent/envs/
COPY MorphAgent/segmentation_allen/ /opt/MorphAgent_UI/MorphAgent/segmentation_allen/
COPY docker/install-environments.sh /usr/local/bin/install-morphagent-environments

RUN chmod 0755 /usr/local/bin/install-morphagent-environments

# Keep each environment in a separate cacheable layer. If an optional backend
# fails, Docker can reuse the already completed UI and sandbox environments.
RUN INSTALL_COMPONENT="ui" \
    /usr/local/bin/install-morphagent-environments
RUN INSTALL_COMPONENT="sandbox" \
    /usr/local/bin/install-morphagent-environments
RUN INSTALL_COMPONENT="allen" INSTALL_ALLEN="${INSTALL_ALLEN}" \
    /usr/local/bin/install-morphagent-environments

# Copy the complete application only after the reusable dependency layer.
COPY MorphAgent/ /opt/MorphAgent_UI/MorphAgent/
COPY scripts/verify_install.py /opt/MorphAgent_UI/scripts/verify_install.py
COPY docker/entrypoint.sh /usr/local/bin/morphagent-entrypoint
COPY docker/healthcheck.sh /usr/local/bin/morphagent-healthcheck

RUN chmod 0755 \
        /usr/local/bin/morphagent-entrypoint \
        /usr/local/bin/morphagent-healthcheck \
    && conda run --no-capture-output -n morphagent \
        python -m pip install --no-cache-dir --no-deps /opt/MorphAgent_UI/MorphAgent \
    && mkdir -p /opt/morphagent-seed \
    && cp -a \
        /opt/MorphAgent_UI/MorphAgent/demo/data/results/completed_demo_run \
        /opt/morphagent-seed/completed_demo_run \
    && QT_QPA_PLATFORM=offscreen conda run --no-capture-output -n morphagent \
        python /opt/MorphAgent_UI/scripts/verify_install.py --ui-smoke \
    && conda clean -afy \
    && find /opt/conda -type d -name '__pycache__' -prune -exec rm -rf '{}' +

VOLUME ["/data", "/workspace"]
EXPOSE 6080

HEALTHCHECK --interval=20s --timeout=5s --start-period=90s --retries=10 \
    CMD ["/usr/local/bin/morphagent-healthcheck"]

ENTRYPOINT ["tini", "--", "/usr/local/bin/morphagent-entrypoint"]
