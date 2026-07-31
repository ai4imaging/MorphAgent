# MorphAgent UI — browser-accessible desktop (noVNC)
# Build from MorphAgent_UI/:
#   docker build -f docker/Dockerfile -t morphagent-ui:latest .
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

# Create UI + sandbox conda envs (Allen skipped — demo ships masks).
RUN bash scripts/setup.sh \
    && conda clean -afy \
    && find /opt/conda -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

COPY docker/entrypoint.sh /usr/local/bin/morphagent-entrypoint.sh
RUN chmod +x /usr/local/bin/morphagent-entrypoint.sh

# Persist user API config / results outside the image when mounted.
VOLUME ["/data"]

EXPOSE 6080 5900

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=5 \
    CMD curl -fsS "http://127.0.0.1:${NO_VNC_PORT}/" >/dev/null || exit 1

ENTRYPOINT ["/usr/local/bin/morphagent-entrypoint.sh"]
