# MorphAgent Reproducible Docker Runtime Design

## Goal and user experience

MorphAgent should run consistently on Windows, macOS, and Linux without asking
the host to install Conda, Python, Qt, or scientific packages. The host needs
only Docker Desktop/Engine and a browser. One Compose command starts a Linux
container, and the existing PyQt5 desktop appears through noVNC at
`http://127.0.0.1:6080`. The port is deliberately bound to loopback so the UI
is not exposed to the LAN or internet.

The container is fixed to `linux/amd64`. That gives the legacy Python 3.6 Allen
stack one consistent target; Apple Silicon runs it through Docker Desktop's
emulation. This favors reproducibility over native ARM build speed. The first
build is expected to be slow, while subsequent starts reuse the image.

## Image architecture

The image uses a pinned Miniconda base and installs system packages for Xvfb,
Fluxbox, x11vnc, noVNC, Qt/XCB, fonts, and health diagnostics. It does not call
the host-oriented `scripts/setup.sh`. Instead, a Docker-only installer creates
three named Conda environments explicitly:

- `morphagent`: PyQt5, the UI, API clients, and application package;
- `morphagent_sandbox`: the pinned scientific stack used by generated feature
  code;
- `morphagent_allen`: the legacy Python 3.6 Allen segmentation stack.

The build runs the repository's offscreen installation verifier. Runtime uses
`tini` for signal handling and launches Xvfb, Fluxbox, x11vnc, websockify, and
the existing `launch_ui.py`. Port 5900 remains internal; only noVNC port 6080 is
published.

## Persistence and security

Compose bind-mounts `MorphAgent_UI_Docker/docker-data` to `/data` and
`MorphAgent_UI_Docker/workspace` to `/workspace`. `/data/config/.env` is symlinked to
the repository `.env`, so API settings saved in the UI survive container
replacement without entering the image or Git. Demo results are also moved to
persistent state and seeded with `completed_demo_run` on first launch. Custom
datasets and their result directories live under `/workspace`.

The container publishes no raw VNC port, binds noVNC to `127.0.0.1`, drops
Linux capabilities, and enables `no-new-privileges`. Generated feature code is
still untrusted code and can modify the mounted workspace; documentation must
state that users should mount only project data they are willing to expose to
the run.

## Verification

Static contract tests validate the Dockerfile, Compose mounts and loopback
binding, entrypoint persistence, absence of duplicated legacy packages, and
documentation. Shell scripts pass `bash -n`, Compose passes `docker compose
config`, the image must build, and a running container must become healthy.
The final smoke test requests noVNC over HTTP and runs `verify_install.py
--ui-smoke` inside the container. Existing upstream UI test failures are
recorded separately and are outside this change.
