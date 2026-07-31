# MorphAgent Docker runtime

This is the single supported Docker workflow for the MorphAgent Qt UI. The
application runs on a virtual Linux desktop and is displayed locally in a web
browser through noVNC. The host does not need Conda, Python, Qt, or an X server.

## Requirements

- Docker Desktop on Windows/macOS, or Docker Engine with Compose on Linux;
- at least 8 GB RAM and about 12 GB free disk for the first full build;
- an internet connection during the first build;
- an OpenAI-compatible API only when starting a new experiment.

The image is intentionally `linux/amd64` so the legacy Python 3.6 Allen
segmentation environment has one reproducible target. Apple Silicon runs the
image through Docker Desktop emulation, so the first build and segmentation
can be slower than on an x86-64 computer.

## Build and start

Run from `MorphAgent_UI/`:

```bash
mkdir -p docker-data workspace
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up -d
```

Open:

```text
http://127.0.0.1:6080/vnc.html?autoconnect=true&resize=remote
```

The browser endpoint is bound to `127.0.0.1`; it is not exposed to the local
network or internet. Raw VNC port 5900 is never published.

Check status and logs:

```bash
docker compose -f docker/docker-compose.yml ps
docker compose -f docker/docker-compose.yml logs -f morphagent-ui
```

Stop without deleting data:

```bash
docker compose -f docker/docker-compose.yml down
```

Normal restarts do not need another build:

```bash
docker compose -f docker/docker-compose.yml up -d
```

## Persistent directories

| Host path | Container path | Contents |
|---|---|---|
| `docker-data/` | `/data` | API `.env`, UI preferences, logs, and demo results |
| `workspace/` | `/workspace` | Custom datasets and their generated results |

Put custom data below `MorphAgent_UI/workspace/`, then select `/workspace` (or
one of its child folders) from the Configure page. The selected folder must
contain the normal `dataset/<sample>/image.tif` structure. Generated results
remain in the same mounted workspace and therefore survive container updates.

The UI saves API settings through a symlink to
`docker-data/config/.env`. This file is ignored by Git and is never copied into
the image. Do not share it or commit it.

The bundled `completed_demo_run` is seeded automatically into
`docker-data/demo-results/` on first start. It can be opened from Home without
an API key.

## Configuration

Optional environment variables can be set before Compose commands:

| Variable | Default | Meaning |
|---|---:|---|
| `MORPHAGENT_NOVNC_PORT` | `6080` | Local browser port |
| `MORPHAGENT_RESOLUTION` | `1920x1080x24` | Virtual desktop size/depth |
| `MORPHAGENT_VNC_PASSWORD` | empty | Optional VNC session password |
| `MORPHAGENT_STATE_DIR` | `../docker-data` | Persistent state bind mount |
| `MORPHAGENT_WORKSPACE` | `../workspace` | Dataset/results bind mount |
| `MORPHAGENT_INSTALL_ALLEN` | `1` | Build legacy Allen segmentation env |

For example, use local port 6081:

```bash
MORPHAGENT_NOVNC_PORT=6081 docker compose -f docker/docker-compose.yml up -d
```

To build a smaller review-only image without Allen:

```bash
MORPHAGENT_INSTALL_ALLEN=0 docker compose -f docker/docker-compose.yml build --no-cache
```

The bundled demo already has masks. Custom datasets without masks need the
default full image (`MORPHAGENT_INSTALL_ALLEN=1`).

## Verification and troubleshooting

After startup, `docker compose ... ps` should report `healthy`. To run the
repository verifier inside the container:

```bash
docker compose -f docker/docker-compose.yml exec morphagent-ui \
  conda run --no-capture-output -n morphagent \
  python /opt/MorphAgent_UI/scripts/verify_install.py --ui-smoke
```

Useful local logs are written to `docker-data/logs/` (`ui.log`, `novnc.log`,
`x11vnc.log`, `xvfb.log`, and `fluxbox.log`).

- Port already in use: set `MORPHAGENT_NOVNC_PORT` to another port.
- UI page opens but remains blank: inspect `docker-data/logs/ui.log` and the
  Compose logs.
- Build fails while downloading packages: rerun the build on a stable network;
  a successfully built image is reused on later starts.
- Resetting `docker-data/` deletes saved API configuration and demo results.
  Back it up before removing it.

Generated feature code executes inside the container but can modify mounted
files. Mount only datasets and project files that the run is allowed to access.
