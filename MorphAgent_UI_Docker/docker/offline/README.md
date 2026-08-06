# Offline Docker image

Use this prebuilt `linux/amd64` image when Docker Hub or package downloads are
unavailable during a normal Docker build.

## Download

Download `morphagent-ui-linux-amd64.tar.gz` from
[Google Drive](https://drive.google.com/file/d/1KGMJLRoipqaFYV5B3TbIMFh6zN-94CW3/view?usp=drive_link),
then place it in this directory:

```text
MorphAgent_UI_Docker/docker/offline/morphagent-ui-linux-amd64.tar.gz
```

Google Drive may show a large-file virus-scan warning. Choose **Download
anyway** to continue.

## Artifact identity

- Docker tag after loading: `morphagent-ui:local`
- Source commit: `fcbe9e7c14a1e7ce56fe8e13e970df83a2ecd1bd`
- Image ID: `sha256:e36fc17580f6276221dee0c3f634cd6bfd983dcab4f852c31906c6eaf03502b7`
- Archive SHA-256: `8853ef423b78fe0cb6ffc28448d1d6f8d019ee19bc2e3dccf9ed2a28064c2272`

The image was built from the repository Dockerfile and passed the embedded
installation and offscreen Qt UI verification during the build.

## Verify, load, and start

From `MorphAgent_UI_Docker/` on macOS or Linux:

```bash
(cd docker/offline && shasum -a 256 -c morphagent-ui-linux-amd64.tar.gz.sha256)
docker load -i docker/offline/morphagent-ui-linux-amd64.tar.gz
mkdir -p docker-data workspace
docker compose -f docker/docker-compose.yml up -d --no-build
```

On Windows PowerShell, verify that this command prints the archive SHA-256
listed above, then load and start:

```powershell
(Get-FileHash .\docker\offline\morphagent-ui-linux-amd64.tar.gz -Algorithm SHA256).Hash
docker load -i .\docker\offline\morphagent-ui-linux-amd64.tar.gz
New-Item -ItemType Directory -Force docker-data, workspace
docker compose -f docker/docker-compose.yml up -d --no-build
```

Open:

```text
http://127.0.0.1:6080/vnc.html?autoconnect=true&resize=scale
```
