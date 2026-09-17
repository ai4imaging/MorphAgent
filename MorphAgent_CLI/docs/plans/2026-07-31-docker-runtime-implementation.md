# MorphAgent Reproducible Docker Runtime Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the upstream Docker package with one reproducible, browser-accessible, persistent, and locally bound MorphAgent runtime.

**Architecture:** A pinned `linux/amd64` Miniconda image creates the UI, code-sandbox, and Allen Conda environments during build. Xvfb/noVNC presents the unchanged PyQt UI in a browser, while bind-mounted state and workspace directories preserve credentials, datasets, and results.

**Tech Stack:** Docker BuildKit, Docker Compose, Conda, Bash, Xvfb, Fluxbox, x11vnc, noVNC/websockify, PyQt5, Python `unittest`.

---

### Task 1: Add failing Docker contract tests

**Files:**
- Create: `MorphAgent_UI_Docker/MorphAgent/tests/test_docker_runtime.py`

**Steps:**
1. Test loopback-only noVNC publishing, persistent `/data` and `/workspace`
   mounts, fixed amd64 platform, and absence of a published VNC port.
2. Test that the Dockerfile never calls `scripts/setup.sh`, installs all three
   environments explicitly, runs the verifier, and uses `tini`.
3. Test that the entrypoint symlinks persistent `.env`, seeds persistent demo
   results, validates resolution, and starts x11vnc in localhost mode.
4. Test that obsolete package Compose/launchers and the Dockerfile generator
   are absent.
5. Run the test and confirm failures against the upstream implementation.

### Task 2: Replace the image build and runtime

**Files:**
- Replace: `MorphAgent_UI_Docker/docker/Dockerfile`
- Create: `MorphAgent_UI_Docker/docker/install-environments.sh`
- Replace: `MorphAgent_UI_Docker/docker/entrypoint.sh`
- Create: `MorphAgent_UI_Docker/docker/healthcheck.sh`

**Steps:**
1. Delete the upstream Docker runtime files covered by replacements.
2. Implement explicit, Docker-only Conda environment creation.
3. Install noVNC/X11 packages and copy only required application inputs.
4. Seed the completed demo output and run the offscreen verifier at build time.
5. Implement persistent config/results initialization and supervised startup.
6. Run contract tests and shell syntax checks.

### Task 3: Replace Compose and remove duplicate packages

**Files:**
- Replace: `MorphAgent_UI_Docker/docker/docker-compose.yml`
- Delete: `MorphAgent_UI_Docker/docker/docker-compose.package.yml`
- Delete: `MorphAgent_UI_Docker/docker/linux/MorphAgent-UI.sh`
- Delete: `MorphAgent_UI_Docker/docker/mac/MorphAgent-UI.command`
- Delete: `MorphAgent_UI_Docker/docker/win/MorphAgent-UI.bat`
- Delete: `MorphAgent_UI_Docker/scripts/build_docker_packages.sh`
- Modify: `MorphAgent_UI_Docker/.dockerignore`
- Modify: `MorphAgent_UI_Docker/.gitignore`

**Steps:**
1. Add loopback-only port publishing and fixed amd64 platform.
2. Add state/workspace bind mounts, healthcheck, init, capability drop, and
   resource-safe shared memory.
3. Remove all generated-package duplication.
4. Ignore runtime state, workspace, secrets, and build output.
5. Run contract tests and `docker compose config`.

### Task 4: Document the one supported workflow

**Files:**
- Create: `MorphAgent_UI_Docker/docker/README.md`
- Modify: `MorphAgent_UI_Docker/README_UI.md`

**Steps:**
1. Document prerequisites, first build, normal start, logs, stop, reset, ports,
   mounts, API persistence, custom data, Apple Silicon emulation, and Allen.
2. Remove references to downloadable platform-specific packages.
3. Add troubleshooting commands with exact expected paths.
4. Run documentation contract tests and search for stale launcher references.

### Task 5: Build and smoke-test

**Files:**
- No new production files.

**Steps:**
1. Run all Docker contract tests and `bash -n` on every new shell script.
2. Run `docker compose config` and inspect the resolved loopback binding and
   mounts.
3. Build the image with `docker compose build --no-cache`.
4. Start the container and wait for healthy status.
5. Request `http://127.0.0.1:6080/` and run the offscreen verifier inside the
   container.
6. Stop the container without deleting persistent state.
7. Run `git diff --check` and review the final diff for secrets or unrelated
   changes.
