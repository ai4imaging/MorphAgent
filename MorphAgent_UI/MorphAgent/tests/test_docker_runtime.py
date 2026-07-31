from __future__ import annotations

import re
import unittest
from pathlib import Path


UI_ROOT = Path(__file__).resolve().parents[2]
DOCKER_DIR = UI_ROOT / "docker"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class DockerRuntimeContractTests(unittest.TestCase):
    def test_compose_is_local_only_and_persists_state_and_workspace(self) -> None:
        compose = read(DOCKER_DIR / "docker-compose.yml")

        self.assertIn("platform: linux/amd64", compose)
        self.assertIn(
            '127.0.0.1:${MORPHAGENT_NOVNC_PORT:-6080}:6080',
            compose,
        )
        self.assertNotIn(":5900", compose)
        self.assertIn("${MORPHAGENT_STATE_DIR:-../docker-data}:/data", compose)
        self.assertIn("${MORPHAGENT_WORKSPACE:-../workspace}:/workspace", compose)
        self.assertIn("no-new-privileges:true", compose)
        self.assertIn("cap_drop:", compose)
        self.assertIn("- ALL", compose)
        self.assertNotIn(
            "\n    init: true",
            compose,
            "The image already uses tini as PID 1; Compose must not wrap it again",
        )

    def test_image_builds_docker_environments_without_host_setup_script(self) -> None:
        dockerfile = read(DOCKER_DIR / "Dockerfile")
        installer = read(DOCKER_DIR / "install-environments.sh")

        self.assertFalse(
            dockerfile.startswith("# syntax="),
            "Use Docker Desktop's built-in parser instead of downloading a frontend image",
        )
        self.assertNotIn("FROM --platform=linux/amd64", dockerfile)
        self.assertNotIn("scripts/setup.sh", dockerfile)
        self.assertIn("docker/install-environments.sh", dockerfile)
        self.assertIn("docker/healthcheck.sh", dockerfile)
        self.assertIn("ARG INSTALL_ALLEN=1", dockerfile)
        self.assertIn("ARG NOVNC_VERSION=1.5.0", dockerfile)
        self.assertIn("ARG WEBSOCKIFY_VERSION=0.12.0", dockerfile)
        self.assertNotIn("\n        novnc \\", dockerfile)
        self.assertNotIn("\n        websockify \\", dockerfile)
        self.assertIn("verify_install.py", dockerfile)
        self.assertIn('ENTRYPOINT ["tini", "--"', dockerfile)
        self.assertIn('INSTALL_COMPONENT="ui"', dockerfile)
        self.assertIn('INSTALL_COMPONENT="sandbox"', dockerfile)
        self.assertIn('INSTALL_COMPONENT="allen"', dockerfile)
        self.assertGreaterEqual(
            dockerfile.count("install-morphagent-environments"),
            4,
            "Each Conda environment must have its own reusable Docker layer",
        )

        self.assertIn('conda create -y -n "${UI_ENV}"', installer)
        self.assertIn('conda create -y -n "${SANDBOX_ENV}"', installer)
        self.assertIn('conda create -y -n "${ALLEN_ENV}"', installer)
        self.assertIn('INSTALL_COMPONENT="${INSTALL_COMPONENT:-all}"', installer)
        self.assertIn("conda clean -afy", installer)
        self.assertNotIn("conda env create -y", installer)
        self.assertIn("requirements-demo-ui.txt", installer)
        self.assertIn("requirements-sandbox.txt", installer)
        self.assertIn("requirements-allen.txt", installer)

    def test_entrypoint_persists_config_and_demo_results_and_secures_vnc(self) -> None:
        entrypoint = read(DOCKER_DIR / "entrypoint.sh")
        healthcheck = read(DOCKER_DIR / "healthcheck.sh")

        self.assertRegex(entrypoint, r"RESOLUTION.*\^\[0-9\]\+x\[0-9\]\+x")
        self.assertIn('${DATA_ROOT}/config/.env', entrypoint)
        self.assertIn("ln -s", entrypoint)
        self.assertIn("/opt/morphagent-seed/completed_demo_run", entrypoint)
        self.assertIn('${DATA_ROOT}/demo-results', entrypoint)
        self.assertIn("-localhost", entrypoint)
        self.assertIn("websockify", entrypoint)
        self.assertIn("conda run --no-capture-output", entrypoint)
        self.assertIn("/vnc.html", healthcheck)

    def test_legacy_platform_packages_and_generated_dockerfile_are_removed(self) -> None:
        obsolete = (
            DOCKER_DIR / "docker-compose.package.yml",
            DOCKER_DIR / "linux" / "MorphAgent-UI.sh",
            DOCKER_DIR / "mac" / "MorphAgent-UI.command",
            DOCKER_DIR / "win" / "MorphAgent-UI.bat",
            UI_ROOT / "scripts" / "build_docker_packages.sh",
        )

        existing = [str(path.relative_to(UI_ROOT)) for path in obsolete if path.exists()]
        self.assertEqual([], existing)

    def test_documentation_describes_the_single_compose_workflow(self) -> None:
        docker_readme = read(DOCKER_DIR / "README.md")
        ui_readme = read(UI_ROOT / "README_UI.md")
        docs = docker_readme + "\n" + ui_readme

        self.assertIn("docker compose -f docker/docker-compose.yml build", docs)
        self.assertIn("docker compose -f docker/docker-compose.yml up -d", docs)
        self.assertIn("http://127.0.0.1:6080", docs)
        self.assertIn("/workspace", docs)
        self.assertIn("docker-data", docs)
        self.assertNotIn("MorphAgent-UI-Docker-macOS.zip", ui_readme)
        self.assertNotIn("build_docker_packages.sh", ui_readme)


if __name__ == "__main__":
    unittest.main()
