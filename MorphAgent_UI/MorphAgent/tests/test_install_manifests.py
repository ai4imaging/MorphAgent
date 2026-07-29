"""Regression tests for dependencies required by supported install paths."""

from __future__ import annotations

import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]


def _declares_package(text: str, package: str) -> bool:
    """Return whether a dependency manifest contains an exact package entry."""
    normalized = package.lower().replace("_", "-")
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("-").strip().strip("\",")
        if not line or line.startswith("#"):
            continue
        name = line.split(maxsplit=1)[0]
        for separator in ("<", ">", "=", "!", "~", "["):
            name = name.split(separator, maxsplit=1)[0]
        if name.lower().replace("_", "-") == normalized:
            return True
    return False


class InstallManifestTests(unittest.TestCase):
    def test_all_default_install_paths_include_a_qt_runtime(self) -> None:
        pyproject = (REPOSITORY / "pyproject.toml").read_text(encoding="utf-8")
        project_dependencies = pyproject.split("[project.optional-dependencies]", maxsplit=1)[0]
        manifests = {
            "editable package": project_dependencies,
            "unified conda environment": (REPOSITORY / "envs" / "environment.yml").read_text(encoding="utf-8"),
            "unified pip requirements": (REPOSITORY / "envs" / "requirements.txt").read_text(encoding="utf-8"),
        }

        for install_path, manifest in manifests.items():
            with self.subTest(install_path=install_path):
                self.assertTrue(_declares_package(manifest, "qtpy"))
                self.assertTrue(_declares_package(manifest, "PyQt5"))


if __name__ == "__main__":
    unittest.main()
