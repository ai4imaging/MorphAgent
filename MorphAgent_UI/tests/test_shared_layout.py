"""The UI installer, launcher, and analysis use one repository-root source tree."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_demo_and_verifier_resolve_shared_root():
    spec = importlib.util.spec_from_file_location(
        "ui_verify", ROOT / "scripts/verify_install.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from morphagent_ui.widgets.home import bundled_demo_results_dir

    assert module.REPO_ROOT == ROOT
    assert module.COMPLETED.is_dir()
    assert bundled_demo_results_dir() == module.COMPLETED
    assert (ROOT / "launch_ui.py").is_file()
    assert not (ROOT / "MorphAgent").exists()


@pytest.mark.skipif(shutil.which("bash") is None, reason="Bash launcher requires bash")
@pytest.mark.parametrize("launcher", ["start_ui.sh", "start_desktop_ui.sh"])
def test_shell_launcher_works_from_another_directory(tmp_path, launcher):
    fake = tmp_path / "conda"
    fake.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n', encoding="utf-8")
    fake.chmod(0o755)
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
               MORPHAGENT_ENV_NAME="custom-ui-env")
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / launcher), "--port", "8767"],
        cwd=tmp_path, env=env, text=True, capture_output=True, check=True,
    )
    args = result.stdout.splitlines()
    assert args == ["run", "--no-capture-output", "-n", "custom-ui-env", "python",
                    str(ROOT / "launch_desktop_ui.py"), "--port", "8767"]


def test_ui_pipeline_uses_shared_main_and_lightweight_knowledge():
    from morphagent_ui.models import RunConfig
    config = RunConfig()
    assert Path(config.repository_root) == ROOT
    assert str(ROOT / "main.py") in config.build_command()
    assert config.pipeline_environment()["MORPHAGENT_KNOWLEDGE_MODE"] == "lite"


def test_paper_knowledge_is_available_after_move():
    knowledge = ROOT / "src/morphagent_ui/reviewer_knowledge/knowledge.json"
    assert json.loads(knowledge.read_text(encoding="utf-8"))["chunks"]
