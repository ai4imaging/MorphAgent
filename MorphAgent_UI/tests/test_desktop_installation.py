"""Desktop setup/launch contract, without installing packages in the test runner."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT


def requirements(path):
    result = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.startswith("-r "):
            result.extend(requirements(path.parent / line[3:].strip()))
        elif line:
            result.append(line)
    return result


def test_standard_requirements_install_both_desktop_and_legacy_qt():
    packages = requirements(UI / "dependencies/requirements-lite.txt")
    desktop = requirements(UI / "dependencies/requirements-desktop.txt")
    assert desktop and all(package in packages for package in desktop)
    assert "PySide6==6.10.2" in packages
    assert "PyQt5>=5.15,<5.16" in packages


def test_windows_default_launcher_selects_new_desktop():
    launcher = (UI / "scripts/start_ui_windows.ps1").read_text(encoding="utf-8-sig")
    assert 'Join-Path $RepoRoot "launch_desktop_ui.py"' in launcher
    assert 'Join-Path $RepoRoot "launch_ui.py"' not in launcher
    installer = (UI / "scripts/setup_windows.ps1").read_text(encoding="utf-8-sig")
    assert "requirements-lite.txt" in installer
    assert "python -m pip install -r $ReqFile" in installer
    assert "verify_install.py" in installer


@pytest.mark.skipif(shutil.which("bash") is None, reason="Bash setup requires bash")
@pytest.mark.parametrize("fail_install", [False, True])
def test_setup_reuses_environment_and_stops_on_install_error(tmp_path, fail_install):
    # Only the external conda executable is replaced: run the actual shell script.
    repo = tmp_path / "repo with spaces"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(UI / "scripts/setup.sh", scripts / "setup.sh")
    shutil.copytree(UI / "dependencies", repo / "dependencies")
    (scripts / "verify_install.py").write_text("# not executed by this test\n")
    base = tmp_path / "conda base"
    hooks = base / "etc/profile.d"
    hooks.mkdir(parents=True)
    (hooks / "conda.sh").write_text("# No shell hook is needed by the recorder.\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    recorder = bin_dir / "conda"
    recorder.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "with open(os.environ['SETUP_TEST_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps(args) + '\\n')\n"
        "if args == ['info', '--base']:\n"
        "    print(os.environ['SETUP_TEST_BASE'])\n"
        "elif args == ['env', 'list']:\n"
        "    print('test-desktop-env /some/existing/environment')\n"
        "elif 'pip' in args and '-r' in args and os.environ['SETUP_TEST_FAIL'] == '1':\n"
        "    sys.exit(7)\n",
        encoding="utf-8",
    )
    recorder.chmod(0o755)
    log = tmp_path / "commands.jsonl"
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
               MORPHAGENT_ENV_NAME="test-desktop-env", MORPHAGENT_RECREATE_ENVS="0",
               SETUP_TEST_LOG=str(log), SETUP_TEST_BASE=str(base),
               SETUP_TEST_FAIL=str(int(fail_install)))
    result = subprocess.run(["bash", str(scripts / "setup.sh")], cwd=tmp_path,
                            env=env, text=True, capture_output=True, timeout=30)
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    assert not any(cmd[:1] == ["create"] or cmd[:2] == ["env", "remove"] for cmd in commands)
    installs = [cmd for cmd in commands if "pip" in cmd and "-r" in cmd]
    assert installs
    packages = requirements(Path(installs[0][installs[0].index("-r") + 1]))
    assert "PySide6==6.10.2" in packages
    verifies = [cmd for cmd in commands if str(scripts / "verify_install.py") in cmd]
    if fail_install:
        assert result.returncode != 0
        assert not verifies
        assert "setup complete" not in result.stdout.lower()
    else:
        assert result.returncode == 0, result.stderr
        assert len(verifies) == 1
        assert commands.index(installs[0]) < commands.index(verifies[0])
        assert "start_ui.sh" in result.stdout
        for cmd in commands:
            if cmd[0] == "run":
                assert cmd[cmd.index("-n") + 1] == "test-desktop-env"


def load_verifier():
    spec = importlib.util.spec_from_file_location("desktop_verify", UI / "scripts/verify_install.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verifier_checks_webengine_in_a_separate_process(monkeypatch):
    verifier = load_verifier()
    assert hasattr(verifier, "check_qt_bindings"), "Qt6 and Qt5 must be checked in separate processes"
    calls = []

    def capture(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "Qt import OK", "")

    monkeypatch.setattr(verifier.subprocess, "run", capture)
    assert verifier.check_qt_bindings() == []
    assert len(calls) == 2
    assert all(command[0] == sys.executable for command, _ in calls)
    scripts = [command[-1] for command, _ in calls]
    assert any("PySide6.QtWebEngineWidgets" in code for code in scripts)
    assert any("PySide6.QtWebEngineCore" in code for code in scripts)
    assert any("PyQt5" in code for code in scripts)
    assert not any("PyQt5" in code and "PySide6" in code for code in scripts)
    assert "qtpy" not in verifier.REQUIRED_MODULES


@pytest.mark.parametrize("failure", ["missing", "timeout"])
def test_verifier_reports_missing_or_broken_qt(monkeypatch, failure):
    verifier = load_verifier()
    assert hasattr(verifier, "check_qt_bindings"), "Qt dependency checks are missing"

    def broken(command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 30)
        return subprocess.CompletedProcess(command, 1, "", "No module named PySide6")

    monkeypatch.setattr(verifier.subprocess, "run", broken)
    errors = verifier.check_qt_bindings()
    assert len(errors) == 2
    assert any("PySide6" in message for message in errors)
