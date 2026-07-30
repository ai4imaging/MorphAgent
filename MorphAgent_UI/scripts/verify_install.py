#!/usr/bin/env python3
"""Offline structural and UI smoke test for the MorphAgent handoff bundle."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


HANDOFF_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = HANDOFF_ROOT / "MorphAgent"
DEMO_DATA = REPOSITORY / "demo" / "data"
DATASET = DEMO_DATA / "dataset"
COMPLETED_RUN = DEMO_DATA / "results" / "completed_demo_run"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def import_required_modules() -> None:
    modules = (
        "dotenv",
        "qtpy",
        "PyQt5",
        "openai",
        "langchain_core",
        "langchain_openai",
        "langgraph",
        "numpy",
        "pandas",
        "scipy",
        "skimage",
        "tifffile",
        "cv2",
        "mahotas",
    )
    for name in modules:
        importlib.import_module(name)
    print(f"[OK] Imported {len(modules)} required modules")


def check_demo() -> None:
    require((DATASET / "dataset_index.txt").is_file(), "dataset_index.txt is missing")
    samples = sorted(path for path in DATASET.iterdir() if path.is_dir() and not path.name.startswith("."))
    require(len(samples) == 10, f"Expected 10 demo samples, found {len(samples)}")
    for sample in samples:
        require((sample / "image.tif").is_file(), f"Missing image.tif: {sample.name}")
        require(any((sample / "slices").glob("*.png")), f"Missing VLM slice: {sample.name}")
        masks = list((sample / "segmentation").glob("*.tif"))
        require(len(masks) >= 1, f"Missing segmentation masks: {sample.name}")
    require((DEMO_DATA / "expert_knowledge").is_dir(), "Expert knowledge is missing")
    require((DEMO_DATA / "deep_research" / "report.md").is_file(), "Deep research report is missing")
    require(len(list((DEMO_DATA / "RAG").glob("*.pdf"))) == 3, "Expected 3 bundled RAG PDFs")
    print(f"[OK] Demo data: {len(samples)} samples with images, slices, and masks")


def check_completed_run() -> None:
    sys.path.insert(0, str(REPOSITORY))
    from morphagent_ui.models import list_result_artifacts, load_feature_cards

    require((COMPLETED_RUN / "features.csv").is_file(), "Completed run features.csv is missing")
    cards = load_feature_cards(COMPLETED_RUN)
    require(len(cards) == 10, f"Expected 10 feature cards, found {len(cards)}")
    artifacts = list_result_artifacts(COMPLETED_RUN)
    require(bool(artifacts), "No evidence artifacts were discovered")
    print(f"[OK] Completed run: {len(cards)} feature cards, {len(artifacts)} curated artifacts")


def check_optional_segmentation() -> None:
    available = []
    missing = []
    for name in ("torch", "torchvision", "cellpose"):
        try:
            importlib.import_module(name)
            available.append(name)
        except Exception:
            missing.append(name)
    if missing:
        print("[INFO] Optional Cellpose mask-regeneration stack is not installed: " + ", ".join(missing))
    else:
        print("[OK] Optional Cellpose mask-regeneration stack is installed")


def check_allen_environment() -> None:
    """Verify the isolated morphagent_allen env used by the UI default backend."""
    import shutil
    import subprocess

    allen_env = os.environ.get("MORPHAGENT_ALLEN_ENV_NAME", "morphagent_allen")
    if shutil.which("conda") is None:
        print("[WARN] conda not found; cannot verify Allen environment")
        return

    listed = subprocess.run(
        ["conda", "env", "list"],
        capture_output=True,
        text=True,
        check=False,
    )
    names = {line.split()[0] for line in listed.stdout.splitlines() if line.strip() and not line.startswith("#")}
    if allen_env not in names:
        print(
            f"[WARN] Allen env '{allen_env}' is missing. "
            "Custom datasets without masks will skip segmentation. "
            "Install with: bash scripts/setup.sh"
        )
        return

    check_script = REPOSITORY / "segmentation_allen" / "check_installation.py"
    result = subprocess.run(
        ["conda", "run", "-n", allen_env, "python", str(check_script)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(
            f"[WARN] Allen env '{allen_env}' exists but check_installation failed; "
            "custom data without masks will skip segmentation and the run continues."
        )
        if result.stdout:
            print(result.stdout[-800:])
        if result.stderr:
            print(result.stderr[-800:])
        return
    print(f"[OK] Allen environment '{allen_env}' passes check_installation.py")


def check_sandbox_environment() -> None:
    """Verify the isolated morphagent_sandbox env used for feature extract()."""
    import shutil
    import subprocess

    sandbox_env = os.environ.get("MORPHAGENT_SANDBOX_ENV_NAME", "morphagent_sandbox")
    if shutil.which("conda") is None:
        print("[WARN] conda not found; cannot verify code sandbox environment")
        return

    listed = subprocess.run(
        ["conda", "env", "list"],
        capture_output=True,
        text=True,
        check=False,
    )
    names = {line.split()[0] for line in listed.stdout.splitlines() if line.strip() and not line.startswith("#")}
    if sandbox_env not in names:
        print(
            f"[WARN] Code sandbox env '{sandbox_env}' is missing. "
            "UI feature code expects this env. Install with: bash scripts/setup.sh"
        )
        return

    result = subprocess.run(
        [
            "conda",
            "run",
            "-n",
            sandbox_env,
            "python",
            "-c",
            "import numpy, skimage, cv2; print(numpy.__version__, skimage.__version__)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"[WARN] Code sandbox env '{sandbox_env}' exists but imports failed; re-run setup.sh")
        if result.stderr:
            print(result.stderr[-500:])
        return
    print(f"[OK] Code sandbox environment '{sandbox_env}' imports numpy/skimage ({result.stdout.strip()})")


def ui_smoke_test() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.path.insert(0, str(REPOSITORY))
    from launch_ui import create_standalone_window

    application, window, widget = create_standalone_window("home")
    window.show()
    application.processEvents()
    require(widget is not None and window.centralWidget() is widget, "Qt window did not initialize")
    window.close()
    application.processEvents()
    print("[OK] Qt UI initialized Home/Configure/Run/Features/Evidence")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ui-smoke", action="store_true", help="Initialize the Qt window offscreen")
    args = parser.parse_args()

    require(REPOSITORY.is_dir(), f"Repository is missing: {REPOSITORY}")
    import_required_modules()
    check_demo()
    check_completed_run()
    check_optional_segmentation()
    check_allen_environment()
    check_sandbox_environment()
    if args.ui_smoke:
        ui_smoke_test()
    print("[PASS] MorphAgent handoff verification completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
