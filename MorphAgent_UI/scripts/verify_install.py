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
    require(len(samples) == 5, f"Expected 5 demo samples, found {len(samples)}")
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
        print("[INFO] Optional mask regeneration is not installed: " + ", ".join(missing))
    else:
        print("[OK] Optional Cellpose mask-regeneration stack is installed")


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
    if args.ui_smoke:
        ui_smoke_test()
    print("[PASS] MorphAgent handoff verification completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
