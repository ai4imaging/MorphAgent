#!/usr/bin/env python3
"""Verify the desktop workspace and analysis dependencies (morphagent_lite)."""

from __future__ import annotations

import csv
import importlib
import math
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO = REPO_ROOT / "demo" / "data"
DATASET = DEMO / "dataset"
COMPLETED = DEMO / "results" / "completed_demo_run"

REQUIRED_MODULES = (
    "numpy",
    "pandas",
    "scipy",
    "skimage",
    "sklearn",
    "tifffile",
    "mrcfile",
    "PIL",
    "cv2",
    "imageio",
    "tqdm",
    "matplotlib",
    "pypdf",
    "langchain_core",
    "langchain_openai",
    "langgraph",
    "openai",
    "dotenv",
    "socksio",
    "httpx",
)


def check_qt_bindings() -> list[str]:
    """Check both installed bindings without ever loading Qt5 and Qt6 together."""
    checks = (
        ("PySide6 / Qt6 WebEngine", "pyside6", (
            "from PySide6 import QtCore, QtWidgets; "
            "from PySide6.QtWebEngineCore import QWebEngineProfile; "
            "from PySide6.QtWebEngineWidgets import QWebEngineView; "
            "print('Qt6', QtCore.qVersion(), 'WebEngine OK')"
        )),
        ("PyQt5 / legacy Qt UI", "pyqt5", (
            "from PyQt5 import QtCore; import qtpy; "
            "print('Qt5', QtCore.QT_VERSION_STR, 'qtpy OK')"
        )),
    )
    errors = []
    for label, api, code in checks:
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                env=dict(os.environ, QT_API=api),
                capture_output=True, text=True, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{label} dependency check failed: {exc}")
            continue
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
            errors.append(f"{label} dependency check failed: {detail}")
        else:
            print(f"[OK] {result.stdout.strip()}")
    return errors


def main() -> int:
    errors: list[str] = []
    for mod in REQUIRED_MODULES:
        try:
            importlib.import_module(mod)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"import {mod}: {exc}")

    errors.extend(check_qt_bindings())

    for relative in (
        "launch_desktop_ui.py", "launch_ui.py",
        "src/morphagent_ui/desktop_runtime.py", "src/morphagent_ui/desktop_window.py",
        "design-preview/index.html", "design-preview/runtime.js",
    ):
        if not (REPO_ROOT / relative).is_file():
            errors.append(f"missing desktop source: {relative}")
    if not DATASET.is_dir():
        errors.append(f"missing demo dataset: {DATASET}")
    if not (DATASET / "dataset_index.txt").is_file():
        errors.append("missing demo/data/dataset/dataset_index.txt")
    precomputed = REPO_ROOT / "demo" / "precomputed"
    for name in (
        "expert_knowledge_summary.txt",
        "deep_research_summary.txt",
        "rag_knowledge_summary.txt",
    ):
        if not (precomputed / name).is_file():
            errors.append(f"missing precomputed knowledge: {precomputed / name}")
    if not COMPLETED.is_dir():
        errors.append(f"missing completed_demo_run: {COMPLETED}")
    elif (COMPLETED / "features.csv").is_file():
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from morphagent_ui.models import load_feature_cards

        cards = load_feature_cards(COMPLETED)
        if len(cards) != 5:
            errors.append(f"expected 5 completed-run feature cards, found {len(cards)}")
        if any(card.status != "retained" for card in cards):
            errors.append("completed-run feature cards must all be retained")
        with (COMPLETED / "features.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        feature_names = [name for name in (rows[0].keys() if rows else ()) if name != "sample_id"]
        if len(rows) != 10:
            errors.append(f"expected 10 completed-run samples, found {len(rows)}")
        for name in feature_names:
            values = [float(row[name]) for row in rows if row.get(name)]
            if len(values) != len(rows):
                errors.append(f"completed-run feature {name} has missing values")
            elif not all(math.isfinite(value) for value in values):
                errors.append(f"completed-run feature {name} has non-finite values")
            elif len(set(values)) <= 1:
                errors.append(f"completed-run feature {name} has no sample-to-sample variation")
            elif not any(value != 0 for value in values):
                errors.append(f"completed-run feature {name} is all zero")

    if errors:
        print("[FAIL] MorphAgent desktop installation verification failed:")
        for item in errors:
            print(f"  - {item}")
        return 1

    print("[OK] Desktop dependencies, lightweight analysis imports and demo paths look good")
    print(f"     Python: {sys.executable}")
    print(f"     Demo:   {DEMO}")
    print("     Launch: bash scripts/start_ui.sh (Windows: scripts\\start_ui_windows.bat)")
    print("     Checks verify imports and files, not an end-to-end model API run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
