#!/usr/bin/env python3
"""Verify MorphAgent UI Lite single-env install (morphagent_lite)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

HANDOFF = Path(__file__).resolve().parents[1]
DEMO = HANDOFF / "MorphAgent" / "demo" / "data"
COMPLETED = DEMO / "results" / "completed_demo_run"


def main() -> int:
    errors: list[str] = []
    for mod in (
        "numpy",
        "pandas",
        "skimage",
        "qtpy",
        "langchain_core",
        "langchain_openai",
        "langgraph",
        "openai",
        "dotenv",
    ):
        try:
            importlib.import_module(mod)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"import {mod}: {exc}")

    try:
        from PyQt5 import QtCore  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        errors.append(f"import PyQt5: {exc}")

    if not (HANDOFF / "MorphAgent" / "launch_ui.py").is_file():
        errors.append("missing MorphAgent/launch_ui.py")
    if not DEMO.is_dir():
        errors.append(f"missing demo data: {DEMO}")
    if not COMPLETED.is_dir():
        errors.append(f"missing completed_demo_run: {COMPLETED}")

    if errors:
        print("[FAIL] MorphAgent UI Lite verification failed:")
        for item in errors:
            print(f"  - {item}")
        return 1

    print("[OK] morphagent_lite imports and demo paths look good")
    print(f"     Python: {sys.executable}")
    print(f"     Demo:   {DEMO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
