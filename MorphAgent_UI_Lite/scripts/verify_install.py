#!/usr/bin/env python3
"""Verify MorphAgent UI Lite minimal install (morphagent_lite)."""

from __future__ import annotations

import csv
import importlib
import math
import sys
from pathlib import Path

HANDOFF = Path(__file__).resolve().parents[1]
DEMO = HANDOFF / "MorphAgent" / "demo" / "data"
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
    "qtpy",
    "langchain_core",
    "langchain_openai",
    "langgraph",
    "openai",
    "dotenv",
    "socksio",
    "httpx",
)


def main() -> int:
    errors: list[str] = []
    for mod in REQUIRED_MODULES:
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
    if not DATASET.is_dir():
        errors.append(f"missing demo dataset: {DATASET}")
    if not (DATASET / "dataset_index.txt").is_file():
        errors.append("missing demo/data/dataset/dataset_index.txt")
    precomputed = HANDOFF / "MorphAgent" / "demo" / "precomputed"
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
        sys.path.insert(0, str(HANDOFF / "MorphAgent"))
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
        print("[FAIL] MorphAgent UI Lite verification failed:")
        for item in errors:
            print(f"  - {item}")
        return 1

    print("[OK] morphagent_lite minimal imports and demo paths look good")
    print(f"     Python: {sys.executable}")
    print(f"     Demo:   {DEMO}")
    print("     Scope:  Tau demo + Code/VLM; knowledge via precomputed txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
