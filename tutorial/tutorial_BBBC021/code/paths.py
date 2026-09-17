"""Resolve the tutorial root and the four public folders.

Works when imported from `code/`, or when the current working directory is
the tutorial root or `notebook/`.
"""
from __future__ import annotations

from pathlib import Path


def tutorial_root() -> Path:
    here = Path(__file__).resolve()
    if here.parent.name == "code":
        return here.parent.parent
    cwd = Path.cwd().resolve()
    if (cwd / "code").is_dir() and (cwd / "source").is_dir():
        return cwd
    if cwd.name == "notebook" and (cwd.parent / "code").is_dir():
        return cwd.parent
    return here.parent.parent


ROOT = tutorial_root()
CODE_DIR = ROOT / "code"
SOURCE_DIR = ROOT / "source"
DATA_DIR = ROOT / "data"
NOTEBOOK_DIR = ROOT / "notebook"

DATASET_DIR = DATA_DIR / "dataset"
OUTPUT_DIR = DATA_DIR / "outputs"
EVAL_DIR = DATA_DIR / "evaluation"

FEATURE_LIBRARY = SOURCE_DIR / "feature_library"
NAMES_CSV = SOURCE_DIR / "morphagent_467_feature_names.csv"
MOA_MAP_CSV = SOURCE_DIR / "drug_moa_source.csv"
VLM_TEMPLATE = SOURCE_DIR / "vlm_scoring.json"
CACHED_RESULTS = SOURCE_DIR / "cached_results"
L1000_PER_METHOD = SOURCE_DIR / "l1000_per_method.csv"
L1000_PAIRWISE = SOURCE_DIR / "l1000_pairwise.csv"
