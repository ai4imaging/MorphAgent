"""Resolve the tutorial root and the public folders.

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

FEATURE_LIBRARY = SOURCE_DIR / "feature_library"
NAMES_CSV = SOURCE_DIR / "morphagent_hsc_25_feature_names.csv"
CATALOG_CSV = SOURCE_DIR / "feature_catalog.csv"
VLM_TEMPLATE = SOURCE_DIR / "vlm_scoring.json"
CACHED_RESULTS = SOURCE_DIR / "cached_results"

# Figure 3a reproduction bundle (from HSC_Fig3a_YoungOld_reproduction_*.zip)
FIG3A_BUNDLE = SOURCE_DIR / "fig3a_bundle"
FIG3A_INPUT = FIG3A_BUNDLE / "input"
FIG3A_SCRIPTS = FIG3A_BUNDLE / "scripts"
FIG3A_ZIP = ROOT.parent / "HSC-pipeline" / "HSC_Fig3a_YoungOld_reproduction_20260916.zip"

FEATURES_CODE_CSV = FIG3A_INPUT / "features_code_filtered_old_plus_20210624_Young.csv"
FEATURES_MANUAL_CSV = FIG3A_INPUT / "features_manuel_old_plus_20210624_Young.csv"
OMICS_TSV = FIG3A_INPUT / "omics_aligned_to_data_test_25_old_plus_20210624_Young.tsv"
SELECTED_FEATURES_CSV = FIG3A_INPUT / "selected_features.csv"
PSEUDOTIME_CSV = FIG3A_INPUT / "cell_pseudotime_selected_params.csv"
PSEUDOTIME_PALETTE = FIG3A_INPUT / "pseudotime_umap_strict_coolwarm_palette.txt"
FEATURES_TRANSFER_CSV = FIG3A_INPUT / "merged_numeric_features_cleaned.csv"
FEATURES_BLURRED_CSV = FIG3A_INPUT / "merged_blurred_features.csv"
FIG3A_SRC_DIR = ROOT.parent / "HSC-pipeline" / "HSC_Fig3a_YoungOld_reproduction_20260916 2"
