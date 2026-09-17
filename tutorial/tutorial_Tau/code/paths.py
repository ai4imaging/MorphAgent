"""Resolve the tutorial root and the public folders.

Works when imported from `code/`, or when the current working directory is
the tutorial root.
"""
from __future__ import annotations

from pathlib import Path


def tutorial_root() -> Path:
    here = Path(__file__).resolve()
    if here.parent.name == "code":
        return here.parent.parent
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "code").is_dir() and (candidate / "source").is_dir():
            return candidate
    return here.parent.parent


ROOT = tutorial_root()
CODE_DIR = ROOT / "code"
SOURCE_DIR = ROOT / "source"
DATA_DIR = ROOT / "data"

TABLE_DIR = DATA_DIR / "tables"
OUTPUT_DIR = DATA_DIR / "outputs"

FEATURE_LISTS = SOURCE_DIR / "feature_lists"
CACHED_RESULTS = SOURCE_DIR / "cached_results"
SPLIT_DIR = SOURCE_DIR / "splits"
HEATMAP_DIR = SOURCE_DIR / "heatmap"

# Curated feature lists shipped with the tutorial.
LIST_301 = FEATURE_LISTS / "feature_list_301_classification.csv"
LIST_400 = FEATURE_LISTS / "feature_list_400_prediction.csv"

# Figure 4a — paired WT Tau discovery cohort (n = 58 cells).
WT_MORPHAGENT = TABLE_DIR / "wt_morphagent_301_features.csv"
WT_EXPERT = TABLE_DIR / "wt_expert_16_features.csv"
WT_TAU_INTENSITY = TABLE_DIR / "wt_mean_cell_tau_intensity.csv"

# Figure 4d / 4i — mutant cohort with paired super-resolution and wide-field views.
MUTANT_MORPHAGENT = TABLE_DIR / "mutant_morphagent_features.csv"
MUTANT_EXPERT = TABLE_DIR / "mutant_expert_features.csv"

# Figure 5a — morphology to transcriptome prediction.
PRED_MORPHAGENT = TABLE_DIR / "predict_morphagent_400_features.csv"
PRED_EXPERT = TABLE_DIR / "predict_expert_features.csv"
PRED_TRANSCRIPTOME = TABLE_DIR / "predict_transcriptome.csv"

# Figure 5d / 5e — morphology–gene hierarchy heatmaps.
HEATMAP_MORPHOLOGY_CLASSES = HEATMAP_DIR / "morphology_classes.csv"
HEATMAP_GO_TERMS = HEATMAP_DIR / "go_terms.csv"
HEATMAP_ARCHITECTURES = HEATMAP_DIR / "architectures.csv"
HEATMAP_FEATURE_GENES = HEATMAP_DIR / "feature_genes.csv"
HEATMAP_GRN_MODULES = HEATMAP_DIR / "grn_modules.csv"
CACHED_PANELD_COUNTS = CACHED_RESULTS / "panelD_shared_gene_counts.csv"

TAU_INTENSITY_FEATURE = "whole_cell_average_intensity"


def rel(path: Path) -> str:
    """Path relative to the tutorial root, for printing."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {rel(path)}. Open notebook/reproduce_tau_experiments.ipynb from the tutorial root."
        )
    return path
