"""Map the locked 25 HSC feature names onto columns of the MorphAgent table.

The discovery table (`final_figures/agent.csv`) is 134 cells × 674 features.
Column names are MorphAgent-generated synonyms, not the Supplementary list-2
slugs. Matching is name-based: exact (after mito/mitochondria normalisation),
then a hand-ranked synonym list, then unused columns. Each agent column is
used at most once.

Cell selection: keep rows that are finite on the 25 matched columns. If that
is still larger than the paper n=110, further drop cells that have any NA in
the original 674-d matrix (QC incomplete). We do not randomly subsample.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# paper name -> preferred agent.csv columns, in rank order
PREFERRED: Dict[str, List[str]] = {
    "mitochondria_background_signal_heterogeneity": [
        "mitochondria_background_intensity_ratio",
        "mito_intensity_heterogeneity_std",
        "mito_intensity_heterogeneity_inter",
        "mito_local_contrast_background",
    ],
    "mito_perimeter_intensity_gradient_mean": [
        "mito_local_contrast_mean",
        "mito_contrast_local_mean",
        "mito_perimeter_to_area_ratio_mean",
        "mitochondria_perimeter_to_area_ratio_mean",
    ],
    "background_mito_signal_to_noise": [
        "vlm_background_noise_score",
        "vlm_mito_background_noise_estimate",
        "vlm_background_noise_level",
        "vlm_mito_background_noise_level",
    ],
    "mitochondria_neighbor_intensity_autocorrelation": [
        "mito_texture_correlation_glcm",
        "mito_glcm_correlation_mean",
        "mitochondria_texture_glcm_correlation",
        "mito_texture_glcm_correlation",
        "mito_correlation_glcm",
    ],
    "mitochondria_texture_anisotropy_glcm": [
        "mito_texture_contrast_glcm",
        "mito_glcm_contrast_mean",
        "mitochondria_texture_glcm_contrast",
        "mito_texture_glcm_contrast",
        "mito_texture_energy_glcm",
        "mito_spatial_anisotropy",
    ],
    "mitochondria_long_tubule_fraction": [
        "mito_elongated_fraction",
        "mito_elongation_fraction",
        "vlm_mito_tubularity_percentage",
        "vlm_mito_tubularity_visual_score",
    ],
    "mitochondrial_area_fraction": [
        "mito_area_fraction",
        "mito_total_area_fraction",
        "total_mitochondria_area_fraction",
        "mito_area_fraction_occupancy",
    ],
    "mitochondria_local_autocorrelation_length": [
        "mito_texture_glcm_correlation",
        "mito_haralick_correlation_mean",
        "mito_glcm_correlation_masked",
        "mito_texture_correlation_mean",
    ],
    "mito_intensity_autocorrelation_length": [
        "mito_texture_correlation_mean",
        "mito_haralick_correlation_mean",
        "mito_glcm_correlation_mean",
        "mito_texture_correlation_glcm",
    ],
    "mitochondria_to_cell_intensity_dynamic_range_ratio": [
        "mito_intensity_cv",
        "mitochondrion_intensity_cv",
        "mito_pixel_intensity_cov",
        "mito_intensity_cv_global",
    ],
    "background_noise_robust_snr": [
        "vlm_mito_background_noise_estimate",
        "vlm_mito_background_noise_level",
        "vlm_background_cleanness_score",
        "vlm_image_background_quality",
        "vlm_background_noise_level",
    ],
    "mito_intensity_geodesic_variation": [
        "mito_inter_object_intensity_variation",
        "mito_intensity_heterogeneity_inter_object",
        "mito_intensity_cv_between",
        "mito_intensity_cv_global",
    ],
    "mitochondria_local_entropy_mean": [
        "mito_texture_local_entropy_mean",
        "mito_texture_entropy_mean",
        "mito_haralick_entropy_mean",
        "mean_mitochondria_internal_texture_entropy",
    ],
    "mitochondrial_radial_anisotropy_index": [
        "mito_spatial_anisotropy",
        "mito_radial_distribution_entropy",
        "mito_radial_distribution_std",
        "mito_orientation_coherence",
    ],
    "mitochondria_network_continuity_score": [
        "mito_network_connectivity_ratio",
        "mitochondrial_network_connectivity_ratio",
        "mito_network_connectivity_index",
        "vlm_mitochondrial_network_connectivity_score",
        "vlm_mito_network_connectivity_score",
    ],
    "mitochondria_length_width_skewness": [
        "mitochondria_size_skewness",
        "mito_area_skewness",
        "skewness_mitochondria_area",
        "mito_std_aspect_ratio",
        "mito_aspect_ratio_std",
    ],
    "mitochondria_intensity_entropy": [
        "mito_intensity_entropy",
        "mito_intensity_entropy_masked",
        "mito_global_intensity_entropy",
    ],
    "mito_clustered_vs_isolated_object_fraction": [
        "mito_clustering_nnd_mean",
        "mito_largest_object_fraction",
        "fragmented_mitochondria_fraction",
    ],
    "mitochondria_intensity_gini_inside_cell": [
        "mito_area_gini_coefficient",
        "mito_size_gini_coefficient",
        "mitochondria_size_gini_coefficient",
        "mito_intensity_cv",
    ],
    "mitochondria_area_fraction_in_perinuclear_band": [
        "vlm_mito_perinuclear_concentration_score",
        "vlm_mito_perinuclear_clustering_score",
        "vlm_mitochondrial_perinuclear_clustering",
        "vlm_perinuclear_clustering_score",
        "vlm_mito_perinuclear_distribution",
        "mito_area_fraction_occupancy",
    ],
    "mitochondria_intensity_entropy_within_cell": [
        "mito_intensity_entropy_within_mask",
        "mito_intensity_entropy_masked",
        "mito_mask_entropy",
        "mito_global_intensity_entropy",
    ],
    "mitochondrial_clusteredness_by_nearest_neighbor": [
        "mito_nearest_neighbor_cv",
        "mito_nearest_neighbor_dist_cv",
        "mito_clustering_nnd_mean",
        "mito_nearest_neighbor_std",
        "mean_nearest_neighbor_distance",
    ],
    "mitochondrial_network_fragmentation_score": [
        "mito_fragmentation_index",
        "mitochondria_fragmentation_index",
        "mito_fragmentation_factor",
        "vlm_mito_fragmentation_score",
        "fragmented_mitochondria_fraction",
    ],
    "mitochondrial_network_compactness": [
        "vlm_mitochondria_network_class",
        "vlm_mito_network_class_visual",
        "vlm_fused_network_dominance",
        "mito_convex_hull_fill_ratio",
        "mito_network_fill_ratio",
    ],
    "mitochondrial_neighbor_intensity_variogram_slope": [
        "mito_haralick_contrast_mean",
        "mito_glcm_contrast_mean",
        "mito_texture_contrast_glcm",
        "mitochondrial_haralick_contrast",
        "mito_contrast_glcm",
    ],
}

PAPER_N = 110


def _norm(name: str) -> str:
    s = str(name).lower().strip()
    s = re.sub(r"^vlm_", "", s)
    s = re.sub(r"(mitochondria|mitochondrial|mitochondrion)", "mito", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def infer_age(sample_id: pd.Series) -> pd.Series:
    sid = sample_id.astype(str)
    out = pd.Series(index=sid.index, dtype=object)
    out[sid.str.contains("old|aged", case=False)] = "Aged"
    out[sid.str.contains("young", case=False)] = "Young"
    return out


def match_columns(
    paper_names: Sequence[str],
    agent_columns: Sequence[str],
) -> pd.DataFrame:
    """Return one row per paper name: matched agent column and how it was found."""
    cols = [c for c in agent_columns if c != "sample_id"]
    by_norm: Dict[str, List[str]] = {}
    for c in cols:
        by_norm.setdefault(_norm(c), []).append(c)

    used = set()
    rows = []
    for name in paper_names:
        nn = _norm(name)
        chosen = None
        how = None
        exact = [c for c in by_norm.get(nn, []) if c not in used]
        if exact:
            # prefer non-vlm if both exist
            non_vlm = [c for c in exact if not str(c).startswith("vlm_")]
            chosen = (non_vlm or exact)[0]
            how = "exact_name"
        if chosen is None:
            for cand in PREFERRED.get(name, []):
                if cand in cols and cand not in used:
                    chosen = cand
                    how = "synonym"
                    break
        if chosen is None:
            how = "unmatched"
        if chosen is not None:
            used.add(chosen)
        rows.append(
            {
                "paper_name": name,
                "agent_column": chosen,
                "match": how,
                "exact_norm": nn,
            }
        )
    return pd.DataFrame(rows)


def select_cells(
    agent: pd.DataFrame,
    feature_cols: Sequence[str],
    target_n: int = PAPER_N,
) -> Tuple[pd.Index, str]:
    """Select cells corresponding to the discovery set.

    1. Keep rows with finite values on the matched 25 features.
    2. If still more than `target_n`, drop cells that have any NA in the
       original MorphAgent matrix (QC incomplete). 134 → 113 in the shipped table.
    """
    X = agent.loc[:, list(feature_cols)].apply(pd.to_numeric, errors="coerce")
    finite_25 = np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
    idx = agent.index[finite_25]
    note = f"finite on 25 matched features: {int(finite_25.sum())}/{len(agent)}"

    if len(idx) <= target_n:
        return idx, note

    feat_all = agent.drop(columns=["sample_id"], errors="ignore")
    complete_all = ~feat_all.isna().any(axis=1)
    idx2 = agent.index[finite_25 & complete_all.to_numpy()]
    note += (
        f"; complete on full {feat_all.shape[1]}-d matrix: "
        f"{int((finite_25 & complete_all.to_numpy()).sum())}"
    )
    if len(idx2) == 0:
        return idx, note
    return idx2, note


def build_discovery_table(
    agent: pd.DataFrame,
    paper_names: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    mapping = match_columns(paper_names, agent.columns)
    missing = mapping["agent_column"].isna()
    if missing.any():
        raise ValueError(
            "Unmatched paper features: "
            + ", ".join(mapping.loc[missing, "paper_name"])
        )
    feat_cols = mapping["agent_column"].tolist()
    idx, note = select_cells(agent, feat_cols)
    sub = agent.loc[idx].copy()
    out = pd.DataFrame({"sample_id": sub["sample_id"].astype(str).values})
    out["age"] = infer_age(out["sample_id"]).values
    for paper, col in zip(mapping["paper_name"], feat_cols):
        out[paper] = pd.to_numeric(sub[col], errors="coerce").values
    return out, mapping, note
