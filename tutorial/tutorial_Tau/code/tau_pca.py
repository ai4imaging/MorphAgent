#!/usr/bin/env python3
"""Figure 4a — MorphAgent versus expert-designed Tau feature space.

Principal-component projection of the paired WT Tau discovery cohort
(n = 58 cells) in two feature spaces:

  * MorphAgent  : the 301 reviewed Tau descriptors (source/feature_lists)
  * Expert      : the 16-feature handcrafted reference panel

Points are coloured and sized by mean cellular Tau intensity (log scale).
Grey 95% covariance ellipses mark the low- and high-intensity halves of the
cohort (median split), which is the visual grouping used in the manuscript.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from paths import (
    CACHED_RESULTS,
    LIST_301,
    OUTPUT_DIR,
    WT_EXPERT,
    WT_MORPHAGENT,
    WT_TAU_INTENSITY,
    rel,
    require,
)

CMAP_BLUE_PURPLE = LinearSegmentedColormap.from_list(
    "blue_purple",
    ["#2166ac", "#67a9cf", "#b8d4e3", "#d8b9e8", "#9970ab", "#5e3c99"],
    N=256,
)

PANEL_TITLES = {
    "morphagent": "MorphAgent feature space",
    "expert": "Expert-designed feature space",
}


def configure_mpl() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "sans-serif"]
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42


def load_feature_list_301() -> List[str]:
    df = pd.read_csv(require(LIST_301))
    return df["feature_name"].astype(str).tolist()


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.fillna(out.median()).fillna(0.0)


def load_wt_cohort() -> Dict[str, pd.DataFrame]:
    """Align the two feature spaces and the Tau intensity readout on sample_id."""
    morph = pd.read_csv(require(WT_MORPHAGENT))
    expert = pd.read_csv(require(WT_EXPERT))
    inten = pd.read_csv(require(WT_TAU_INTENSITY))

    for df in (morph, expert, inten):
        df["sample_id"] = df["sample_id"].astype(str)

    names_301 = load_feature_list_301()
    present = [c for c in names_301 if c in morph.columns]
    if len(present) != len(names_301):
        missing = sorted(set(names_301) - set(present))
        raise KeyError(f"{len(missing)} of the 301 features are absent, e.g. {missing[:5]}")

    common = sorted(
        set(morph["sample_id"]) & set(expert["sample_id"]) & set(inten["sample_id"])
    )
    morph = morph.set_index("sample_id").loc[common]
    expert = expert.set_index("sample_id").loc[common]
    inten = inten.set_index("sample_id").loc[common]

    expert_cols = [c for c in expert.columns]
    return {
        "sample_ids": pd.Series(common, name="sample_id"),
        "morphagent": clean_numeric(morph[present]),
        "expert": clean_numeric(expert[expert_cols]),
        "intensity": pd.to_numeric(inten.iloc[:, 0], errors="coerce"),
    }


def run_pca(features: pd.DataFrame, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Z-score then project. Constant columns are dropped first."""
    keep = features.loc[:, features.std(axis=0) > 0]
    X = StandardScaler().fit_transform(keep.to_numpy(dtype=float))
    pca = PCA(n_components=n_components, random_state=0)
    emb = pca.fit_transform(X)
    return emb, pca.explained_variance_ratio_


def _scale_sizes(values: np.ndarray, base: float = 120.0, scale: float = 600.0) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0 or finite.max() <= finite.min():
        return np.full_like(vals, base + scale / 2.0)
    norm = (vals - finite.min()) / (finite.max() - finite.min())
    norm[~np.isfinite(norm)] = 0.5
    return base + scale * norm


def _confidence_ellipse(ax, x: np.ndarray, y: np.ndarray, n_std: float = 2.0) -> None:
    """Grey covariance ellipse (~95% for a bivariate normal at n_std = 2)."""
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    ax.add_patch(
        Ellipse(
            (float(np.mean(x)), float(np.mean(y))),
            width,
            height,
            angle=angle,
            facecolor="#bdbdbd",
            alpha=0.30,
            edgecolor="#9e9e9e",
            linewidth=0.7,
            zorder=1,
        )
    )


def plot_panel(
    ax,
    emb: np.ndarray,
    evr: np.ndarray,
    intensity: np.ndarray,
    title: str,
    draw_ellipses: bool = True,
):
    raw = np.asarray(intensity, dtype=float)
    log_vals = np.where(np.isfinite(raw) & (raw > 0), np.log1p(raw), np.nan)
    sizes = _scale_sizes(log_vals)

    if draw_ellipses:
        # Median split on Tau intensity -> low / high subpopulation.
        thr = np.nanmedian(raw)
        for mask in (raw <= thr, raw > thr):
            if mask.sum() >= 3:
                _confidence_ellipse(ax, emb[mask, 0], emb[mask, 1])

    positive = raw[np.isfinite(raw) & (raw > 0)]
    norm = mcolors.LogNorm(vmin=positive.min(), vmax=positive.max())
    sc = ax.scatter(
        emb[:, 0],
        emb[:, 1],
        c=raw,
        s=sizes,
        cmap=CMAP_BLUE_PURPLE,
        norm=norm,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )
    ax.set_title(title, pad=10)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.spines[["top", "right"]].set_visible(False)
    return sc


def _median_split_axis(emb: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    thr = np.nanmedian(intensity)
    return emb[intensity > thr].mean(axis=0) - emb[intensity <= thr].mean(axis=0)


def _orient_like(src: np.ndarray, dst: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    """Rotate `src` in the PC1–PC2 plane so its abundance axis matches `dst`.

    PC1 stays on x and PC2 on y. Sign flips are the usual PCA gauge choice so
    the remaining transform is a proper rotation.
    """
    src = np.array(src, dtype=float, copy=True)
    dst = np.array(dst, dtype=float, copy=True)
    inten = np.asarray(intensity, dtype=float)

    if np.corrcoef(src[:, 0], inten)[0, 1] < 0:
        src[:, 0] *= -1
    if np.corrcoef(dst[:, 0], inten)[0, 1] < 0:
        dst[:, 0] *= -1
    if np.sign(np.corrcoef(src[:, 1], inten)[0, 1]) != np.sign(
        np.corrcoef(dst[:, 1], inten)[0, 1]
    ):
        src[:, 1] *= -1

    v_src = _median_split_axis(src, inten)
    v_dst = _median_split_axis(dst, inten)
    theta = np.arctan2(v_dst[1], v_dst[0]) - np.arctan2(v_src[1], v_src[0])
    c, s = np.cos(theta), np.sin(theta)
    return src @ np.array([[c, s], [-s, c]])


def plot_feature_space_comparison(
    cohort: Dict[str, pd.DataFrame],
    out_stem: Optional[Path] = None,
    draw_ellipses: bool = True,
) -> Tuple[Path, pd.DataFrame]:
    configure_mpl()
    out_stem = out_stem or (OUTPUT_DIR / "fig4a_tau_feature_space")
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    intensity = cohort["intensity"].to_numpy(dtype=float)
    embeddings = {}
    evrs = {}
    for key in ("morphagent", "expert"):
        embeddings[key], evrs[key] = run_pca(cohort[key])
    embeddings["morphagent"] = _orient_like(
        embeddings["morphagent"], embeddings["expert"], intensity
    )

    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    sc = None
    for ax, key in zip(axes, ("morphagent", "expert")):
        emb, evr = embeddings[key], evrs[key]
        sc = plot_panel(ax, emb, evr, intensity, PANEL_TITLES[key], draw_ellipses)
        rows.append(
            {
                "feature_space": PANEL_TITLES[key],
                "n_features": int(cohort[key].shape[1]),
                "n_cells": int(cohort[key].shape[0]),
                "pc1_explained_variance": float(evr[0]),
                "pc2_explained_variance": float(evr[1]),
            }
        )

    cbar = fig.colorbar(sc, ax=axes, shrink=0.82, aspect=26, pad=0.02)
    cbar.set_label("Mean cellular Tau intensity", fontsize=12)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {rel(Path(str(out_stem)))}.png/.svg/.pdf")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_stem.parent / "fig4a_pca_summary.csv", index=False)
    return Path(f"{out_stem}.png"), summary


def reference_pca_summary() -> Optional[pd.DataFrame]:
    """Published explained variance for the two panels, for cross-checking."""
    path = CACHED_RESULTS / "pca_explained_variance.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path)


def compare_pca_to_published(summary: pd.DataFrame) -> Optional[pd.DataFrame]:
    ref = reference_pca_summary()
    if ref is None:
        return None
    merged = summary.merge(ref, on="feature_space", suffixes=("", "_ref"))
    for pc in ("pc1", "pc2"):
        merged[f"{pc}_reproduced_pct"] = (merged[f"{pc}_explained_variance"] * 100).round(1)
        merged[f"{pc}_match"] = (
            merged[f"{pc}_reproduced_pct"] - merged[f"{pc}_explained_variance_pct"]
        ).abs() < 0.05
    return merged[
        ["feature_space", "n_features",
         "pc1_reproduced_pct", "pc1_explained_variance_pct", "pc1_match",
         "pc2_reproduced_pct", "pc2_explained_variance_pct", "pc2_match"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-ellipses", action="store_true")
    args = parser.parse_args()
    cohort = load_wt_cohort()
    print(f"WT discovery cohort: {cohort['morphagent'].shape[0]} cells")
    print(f"  MorphAgent features: {cohort['morphagent'].shape[1]}")
    print(f"  Expert features    : {cohort['expert'].shape[1]}")
    _, summary = plot_feature_space_comparison(cohort, draw_ellipses=not args.no_ellipses)
    print(summary.to_string(index=False))
    check = compare_pca_to_published(summary)
    if check is not None:
        print("\nAgainst the published values:")
        print(check.to_string(index=False))


if __name__ == "__main__":
    main()
