"""Reproduce the three manuscript HSC PCA panels (Fig. 3a MorphAgent, Fig. 3d).

Methods lock-in (Manuscript, "Aged-cell discrimination task"):
  1. Restrict to finite, non-constant features.
  2. Standardize.
  3. Project to 2D PCA.
  4. Fit L2-regularized logistic regression in that plane.
  5. Report the in-plane AUC and draw the decision boundary.

Default: replay from source/cached_results/*.csv (the locked 25-feature tables).
Those tables are not shipped in this tutorial; notebook 03 will say so.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from paths import AGENT_CSV, CACHED_RESULTS, NAMES_CSV, OUTPUT_DIR, ROOT

YOUNG_COLOR = "#7BA7C7"
AGED_COLOR = "#B06A68"
PAPER_AUC = {
    "discovery": 0.739,
    "validation_confocal": 0.905,
    "validation_blurred": 0.795,
}
PANEL_TITLES = {
    "discovery": "MorphAgent features",
    "validation_confocal": "Spinning-disk confocal",
    "validation_blurred": "Synthetic low-resolution",
}
TABLE_FILES = {
    "discovery": CACHED_RESULTS / "discovery_25features.csv",
    "validation_confocal": CACHED_RESULTS / "validation_confocal_25features.csv",
    "validation_blurred": CACHED_RESULTS / "validation_blurred_25features.csv",
}


def load_25_names() -> List[str]:
    df = pd.read_csv(NAMES_CSV)
    col = "feature_name" if "feature_name" in df.columns else df.columns[0]
    return df[col].astype(str).tolist()


def _normalize_age(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip()
    aged = out.str.lower().isin(["aged", "old", "age"])
    young = out.str.lower().isin(["young", "yng"])
    mapped = pd.Series(index=out.index, dtype=object)
    mapped[aged] = "Aged"
    mapped[young] = "Young"
    return mapped


def infer_age(df: pd.DataFrame) -> pd.Series:
    for col in ("age", "Age", "group", "Group", "label", "Label"):
        if col in df.columns:
            return _normalize_age(df[col])
    sid = df.iloc[:, 0].astype(str)
    mapped = pd.Series(index=df.index, dtype=object)
    mapped[sid.str.contains("old|aged", case=False)] = "Aged"
    mapped[sid.str.contains("young", case=False)] = "Young"
    return mapped


def load_feature_table(path: Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    names = load_25_names()
    missing = [n for n in names if n not in df.columns]
    if missing:
        raise ValueError(
            f"{path.name} is missing {len(missing)} locked features, "
            f"e.g. {missing[:3]}"
        )
    age = infer_age(df)
    keep = age.isin(["Young", "Aged"])
    X = df.loc[keep, names].apply(pd.to_numeric, errors="coerce")
    var_ok = X.var(axis=0) > 0
    X = X.loc[:, var_ok]
    finite = np.isfinite(X.to_numpy()).all(axis=1)
    X = X.loc[finite]
    y = (age.loc[X.index] == "Aged").astype(int).to_numpy()
    labels = age.loc[X.index].to_numpy()
    return X, y, labels


def pca_logistic(X: pd.DataFrame, y: np.ndarray, random_state: int = 0):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.to_numpy(dtype=float))
    pca = PCA(n_components=2, random_state=random_state)
    Z = pca.fit_transform(Xs)
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        random_state=random_state,
    )
    clf.fit(Z, y)
    scores = clf.decision_function(Z)
    auc = float(roc_auc_score(y, scores))
    return Z, clf, auc, pca


def _decision_line(ax, clf, xlim, ylim):
    w = clf.coef_.ravel()
    b = float(clf.intercept_[0])
    if abs(w[1]) < 1e-12:
        x0 = -b / w[0]
        ax.plot([x0, x0], ylim, ls="--", c="k", lw=1.0, zorder=1)
        return
    xs = np.linspace(xlim[0], xlim[1], 200)
    ys = -(w[0] / w[1]) * xs - b / w[1]
    ax.plot(xs, ys, ls="--", c="k", lw=1.0, zorder=1)


def plot_from_table(
    df: pd.DataFrame,
    title: str,
    out_stem: Path,
    legend_loc: str = "lower right",
) -> dict:
    """Standardize → 2D PCA → L2 logistic, then write a Young/Aged panel."""
    tmp = Path(out_stem).with_suffix(".input.csv")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp, index=False)
    X, y, labels = load_feature_table(tmp)
    Z, clf, auc, pca = pca_logistic(X, y)
    plot_panel(Z, labels, clf, auc, title=title, out_stem=out_stem, legend_loc=legend_loc)
    rec = {
        "n_cells": int(len(labels)),
        "n_young": int((labels == "Young").sum()),
        "n_aged": int((labels == "Aged").sum()),
        "n_features_used": int(X.shape[1]),
        "auc": auc,
        "pc1_var": float(pca.explained_variance_ratio_[0]),
        "pc2_var": float(pca.explained_variance_ratio_[1]),
    }
    return rec


def plot_panel(Z, labels, clf, auc, title: str, out_stem: Path, legend_loc="lower right"):
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    young = labels == "Young"
    aged = labels == "Aged"
    ax.scatter(
        Z[young, 0], Z[young, 1],
        s=58, c=YOUNG_COLOR, edgecolors="white", linewidths=0.8,
        alpha=0.9, label="Young", zorder=3,
    )
    ax.scatter(
        Z[aged, 0], Z[aged, 1],
        s=58, c=AGED_COLOR, edgecolors="white", linewidths=0.8,
        alpha=0.9, label="Aged", zorder=3,
    )
    pad = 0.6
    xlim = (Z[:, 0].min() - pad, Z[:, 0].max() + pad)
    ylim = (Z[:, 1].min() - pad, Z[:, 1].max() + pad)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    _decision_line(ax, clf, xlim, ylim)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title, fontsize=13)
    ax.legend(loc=legend_loc, frameon=True, fancybox=False, edgecolor="0.6", fontsize=9)
    ax.text(
        0.02, 0.02, f"AUC={auc:.3f}",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=11,
    )
    ax.tick_params(direction="out")
    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=200)
    fig.savefig(out_stem.with_suffix(".svg"))
    plt.close(fig)


def tables_available() -> Dict[str, bool]:
    return {k: p.is_file() for k, p in TABLE_FILES.items()}


def reproduce_one(key: str, out_dir: Optional[Path] = None) -> dict:
    path = TABLE_FILES[key]
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path.name}. Place the locked 25-feature table in "
            f"{CACHED_RESULTS}."
        )
    X, y, labels = load_feature_table(path)
    Z, clf, auc, pca = pca_logistic(X, y)
    out_dir = out_dir or OUTPUT_DIR
    legend = "lower right" if key == "discovery" else "upper right"
    plot_panel(
        Z, labels, clf, auc,
        title=PANEL_TITLES[key],
        out_stem=out_dir / f"fig3_{key}",
        legend_loc=legend,
    )
    n_young = int((labels == "Young").sum())
    n_aged = int((labels == "Aged").sum())
    rec = {
        "panel": key,
        "n_cells": int(len(labels)),
        "n_young": n_young,
        "n_aged": n_aged,
        "n_features_used": int(X.shape[1]),
        "auc": auc,
        "paper_auc": PAPER_AUC[key],
        "pc1_var": float(pca.explained_variance_ratio_[0]),
        "pc2_var": float(pca.explained_variance_ratio_[1]),
    }
    return rec


def plot_transfer_pair(out_dir: Optional[Path] = None) -> None:
    """Fig. 3d two-panel layout."""
    out_dir = out_dir or OUTPUT_DIR
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.5))
    fig.suptitle("Transfer to an independent spinning-disk confocal dataset", fontsize=13)
    for ax, key in zip(axes, ("validation_confocal", "validation_blurred")):
        X, y, labels = load_feature_table(TABLE_FILES[key])
        Z, clf, auc, _ = pca_logistic(X, y)
        young = labels == "Young"
        aged = labels == "Aged"
        ax.scatter(Z[young, 0], Z[young, 1], s=42, c=YOUNG_COLOR,
                   edgecolors="white", linewidths=0.7, alpha=0.9, label="Young", zorder=3)
        ax.scatter(Z[aged, 0], Z[aged, 1], s=42, c=AGED_COLOR,
                   edgecolors="white", linewidths=0.7, alpha=0.9, label="Aged", zorder=3)
        pad = 0.5
        xlim = (Z[:, 0].min() - pad, Z[:, 0].max() + pad)
        ylim = (Z[:, 1].min() - pad, Z[:, 1].max() + pad)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        _decision_line(ax, clf, xlim, ylim)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(PANEL_TITLES[key], fontsize=12)
        ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="0.6", fontsize=8)
        ax.text(0.02, 0.02, f"AUC={auc:.3f}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=10)
    fig.tight_layout()
    stem = out_dir / "fig3d_transfer"
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)


def reproduce_available(out_dir: Optional[Path] = None) -> pd.DataFrame:
    out_dir = out_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, present in tables_available().items():
        if not present:
            print(f"[skip] {key}: {TABLE_FILES[key].name} not found")
            continue
        rec = reproduce_one(key, out_dir=out_dir)
        rows.append(rec)
        print(f"[ok]   {key}: n={rec['n_cells']}  AUC={rec['auc']:.3f}  (paper {rec['paper_auc']:.3f})")
    if (
        tables_available()["validation_confocal"]
        and tables_available()["validation_blurred"]
    ):
        plot_transfer_pair(out_dir=out_dir)
        print("[ok]   fig3d two-panel written")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "hsc_pca_auc_metrics.csv", index=False)
    return df


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    df = reproduce_available(out_dir=args.out)
    if df.empty:
        print(
            "No cached 25-feature tables under source/cached_results/. "
            "See source/cached_results/README.md."
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
