#!/usr/bin/env python3

import argparse
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable, Optional

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-codex"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


GROUP_COLORS = {
    "Young": "#5c8ab8",
    "Old": "#9e1b29",
}


def infer_group_from_sample_id(sample_id: str) -> Optional[str]:
    text = str(sample_id).strip()
    if "young" in text.lower():
        return "Young"
    if "old" in text.lower():
        return "Old"
    return None


def parse_subset_sizes(values: Optional[Iterable[int]]) -> list[int]:
    if values is None:
        return [2, 3, 5, 8, 10, 15, 20, 25, 30]
    return sorted({int(v) for v in values if int(v) >= 2})


def read_input_table(input_path: Path) -> pd.DataFrame:
    suffix = str(input_path).lower()
    sep = "\t" if suffix.endswith((".tsv", ".tab", ".txt")) else ","
    return pd.read_csv(input_path, sep=sep)


def make_unique_names(names: Iterable[str]) -> list[str]:
    seen = {}
    unique = []
    for raw_name in names:
        name = str(raw_name).strip()
        if not name:
            name = "feature"
        count = seen.get(name, 0)
        seen[name] = count + 1
        unique.append(name if count == 0 else f"{name}__dup{count + 1}")
    return unique


def gene_by_cell_to_feature_table(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.shape[1] < 3:
        raise ValueError("gene-by-cell input must contain one feature column and at least two cell columns.")
    feature_col = raw.columns[0]
    feature_names = make_unique_names(raw[feature_col].astype(str))
    value_table = raw.drop(columns=[feature_col]).copy()
    df = value_table.transpose()
    df.columns = feature_names
    df.insert(0, "sample_id", df.index.astype(str))
    return df.reset_index(drop=True)


def load_feature_table(features_csv: Path) -> pd.DataFrame:
    raw = read_input_table(features_csv)
    if "sample_id" in raw.columns:
        df = raw.copy()
    else:
        df = gene_by_cell_to_feature_table(raw)
    if "sample_id" not in df.columns:
        raise ValueError("input table must contain sample_id cells or a gene-by-cell matrix.")
    df = df.copy()
    df["Group"] = df["sample_id"].map(infer_group_from_sample_id)
    df = df[df["Group"].isin(["Young", "Old"])].reset_index(drop=True)
    if df.empty:
        raise ValueError("No Young/Old cells were found in sample_id.")
    return df


def build_numeric_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    feature_df = df.drop(columns=["sample_id", "Group"], errors="ignore").copy()
    for col in feature_df.columns:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")

    keep_cols = []
    for col in feature_df.columns:
        values = feature_df[col].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if finite.sum() == 0:
            continue
        if np.nanstd(values) <= 1e-12:
            continue
        keep_cols.append(col)

    feature_df = feature_df[keep_cols].copy()
    if feature_df.shape[1] == 0:
        raise ValueError("No usable numeric feature columns remain after filtering.")
    return feature_df


def rank_features_by_variance(X_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in X_df.columns:
        values = pd.to_numeric(X_df[feature], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        if finite.sum() == 0:
            variance = -np.inf
            mean_value = np.nan
        else:
            finite_values = values[finite]
            median_value = float(np.median(finite_values))
            filled = values.copy()
            filled[~finite] = median_value
            variance = float(np.var(filled, ddof=0))
            mean_value = float(np.mean(filled))
        rows.append({"feature": feature, "variance": variance, "mean": mean_value})

    return (
        pd.DataFrame(rows)
        .sort_values(["variance", "feature"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )


def filter_top_variable_features(X_df: pd.DataFrame, top_n: Optional[int]) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if top_n is None or int(top_n) <= 0:
        return X_df, None
    ranked = rank_features_by_variance(X_df)
    keep_features = ranked.head(min(int(top_n), len(ranked)))["feature"].tolist()
    return X_df[keep_features].copy(), ranked


def read_pseudotime_table(pseudotime_csv: Optional[Path]) -> Optional[pd.DataFrame]:
    if pseudotime_csv is None:
        return None
    pseudotime_csv = Path(pseudotime_csv)
    if not pseudotime_csv.exists():
        raise FileNotFoundError(f"pseudotime CSV does not exist: {pseudotime_csv}")
    table = pd.read_csv(pseudotime_csv)
    cell_col = next((col for col in ["cell_id", "sample_id", "cell", "Cell"] if col in table.columns), None)
    time_col = next((col for col in ["pseudotime", "Pseudotime", "omics_pseudotime"] if col in table.columns), None)
    if cell_col is None or time_col is None:
        raise ValueError("pseudotime CSV must contain cell_id/sample_id and pseudotime columns.")
    out = table[[cell_col, time_col]].copy()
    out.columns = ["cell_id", "omics_pseudotime"]
    out["cell_id"] = out["cell_id"].astype(str)
    out["omics_pseudotime"] = pd.to_numeric(out["omics_pseudotime"], errors="coerce")
    out = out.dropna(subset=["cell_id"]).drop_duplicates(subset=["cell_id"], keep="first")
    return out


def attach_pseudotime(scores: pd.DataFrame, pseudotime_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if pseudotime_df is None:
        return scores
    out = scores.merge(pseudotime_df, on="cell_id", how="left")
    if out["omics_pseudotime"].notna().sum() == 0:
        raise ValueError("No PCA cell IDs matched pseudotime cell IDs.")
    return out


def read_palette_colors(palette_path: Optional[Path]) -> list[str]:
    if palette_path is None:
        return ["#313695", "#4575B4", "#74ADD1", "#FFFFBF", "#FDAE61", "#D73027", "#A50026"]
    palette_path = Path(palette_path)
    if not palette_path.exists():
        raise FileNotFoundError(f"pseudotime palette file does not exist: {palette_path}")
    text = palette_path.read_text(encoding="utf-8")
    inline_match = re.search(r"(?m)^hex_colors_inline:\s*(.+)$", text)
    if inline_match:
        colors = re.findall(r"#[0-9A-Fa-f]{6}", inline_match.group(1))
    else:
        colors = []
        in_block = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped == "hex_colors:":
                in_block = True
                continue
            if in_block:
                if not stripped:
                    break
                if not stripped.startswith("#"):
                    break
                colors.extend(re.findall(r"#[0-9A-Fa-f]{6}", stripped))
        if not colors:
            colors = re.findall(r"#[0-9A-Fa-f]{6}", text)
    if not colors:
        raise ValueError(f"No hex colors were found in palette file: {palette_path}")
    return colors


def preprocess_features(X_df: pd.DataFrame) -> tuple[np.ndarray, SimpleImputer, StandardScaler]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imputed = imputer.fit_transform(X_df)
    X_scaled = scaler.fit_transform(X_imputed)
    return X_scaled.astype(float), imputer, scaler


def fit_pca_projection(X_subset: np.ndarray, random_state: int) -> np.ndarray:
    if X_subset.shape[1] == 1:
        return np.column_stack([X_subset[:, 0], np.zeros(X_subset.shape[0], dtype=float)])
    pca = PCA(n_components=2, random_state=int(random_state))
    return pca.fit_transform(X_subset)


def impute_raw_two_feature_projection(X_df: pd.DataFrame) -> np.ndarray:
    if X_df.shape[1] != 2:
        raise ValueError("--use-raw-two-features-as-pca requires exactly two usable numeric features.")
    imputer = SimpleImputer(strategy="median")
    return imputer.fit_transform(X_df).astype(float)


def make_linear_classifier(random_state: int) -> LogisticRegression:
    return LogisticRegression(
        penalty="l2",
        solver="liblinear",
        C=1.0,
        class_weight="balanced",
        max_iter=5000,
        random_state=int(random_state),
    )


def make_raw_two_feature_classifier(random_state: int, raw_two_features_classifier: str):
    mode = str(raw_two_features_classifier).strip().lower()
    if mode == "linear":
        return make_linear_classifier(random_state)
    if mode == "poly3":
        # Nonlinear classifier on the same 2D coordinates (coords unchanged).
        return Pipeline(
            [
                ("poly", PolynomialFeatures(degree=3, include_bias=False)),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    penalty="l2",
                    solver="liblinear",
                    C=0.3,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=int(random_state),
                )),
            ]
        )
    raise ValueError(f"Unknown raw_two_features_classifier: {raw_two_features_classifier}")


def fit_boundary_and_score(
    coords: np.ndarray,
    y: np.ndarray,
    random_state: int,
    raw_two_features_classifier: str = "linear",
) -> dict:
    clf = make_raw_two_feature_classifier(random_state, raw_two_features_classifier)
    clf.fit(coords, y)
    pred = clf.predict(coords)
    proba = clf.predict_proba(coords)[:, 1]
    if hasattr(clf, "decision_function"):
        decision = clf.decision_function(coords)
    else:
        decision = proba - 0.5
    return {
        "clf": clf,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "pred": pred,
        "proba": proba,
        "decision": decision,
    }


def rank_features_by_fscore(X_scaled: np.ndarray, y: np.ndarray, feature_names: list[str], top_n: int) -> pd.DataFrame:
    scores, pvals = f_classif(X_scaled, y)
    scores = np.nan_to_num(scores, nan=-np.inf, neginf=-np.inf, posinf=np.inf)
    pvals = np.nan_to_num(pvals, nan=1.0)
    ranked = pd.DataFrame(
        {
            "feature": feature_names,
            "f_score": scores,
            "p_value": pvals,
        }
    ).sort_values(["f_score", "feature"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    return ranked.head(int(top_n)).reset_index(drop=True)


def greedy_forward_search(
    X_scaled: np.ndarray,
    y: np.ndarray,
    ranked_candidates: pd.DataFrame,
    feature_to_index: dict[str, int],
    search_max_features: int,
    random_state: int,
) -> pd.DataFrame:
    selected = []
    remaining = ranked_candidates["feature"].tolist()
    rows = []
    max_steps = min(int(search_max_features), len(remaining))

    for step in range(1, max_steps + 1):
        best_feature = None
        best_stats = None
        for feature in remaining:
            subset = selected + [feature]
            idx = [feature_to_index[name] for name in subset]
            coords = fit_pca_projection(X_scaled[:, idx], random_state)
            stats = fit_boundary_and_score(coords, y, random_state)
            candidate = {
                "step": step,
                "feature": feature,
                "subset_size": len(subset),
                "balanced_accuracy": stats["balanced_accuracy"],
                "roc_auc": stats["roc_auc"],
            }
            if best_stats is None or (
                candidate["balanced_accuracy"],
                candidate["roc_auc"],
                feature,
            ) > (
                best_stats["balanced_accuracy"],
                best_stats["roc_auc"],
                best_stats["feature"],
            ):
                best_feature = feature
                best_stats = candidate

        selected.append(best_feature)
        remaining.remove(best_feature)
        rows.append(best_stats)

    return pd.DataFrame(rows)


def evaluate_subset_cv(
    X_df: pd.DataFrame,
    y: np.ndarray,
    subset_features: list[str],
    random_state: int,
    cv_splits: int,
    cv_repeats: int,
    use_raw_two_features_as_pca: bool = False,
    raw_two_features_classifier: str = "linear",
) -> dict:
    min_class_size = int(min(np.bincount(y)))
    effective_splits = min(int(cv_splits), min_class_size)
    if effective_splits < 2:
        raise ValueError("Need at least 2 samples in each class for cross-validation.")

    cv = RepeatedStratifiedKFold(
        n_splits=effective_splits,
        n_repeats=int(cv_repeats),
        random_state=int(random_state),
    )
    ba_scores = []
    auc_scores = []
    X_subset = X_df[subset_features].copy()

    for train_idx, test_idx in cv.split(X_subset, y):
        X_train = X_subset.iloc[train_idx]
        X_test = X_subset.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        if use_raw_two_features_as_pca:
            if X_train.shape[1] != 2:
                raise ValueError("--use-raw-two-features-as-pca requires exactly two usable numeric features.")
            train_coords = imputer.fit_transform(X_train).astype(float)
            test_coords = imputer.transform(X_test).astype(float)
        else:
            X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train))
            X_test_scaled = scaler.transform(imputer.transform(X_test))

            if X_train_scaled.shape[1] == 1:
                train_coords = np.column_stack([X_train_scaled[:, 0], np.zeros(X_train_scaled.shape[0])])
                test_coords = np.column_stack([X_test_scaled[:, 0], np.zeros(X_test_scaled.shape[0])])
            else:
                pca = PCA(n_components=2, random_state=int(random_state))
                train_coords = pca.fit_transform(X_train_scaled)
                test_coords = pca.transform(X_test_scaled)

        if use_raw_two_features_as_pca:
            clf = make_raw_two_feature_classifier(int(random_state), raw_two_features_classifier)
        else:
            clf = make_linear_classifier(int(random_state))
        clf.fit(train_coords, y_train)
        pred = clf.predict(test_coords)
        proba = clf.predict_proba(test_coords)[:, 1]
        ba_scores.append(balanced_accuracy_score(y_test, pred))
        auc_scores.append(roc_auc_score(y_test, proba))

    return {
        "balanced_accuracy": float(np.mean(ba_scores)),
        "balanced_accuracy_std": float(np.std(ba_scores, ddof=0)),
        "roc_auc": float(np.mean(auc_scores)),
        "roc_auc_std": float(np.std(auc_scores, ddof=0)),
        "cv_splits": effective_splits,
        "cv_repeats": int(cv_repeats),
    }


def fit_full_subset_projection(
    df: pd.DataFrame,
    X_df: pd.DataFrame,
    subset_features: list[str],
    random_state: int,
    use_raw_two_features_as_pca: bool = False,
    raw_two_features_classifier: str = "linear",
) -> dict:
    X_subset = X_df[subset_features].copy()
    if use_raw_two_features_as_pca:
        coords = impute_raw_two_feature_projection(X_subset)
    else:
        X_scaled, imputer, scaler = preprocess_features(X_subset)
        coords = fit_pca_projection(X_scaled, random_state)
    y = np.where(df["Group"].eq("Young"), 1, 0).astype(int)
    model_mode = raw_two_features_classifier if use_raw_two_features_as_pca else "linear"
    stats = fit_boundary_and_score(coords, y, random_state, raw_two_features_classifier=model_mode)

    scores = pd.DataFrame(
        {
            "cell_id": df["sample_id"].astype(str),
            "Group": df["Group"].astype(str),
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            "decision_score": stats["decision"],
            "probability_young": stats["proba"],
            "predicted_group": np.where(stats["pred"] == 1, "Young", "Old"),
        }
    )

    return {
        "scores": scores,
        "clf": stats["clf"],
        "full_balanced_accuracy": stats["balanced_accuracy"],
        "full_roc_auc": stats["roc_auc"],
    }


def is_linear_2d_classifier(clf) -> bool:
    if not hasattr(clf, "coef_") or not hasattr(clf, "intercept_"):
        return False
    coef = np.asarray(clf.coef_)
    if coef.ndim != 2 or coef.shape[0] < 1:
        return False
    return int(coef.shape[1]) == 2


def draw_boundary(ax, clf, x_limits: tuple[float, float], y_limits: tuple[float, float]) -> None:
    if is_linear_2d_classifier(clf):
        coef = clf.coef_[0]
        intercept = float(clf.intercept_[0])
        if abs(float(coef[1])) < 1e-12:
            x0 = -intercept / float(coef[0])
            ax.axvline(x0, color="black", linestyle="--", linewidth=1.2)
        else:
            xs = np.linspace(x_limits[0], x_limits[1], 200)
            ys = -(intercept + float(coef[0]) * xs) / float(coef[1])
            ax.plot(xs, ys, color="black", linestyle="--", linewidth=1.2)
        return

    # Nonlinear boundary: draw p=0.5 contour in the same coordinate space.
    gx = np.linspace(x_limits[0], x_limits[1], 240)
    gy = np.linspace(y_limits[0], y_limits[1], 240)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    zz = clf.predict_proba(grid)[:, 1].reshape(xx.shape)
    ax.contour(xx, yy, zz, levels=[0.5], colors=["black"], linewidths=1.2, linestyles="--")


def compute_pca_plot_limits(scores: pd.DataFrame, clf) -> tuple[tuple[float, float], tuple[float, float]]:
    x_values = scores["PC1"].to_numpy(dtype=float)
    y_values = scores["PC2"].to_numpy(dtype=float)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    if finite.sum() == 0:
        return (-1.0, 1.0), (-1.0, 1.0)

    xs = list(x_values[finite])
    ys = list(y_values[finite])
    has_linear = is_linear_2d_classifier(clf)
    if has_linear:
        coef = clf.coef_[0]
        intercept = float(clf.intercept_[0])

    x_min = float(np.min(xs))
    x_max = float(np.max(xs))
    x_span = x_max - x_min
    if x_span <= 1e-12:
        x_span = 1.0
    x_pad = x_span * 0.08
    line_x_min = x_min - x_pad
    line_x_max = x_max + x_pad

    if has_linear:
        if abs(float(coef[1])) < 1e-12:
            xs.append(float(-intercept / float(coef[0])))
        else:
            line_ys = [
                -(intercept + float(coef[0]) * line_x_min) / float(coef[1]),
                -(intercept + float(coef[0]) * line_x_max) / float(coef[1]),
            ]
            ys.extend(line_ys)

    x_min = float(np.min(xs))
    x_max = float(np.max(xs))
    y_min = float(np.min(ys))
    y_max = float(np.max(ys))

    x_span = x_max - x_min
    y_span = y_max - y_min
    if x_span <= 1e-12:
        x_span = 1.0
    if y_span <= 1e-12:
        y_span = 1.0

    return (
        (x_min - x_span * 0.08, x_max + x_span * 0.08),
        (y_min - y_span * 0.08, y_max + y_span * 0.08),
    )


def plot_group_scatter(scores: pd.DataFrame, clf, title: str, out_png: Path, out_svg: Path) -> None:
    x_limits, y_limits = compute_pca_plot_limits(scores, clf)
    fig, ax = plt.subplots(figsize=(4, 4))
    for group in ["Young", "Old"]:
        sub = scores.loc[scores["Group"] == group]
        ax.scatter(
            sub["PC1"],
            sub["PC2"],
            s=80,
            c=GROUP_COLORS[group],
            edgecolors="#666666",
            linewidths=0.6,
            alpha=0.8,
            label=group,
        )
    ax.set_xlabel("PCA 1", fontsize=12, fontweight="bold")
    ax.set_ylabel("PCA 2", fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    draw_boundary(ax, clf, x_limits, y_limits)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    fig.tight_layout()
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    fig.savefig(out_svg, dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_pseudotime_scatter(
    scores: pd.DataFrame,
    clf,
    palette_colors: list[str],
    title: str,
    out_png: Path,
    out_svg: Path,
) -> None:
    if "omics_pseudotime" not in scores.columns:
        return
    plot_df = scores.dropna(subset=["omics_pseudotime"]).copy()
    if plot_df.empty:
        return

    x_limits, y_limits = compute_pca_plot_limits(scores, clf)
    cmap = LinearSegmentedColormap.from_list("omics_pseudotime_palette", palette_colors)
    fig, ax = plt.subplots(figsize=(4, 4))
    scatter = ax.scatter(
        plot_df["PC1"],
        plot_df["PC2"],
        s=80,
        c=plot_df["omics_pseudotime"],
        cmap=cmap,
        edgecolors="#666666",
        linewidths=0.5,
        alpha=0.8,
    )
    ax.set_xlabel("PCA 1", fontsize=12, fontweight="bold")
    ax.set_ylabel("PCA 2", fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    draw_boundary(ax, clf, x_limits, y_limits)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    # Keep the PCA axes the same size as the Young/Old plot; put the colorbar
    # outside the axes instead of letting fig.colorbar shrink the scatter panel.
    fig.tight_layout()
    cbar_ax = ax.inset_axes([1.04, 0.0, 0.045, 1.0], transform=ax.transAxes)
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label("Omics pseudotime", fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=9)
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    fig.savefig(out_svg, dpi=320, bbox_inches="tight")
    plt.close(fig)


def save_contact_sheet(plot_paths: list[Path], out_prefix: Path, cols: int = 3) -> None:
    if not plot_paths:
        return
    rows = int(np.ceil(len(plot_paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.4, rows * 4.6))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, png_path in zip(axes.ravel(), plot_paths):
        image = plt.imread(png_path)
        ax.imshow(image)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".svg"), dpi=220, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_all_features_analysis(
    df: pd.DataFrame,
    X_df: pd.DataFrame,
    y: np.ndarray,
    out_dir: Path,
    random_state: int,
    cv_splits: int,
    cv_repeats: int,
    pseudotime_df: Optional[pd.DataFrame] = None,
    palette_colors: Optional[list[str]] = None,
    use_raw_two_features_as_pca: bool = False,
    raw_two_features_classifier: str = "linear",
) -> dict:
    run_dir = out_dir / "all_features"
    run_dir.mkdir(parents=True, exist_ok=True)
    subset_features = list(X_df.columns)

    cv_stats = evaluate_subset_cv(
        X_df=X_df,
        y=y,
        subset_features=subset_features,
        random_state=int(random_state),
        cv_splits=int(cv_splits),
        cv_repeats=int(cv_repeats),
        use_raw_two_features_as_pca=bool(use_raw_two_features_as_pca),
        raw_two_features_classifier=str(raw_two_features_classifier),
    )
    full_fit = fit_full_subset_projection(
        df,
        X_df,
        subset_features,
        int(random_state),
        use_raw_two_features_as_pca=bool(use_raw_two_features_as_pca),
        raw_two_features_classifier=str(raw_two_features_classifier),
    )
    full_fit["scores"] = attach_pseudotime(full_fit["scores"], pseudotime_df)

    pd.DataFrame({"feature": subset_features}).to_csv(run_dir / "selected_features.csv", index=False)
    full_fit["scores"].to_csv(run_dir / "cell_pca_scores.csv", index=False)

    title = (
        f"All {len(subset_features)} features | "
        f"CV bal_acc={cv_stats['balanced_accuracy']:.3f} | "
        f"AUC={cv_stats['roc_auc']:.3f}"
    )
    plot_png = run_dir / "pca12_group_scatter.png"
    plot_svg = run_dir / "pca12_group_scatter.svg"
    plot_group_scatter(full_fit["scores"], full_fit["clf"], title, plot_png, plot_svg)
    pseudotime_plot_png = None
    if pseudotime_df is not None:
        pseudotime_plot_png = run_dir / "pca12_pseudotime_scatter.png"
        plot_pseudotime_scatter(
            full_fit["scores"],
            full_fit["clf"],
            palette_colors or read_palette_colors(None),
            f"All {len(subset_features)} features | Omics pseudotime",
            pseudotime_plot_png,
            run_dir / "pca12_pseudotime_scatter.svg",
        )

    summary = pd.DataFrame(
        [
            {
                "subset_size": int(len(subset_features)),
                "feature_count": int(len(subset_features)),
                "balanced_accuracy": cv_stats["balanced_accuracy"],
                "balanced_accuracy_std": cv_stats["balanced_accuracy_std"],
                "roc_auc": cv_stats["roc_auc"],
                "roc_auc_std": cv_stats["roc_auc_std"],
                "full_balanced_accuracy": full_fit["full_balanced_accuracy"],
                "full_roc_auc": full_fit["full_roc_auc"],
                "output_dir": str(run_dir),
            }
        ]
    )
    summary.to_csv(out_dir / "pca12_search_summary.csv", index=False)

    save_contact_sheet([plot_png], out_dir / "pca12_search_contact_sheet")
    if pseudotime_plot_png is not None:
        save_contact_sheet([pseudotime_plot_png], out_dir / "pca12_pseudotime_contact_sheet")

    return {
        "best_subset_size": int(len(subset_features)),
        "best_balanced_accuracy": float(cv_stats["balanced_accuracy"]),
        "best_roc_auc": float(cv_stats["roc_auc"]),
        "out_dir": str(out_dir),
    }


def run_analysis(
    features_csv: Path,
    out_dir: Path,
    random_state: int = 42,
    prefilter_top_n: int = 80,
    search_max_features: int = 30,
    subset_sizes: Optional[Iterable[int]] = None,
    cv_splits: int = 5,
    cv_repeats: int = 5,
    use_all_features: bool = False,
    pseudotime_csv: Optional[Path] = None,
    pseudotime_palette: Optional[Path] = None,
    top_variable_features: Optional[int] = None,
    ranked_subsets: bool = False,
    use_raw_two_features_as_pca: bool = False,
    raw_two_features_classifier: str = "linear",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_feature_table(Path(features_csv))
    X_df = build_numeric_feature_matrix(df)
    X_df, variance_ranked = filter_top_variable_features(X_df, top_variable_features)
    if variance_ranked is not None:
        variance_ranked.to_csv(out_dir / "variance_ranked_features.csv", index=False)
        variance_ranked.head(X_df.shape[1]).to_csv(out_dir / "top_variable_features.csv", index=False)
    y = np.where(df["Group"].eq("Young"), 1, 0).astype(int)
    feature_names = list(X_df.columns)
    X_scaled, _, _ = preprocess_features(X_df)
    pseudotime_df = read_pseudotime_table(pseudotime_csv)
    palette_colors = read_palette_colors(pseudotime_palette) if pseudotime_df is not None else None

    if use_all_features:
        return run_all_features_analysis(
            df=df,
            X_df=X_df,
            y=y,
            out_dir=out_dir,
            random_state=int(random_state),
            cv_splits=int(cv_splits),
            cv_repeats=int(cv_repeats),
            pseudotime_df=pseudotime_df,
            palette_colors=palette_colors,
            use_raw_two_features_as_pca=bool(use_raw_two_features_as_pca),
            raw_two_features_classifier=str(raw_two_features_classifier),
        )

    ranked = rank_features_by_fscore(X_scaled, y, feature_names, top_n=min(int(prefilter_top_n), len(feature_names)))
    ranked.to_csv(out_dir / "prefilter_ranked_features.csv", index=False)

    if ranked_subsets:
        feature_order = ranked["feature"].tolist()
        ranked_order = ranked.copy()
        ranked_order.insert(0, "step", np.arange(1, len(ranked_order) + 1))
        ranked_order.to_csv(out_dir / "ranked_feature_order.csv", index=False)
    else:
        feature_to_index = {name: idx for idx, name in enumerate(feature_names)}
        greedy = greedy_forward_search(
            X_scaled=X_scaled,
            y=y,
            ranked_candidates=ranked,
            feature_to_index=feature_to_index,
            search_max_features=min(int(search_max_features), len(ranked)),
            random_state=int(random_state),
        )
        greedy.to_csv(out_dir / "greedy_feature_order.csv", index=False)
        feature_order = greedy["feature"].tolist()

    chosen_subset_sizes = [s for s in parse_subset_sizes(subset_sizes) if s <= len(feature_order)]
    if not chosen_subset_sizes:
        raise ValueError("No valid subset sizes remain after feature ranking.")

    summary_rows = []
    plot_paths = []
    pseudotime_plot_paths = []

    for subset_size in chosen_subset_sizes:
        subset_features = feature_order[:subset_size]
        run_dir = out_dir / f"top_{subset_size:03d}_features"
        run_dir.mkdir(parents=True, exist_ok=True)

        cv_stats = evaluate_subset_cv(
            X_df=X_df,
            y=y,
            subset_features=subset_features,
            random_state=int(random_state),
            cv_splits=int(cv_splits),
            cv_repeats=int(cv_repeats),
            use_raw_two_features_as_pca=bool(use_raw_two_features_as_pca),
            raw_two_features_classifier=str(raw_two_features_classifier),
        )
        full_fit = fit_full_subset_projection(
            df,
            X_df,
            subset_features,
            int(random_state),
            use_raw_two_features_as_pca=bool(use_raw_two_features_as_pca),
            raw_two_features_classifier=str(raw_two_features_classifier),
        )
        full_fit["scores"] = attach_pseudotime(full_fit["scores"], pseudotime_df)

        pd.DataFrame({"feature": subset_features}).to_csv(run_dir / "selected_features.csv", index=False)
        full_fit["scores"].to_csv(run_dir / "cell_pca_scores.csv", index=False)

        title = (
            f"Top {subset_size} features | "
            f"CV bal_acc={cv_stats['balanced_accuracy']:.3f} | "
            f"AUC={cv_stats['roc_auc']:.3f}"
        )
        plot_png = run_dir / "pca12_group_scatter.png"
        plot_svg = run_dir / "pca12_group_scatter.svg"
        plot_group_scatter(full_fit["scores"], full_fit["clf"], title, plot_png, plot_svg)
        if pseudotime_df is not None:
            pseudotime_plot_png = run_dir / "pca12_pseudotime_scatter.png"
            plot_pseudotime_scatter(
                full_fit["scores"],
                full_fit["clf"],
                palette_colors or read_palette_colors(None),
                f"Top {subset_size} features | Omics pseudotime",
                pseudotime_plot_png,
                run_dir / "pca12_pseudotime_scatter.svg",
            )
            pseudotime_plot_paths.append(pseudotime_plot_png)
        plot_paths.append(plot_png)

        summary_rows.append(
            {
                "subset_size": int(subset_size),
                "balanced_accuracy": cv_stats["balanced_accuracy"],
                "balanced_accuracy_std": cv_stats["balanced_accuracy_std"],
                "roc_auc": cv_stats["roc_auc"],
                "roc_auc_std": cv_stats["roc_auc_std"],
                "full_balanced_accuracy": full_fit["full_balanced_accuracy"],
                "full_roc_auc": full_fit["full_roc_auc"],
                "output_dir": str(run_dir),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("subset_size").reset_index(drop=True)
    summary.to_csv(out_dir / "pca12_search_summary.csv", index=False)

    summary_sorted = summary.sort_values(
        ["balanced_accuracy", "roc_auc", "subset_size"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    best_row = summary_sorted.iloc[0]

    save_contact_sheet(plot_paths, out_dir / "pca12_search_contact_sheet")
    save_contact_sheet(pseudotime_plot_paths, out_dir / "pca12_pseudotime_contact_sheet")

    return {
        "best_subset_size": int(best_row["subset_size"]),
        "best_balanced_accuracy": float(best_row["balanced_accuracy"]),
        "best_roc_auc": float(best_row["roc_auc"]),
        "out_dir": str(out_dir),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search feature combinations whose PCA1/PCA2 best separate Young and Old cells."
    )
    parser.add_argument("features_csv", help="CSV with sample_id features, or gene-by-cell TSV matrix.")
    parser.add_argument("out_dir", help="Output directory.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--prefilter-top-n", type=int, default=80)
    parser.add_argument("--search-max-features", type=int, default=30)
    parser.add_argument("--subset-sizes", default="2,3,5,8,10,15,20,25,30")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=5)
    parser.add_argument("--pseudotime-csv", default=None, help="Optional CSV with cell_id and pseudotime columns.")
    parser.add_argument("--pseudotime-palette", default=None, help="Optional text file containing hex colors for pseudotime.")
    parser.add_argument(
        "--top-variable-features",
        type=int,
        default=None,
        help="Keep only the top N highest-variance features/genes before PCA classification or feature search.",
    )
    parser.add_argument(
        "--ranked-subsets",
        action="store_true",
        help="Evaluate top-N F-score ranked subsets directly, without greedy forward search.",
    )
    parser.add_argument(
        "--use-all-features",
        action="store_true",
        help="Skip feature-number search and use every usable numeric feature for PCA1/PCA2.",
    )
    parser.add_argument(
        "--use-raw-two-features-as-pca",
        action="store_true",
        help="For two-feature manual inputs, use the two raw numeric features directly as PCA 1 and PCA 2.",
    )
    parser.add_argument(
        "--raw-two-features-classifier",
        choices=["linear", "poly3"],
        default="linear",
        help="Classifier used when --use-raw-two-features-as-pca is enabled.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    subset_sizes = [int(x) for x in str(args.subset_sizes).split(",") if str(x).strip()]
    summary = run_analysis(
        features_csv=Path(args.features_csv),
        out_dir=Path(args.out_dir),
        random_state=int(args.random_state),
        prefilter_top_n=int(args.prefilter_top_n),
        search_max_features=int(args.search_max_features),
        subset_sizes=subset_sizes,
        cv_splits=int(args.cv_splits),
        cv_repeats=int(args.cv_repeats),
        use_all_features=bool(args.use_all_features),
        pseudotime_csv=Path(args.pseudotime_csv) if args.pseudotime_csv else None,
        pseudotime_palette=Path(args.pseudotime_palette) if args.pseudotime_palette else None,
        top_variable_features=args.top_variable_features,
        ranked_subsets=bool(args.ranked_subsets),
        use_raw_two_features_as_pca=bool(args.use_raw_two_features_as_pca),
        raw_two_features_classifier=str(args.raw_two_features_classifier),
    )
    print(f"Output dir: {summary['out_dir']}")
    print(f"Best subset size: {summary['best_subset_size']}")
    print(f"Best balanced accuracy: {summary['best_balanced_accuracy']:.4f}")
    print(f"Best ROC AUC: {summary['best_roc_auc']:.4f}")


if __name__ == "__main__":
    main()
