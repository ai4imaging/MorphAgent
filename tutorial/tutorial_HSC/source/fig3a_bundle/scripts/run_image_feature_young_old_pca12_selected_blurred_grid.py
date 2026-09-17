#!/usr/bin/env python3

import argparse
import itertools
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from run_image_feature_young_old_pca12_search import (
    build_numeric_feature_matrix,
    draw_boundary,
    load_feature_table,
)
from run_image_feature_young_old_pca12_selected import read_selected_features


GROUP_COLORS = {
    "Young": "#5c8ab8",
    "Old": "#9e1b29",
}


def parse_float_list(text: str) -> list[float]:
    values = []
    for x in str(text).split(","):
        x = x.strip()
        if not x:
            continue
        values.append(float(x))
    return values


def parse_bool_list(text: str) -> list[bool]:
    out = []
    for x in str(text).split(","):
        s = x.strip().lower()
        if s in {"1", "true", "t", "yes", "y"}:
            out.append(True)
        elif s in {"0", "false", "f", "no", "n"}:
            out.append(False)
        else:
            raise ValueError(f"Invalid bool value: {x}")
    return out


def parse_str_list(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def make_scaler(mode: str):
    mode = mode.lower().strip()
    if mode == "standard":
        return StandardScaler()
    if mode == "robust":
        return RobustScaler()
    if mode == "minmax":
        return MinMaxScaler()
    if mode == "none":
        return None
    raise ValueError(f"Unsupported scaler: {mode}")


def signed_log1p(X: np.ndarray) -> np.ndarray:
    return np.sign(X) * np.log1p(np.abs(X))


def winsorize_by_quantile(X: np.ndarray, q: float) -> np.ndarray:
    if q <= 0:
        return X
    lo = np.nanquantile(X, q, axis=0)
    hi = np.nanquantile(X, 1.0 - q, axis=0)
    return np.clip(X, lo, hi)


def preprocess_train_test(
    X_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    scaler_mode: str,
    clip_q: float,
    use_log1p: bool,
) -> tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train_df)
    X_test = imputer.transform(X_test_df)

    if clip_q > 0:
        lo = np.quantile(X_train, clip_q, axis=0)
        hi = np.quantile(X_train, 1.0 - clip_q, axis=0)
        X_train = np.clip(X_train, lo, hi)
        X_test = np.clip(X_test, lo, hi)

    if use_log1p:
        X_train = signed_log1p(X_train)
        X_test = signed_log1p(X_test)

    scaler = make_scaler(scaler_mode)
    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train.astype(float), X_test.astype(float)


def pca_project_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    random_state: int,
    whiten: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if X_train.shape[1] == 1:
        train_coords = np.column_stack([X_train[:, 0], np.zeros(X_train.shape[0], dtype=float)])
        test_coords = np.column_stack([X_test[:, 0], np.zeros(X_test.shape[0], dtype=float)])
        return train_coords, test_coords
    pca = PCA(n_components=2, random_state=int(random_state), whiten=bool(whiten))
    train_coords = pca.fit_transform(X_train)
    test_coords = pca.transform(X_test)
    return train_coords, test_coords


def evaluate_cv(
    X_df: pd.DataFrame,
    y: np.ndarray,
    random_state: int,
    cv_splits: int,
    cv_repeats: int,
    scaler_mode: str,
    clip_q: float,
    use_log1p: bool,
    pca_whiten: bool,
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

    for train_idx, test_idx in cv.split(X_df, y):
        X_train_df = X_df.iloc[train_idx]
        X_test_df = X_df.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        X_train, X_test = preprocess_train_test(
            X_train_df, X_test_df, scaler_mode=scaler_mode, clip_q=float(clip_q), use_log1p=bool(use_log1p)
        )
        train_coords, test_coords = pca_project_train_test(
            X_train, X_test, random_state=int(random_state), whiten=bool(pca_whiten)
        )

        clf = LogisticRegression(
            penalty="l2",
            solver="liblinear",
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=int(random_state),
        )
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
    }


def full_fit(
    X_df: pd.DataFrame,
    df_meta: pd.DataFrame,
    random_state: int,
    scaler_mode: str,
    clip_q: float,
    use_log1p: bool,
    pca_whiten: bool,
) -> dict:
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X_df)
    if clip_q > 0:
        X = winsorize_by_quantile(X, clip_q)
    if use_log1p:
        X = signed_log1p(X)
    scaler = make_scaler(scaler_mode)
    if scaler is not None:
        X = scaler.fit_transform(X)

    if X.shape[1] == 1:
        coords = np.column_stack([X[:, 0], np.zeros(X.shape[0], dtype=float)])
    else:
        pca = PCA(n_components=2, random_state=int(random_state), whiten=bool(pca_whiten))
        coords = pca.fit_transform(X)

    y = np.where(df_meta["Group"].eq("Young"), 1, 0).astype(int)
    clf = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        C=1.0,
        class_weight="balanced",
        max_iter=5000,
        random_state=int(random_state),
    )
    clf.fit(coords, y)
    pred = clf.predict(coords)
    proba = clf.predict_proba(coords)[:, 1]

    scores = pd.DataFrame(
        {
            "cell_id": df_meta["sample_id"].astype(str),
            "Group": df_meta["Group"].astype(str),
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            "probability_young": proba,
            "predicted_group": np.where(pred == 1, "Young", "Old"),
        }
    )

    # 分散度指标：PC1/PC2 协方差椭圆面积 proxy（sqrt(det(cov)))
    cov = np.cov(coords[:, 0], coords[:, 1])
    det_cov = float(np.linalg.det(cov)) if np.all(np.isfinite(cov)) else np.nan
    spread_score = float(np.sqrt(max(det_cov, 0.0))) if np.isfinite(det_cov) else np.nan

    return {
        "scores": scores,
        "clf": clf,
        "full_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "full_roc_auc": float(roc_auc_score(y, proba)),
        "spread_score": spread_score,
    }


def compute_plot_limits(scores: pd.DataFrame, quantile_clip: float = 0.01) -> tuple[tuple[float, float], tuple[float, float]]:
    x = scores["PC1"].to_numpy(dtype=float)
    y = scores["PC2"].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0:
        return (-1.0, 1.0), (-1.0, 1.0)
    q = float(max(0.0, min(0.2, quantile_clip)))
    x0, x1 = np.quantile(x, [q, 1.0 - q])
    y0, y1 = np.quantile(y, [q, 1.0 - q])
    if abs(x1 - x0) < 1e-12:
        x0, x1 = float(np.min(x)), float(np.max(x))
    if abs(y1 - y0) < 1e-12:
        y0, y1 = float(np.min(y)), float(np.max(y))
    xpad = (x1 - x0) * 0.10 if x1 > x0 else 1.0
    ypad = (y1 - y0) * 0.10 if y1 > y0 else 1.0
    return (float(x0 - xpad), float(x1 + xpad)), (float(y0 - ypad), float(y1 + ypad))


def plot_group_scatter(scores: pd.DataFrame, clf, title: str, out_png: Path, out_svg: Path) -> None:
    x_limits, y_limits = compute_plot_limits(scores, quantile_clip=0.01)
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


def run_grid(
    features_csv: Path,
    selected_features_csv: Path,
    out_dir: Path,
    random_state: int,
    cv_splits: int,
    cv_repeats: int,
    scaler_modes: list[str],
    clip_quantiles: list[float],
    log1p_options: list[bool],
    pca_whiten_options: list[bool],
    combo_index: Optional[int] = None,
) -> pd.DataFrame:
    total_combos = len(scaler_modes) * len(clip_quantiles) * len(log1p_options) * len(pca_whiten_options)
    if combo_index is not None and not 1 <= combo_index <= total_combos:
        raise ValueError(f"--combo-index must be between 1 and {total_combos}.")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_feature_table(Path(features_csv))
    X_df_all = build_numeric_feature_matrix(df)
    selected = read_selected_features(Path(selected_features_csv))
    used = [f for f in selected if f in X_df_all.columns]
    missing = [f for f in selected if f not in X_df_all.columns]
    if len(used) < 2:
        raise ValueError(f"Need at least 2 selected features present in input. Found {len(used)}.")
    X_df = X_df_all[used].copy()
    y = np.where(df["Group"].eq("Young"), 1, 0).astype(int)

    pd.DataFrame({"feature": used}).to_csv(out_dir / "selected_features_used.csv", index=False)
    pd.DataFrame({"feature": missing}).to_csv(out_dir / "missing_selected_features.csv", index=False)

    rows = []
    combo_idx = 0
    for scaler_mode, clip_q, use_log1p, pca_whiten in itertools.product(
        scaler_modes, clip_quantiles, log1p_options, pca_whiten_options
    ):
        combo_idx += 1
        if combo_index is not None and combo_idx != combo_index:
            continue
        combo_name = (
            f"combo_{combo_idx:03d}__scaler_{scaler_mode}"
            f"__clip_{clip_q:g}__log1p_{int(use_log1p)}__whiten_{int(pca_whiten)}"
        )
        combo_dir = out_dir / combo_name
        combo_dir.mkdir(parents=True, exist_ok=True)

        cv_stats = evaluate_cv(
            X_df=X_df,
            y=y,
            random_state=int(random_state),
            cv_splits=int(cv_splits),
            cv_repeats=int(cv_repeats),
            scaler_mode=scaler_mode,
            clip_q=float(clip_q),
            use_log1p=bool(use_log1p),
            pca_whiten=bool(pca_whiten),
        )
        ff = full_fit(
            X_df=X_df,
            df_meta=df[["sample_id", "Group"]],
            random_state=int(random_state),
            scaler_mode=scaler_mode,
            clip_q=float(clip_q),
            use_log1p=bool(use_log1p),
            pca_whiten=bool(pca_whiten),
        )

        ff["scores"].to_csv(combo_dir / "cell_pca_scores.csv", index=False)
        title = (
            f"{combo_name}\n"
            f"CV bal_acc={cv_stats['balanced_accuracy']:.3f} | AUC={cv_stats['roc_auc']:.3f}"
        )
        plot_group_scatter(
            scores=ff["scores"],
            clf=ff["clf"],
            title=title,
            out_png=combo_dir / "pca12_group_scatter.png",
            out_svg=combo_dir / "pca12_group_scatter.svg",
        )

        rows.append(
            {
                "combo_name": combo_name,
                "feature_count": int(len(used)),
                "scaler": scaler_mode,
                "clip_quantile": float(clip_q),
                "log1p": bool(use_log1p),
                "pca_whiten": bool(pca_whiten),
                "cv_balanced_accuracy": cv_stats["balanced_accuracy"],
                "cv_balanced_accuracy_std": cv_stats["balanced_accuracy_std"],
                "cv_roc_auc": cv_stats["roc_auc"],
                "cv_roc_auc_std": cv_stats["roc_auc_std"],
                "full_balanced_accuracy": ff["full_balanced_accuracy"],
                "full_roc_auc": ff["full_roc_auc"],
                "spread_score": ff["spread_score"],
                "plot_svg": str(combo_dir / "pca12_group_scatter.svg"),
                "plot_png": str(combo_dir / "pca12_group_scatter.png"),
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        ["cv_roc_auc", "spread_score", "cv_balanced_accuracy"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    summary.to_csv(out_dir / "grid_summary.csv", index=False)
    summary.head(min(10, len(summary))).to_csv(out_dir / "grid_top10.csv", index=False)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Grid search preprocessing/PCA params on blurred fixed-25 features and save all plots + AUC table."
    )
    p.add_argument(
        "--features-csv",
        default="/Users/pengrui/Desktop/投稿/yez/blurred_data/merged_blurred_features.csv",
    )
    p.add_argument(
        "--selected-features-csv",
        default="/Users/pengrui/Desktop/投稿/yez/results_merged_features_pca12_young_old_feature_search/top_025_features/selected_features.csv",
    )
    p.add_argument(
        "--out-dir",
        default="/Users/pengrui/Desktop/投稿/yez/results_blurred_image_feature_pca12_fixed25_grid",
    )
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--cv-repeats", type=int, default=5)
    p.add_argument("--scalers", default="standard,robust,minmax")
    p.add_argument("--clip-quantiles", default="0,0.01,0.02")
    p.add_argument("--log1p-options", default="0,1")
    p.add_argument("--pca-whiten-options", default="0,1")
    p.add_argument("--combo-index", type=int, default=None, help="Run one combination by its original grid index.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    summary = run_grid(
        features_csv=Path(args.features_csv),
        selected_features_csv=Path(args.selected_features_csv),
        out_dir=Path(args.out_dir),
        random_state=int(args.random_state),
        cv_splits=int(args.cv_splits),
        cv_repeats=int(args.cv_repeats),
        scaler_modes=parse_str_list(args.scalers),
        clip_quantiles=parse_float_list(args.clip_quantiles),
        log1p_options=parse_bool_list(args.log1p_options),
        pca_whiten_options=parse_bool_list(args.pca_whiten_options),
        combo_index=args.combo_index,
    )
    print(f"Output dir: {args.out_dir}")
    print(f"Combinations: {len(summary)}")
    if len(summary) > 0:
        top = summary.iloc[0]
        print(
            "Best combo: "
            f"{top['combo_name']} | "
            f"CV bal_acc={top['cv_balanced_accuracy']:.4f} | "
            f"AUC={top['cv_roc_auc']:.4f} | "
            f"spread={top['spread_score']:.4f}"
        )


if __name__ == "__main__":
    main()
