#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from run_image_feature_young_old_pca12_search import (
    attach_pseudotime,
    build_numeric_feature_matrix,
    evaluate_subset_cv,
    fit_full_subset_projection,
    load_feature_table,
    plot_group_scatter,
    plot_pseudotime_scatter,
    read_palette_colors,
    read_pseudotime_table,
)


def read_table_auto(path: Path) -> pd.DataFrame:
    suffix = str(path).lower()
    sep = "\t" if suffix.endswith((".tsv", ".tab", ".txt")) else ","
    return pd.read_csv(path, sep=sep)


def read_selected_features(selected_features_csv: Path) -> list[str]:
    table = read_table_auto(selected_features_csv)
    if table.empty:
        raise ValueError(f"Selected-features file is empty: {selected_features_csv}")
    if "feature" in table.columns:
        raw = table["feature"]
    else:
        raw = table.iloc[:, 0]

    ordered = []
    seen = set()
    for value in raw.astype(str):
        name = value.strip()
        if not name or name.lower() == "nan" or name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    if not ordered:
        raise ValueError(f"No valid feature names found in: {selected_features_csv}")
    return ordered


def split_used_and_missing(selected: list[str], available_columns: list[str]) -> tuple[list[str], list[str]]:
    available = set(available_columns)
    used = [name for name in selected if name in available]
    missing = [name for name in selected if name not in available]
    return used, missing


def run_analysis(
    features_csv: Path,
    selected_features_csv: Path,
    out_dir: Path,
    random_state: int = 42,
    cv_splits: int = 5,
    cv_repeats: int = 5,
    pseudotime_csv: Optional[Path] = None,
    pseudotime_palette: Optional[Path] = None,
    use_raw_two_features_as_pca: bool = False,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_feature_table(Path(features_csv))
    X_df = build_numeric_feature_matrix(df)
    selected = read_selected_features(Path(selected_features_csv))
    used_features, missing_features = split_used_and_missing(selected, list(X_df.columns))
    if len(used_features) < 2:
        raise ValueError(
            f"Need at least 2 selected features present in input. Found {len(used_features)}."
        )

    y = np.where(df["Group"].eq("Young"), 1, 0).astype(int)
    cv_stats = evaluate_subset_cv(
        X_df=X_df,
        y=y,
        subset_features=used_features,
        random_state=int(random_state),
        cv_splits=int(cv_splits),
        cv_repeats=int(cv_repeats),
        use_raw_two_features_as_pca=bool(use_raw_two_features_as_pca),
    )
    full_fit = fit_full_subset_projection(
        df=df,
        X_df=X_df,
        subset_features=used_features,
        random_state=int(random_state),
        use_raw_two_features_as_pca=bool(use_raw_two_features_as_pca),
    )

    pseudotime_df = read_pseudotime_table(pseudotime_csv)
    palette_colors = read_palette_colors(pseudotime_palette) if pseudotime_df is not None else None
    full_fit["scores"] = attach_pseudotime(full_fit["scores"], pseudotime_df)

    pd.DataFrame({"feature": used_features}).to_csv(out_dir / "selected_features_used.csv", index=False)
    pd.DataFrame({"feature": missing_features}).to_csv(out_dir / "missing_selected_features.csv", index=False)
    full_fit["scores"].to_csv(out_dir / "cell_pca_scores.csv", index=False)

    title = (
        f"Selected {len(used_features)} features | "
        f"CV bal_acc={cv_stats['balanced_accuracy']:.3f} | "
        f"AUC={cv_stats['roc_auc']:.3f}"
    )
    group_png = out_dir / "pca12_group_scatter.png"
    group_svg = out_dir / "pca12_group_scatter.svg"
    plot_group_scatter(full_fit["scores"], full_fit["clf"], title, group_png, group_svg)

    pseudo_png = None
    pseudo_svg = None
    if pseudotime_df is not None:
        pseudo_png = out_dir / "pca12_pseudotime_scatter.png"
        pseudo_svg = out_dir / "pca12_pseudotime_scatter.svg"
        plot_pseudotime_scatter(
            scores=full_fit["scores"],
            clf=full_fit["clf"],
            palette_colors=palette_colors or read_palette_colors(None),
            title=f"Selected {len(used_features)} features | Omics pseudotime",
            out_png=pseudo_png,
            out_svg=pseudo_svg,
        )

    summary = pd.DataFrame(
        [
            {
                "feature_count": int(len(used_features)),
                "missing_feature_count": int(len(missing_features)),
                "balanced_accuracy": cv_stats["balanced_accuracy"],
                "balanced_accuracy_std": cv_stats["balanced_accuracy_std"],
                "roc_auc": cv_stats["roc_auc"],
                "roc_auc_std": cv_stats["roc_auc_std"],
                "full_balanced_accuracy": full_fit["full_balanced_accuracy"],
                "full_roc_auc": full_fit["full_roc_auc"],
                "group_plot_png": str(group_png),
                "group_plot_svg": str(group_svg),
                "pseudotime_plot_png": str(pseudo_png) if pseudo_png is not None else "",
                "pseudotime_plot_svg": str(pseudo_svg) if pseudo_svg is not None else "",
            }
        ]
    )
    summary.to_csv(out_dir / "pca12_summary.csv", index=False)

    return {
        "used_feature_count": int(len(used_features)),
        "missing_feature_count": int(len(missing_features)),
        "balanced_accuracy": float(cv_stats["balanced_accuracy"]),
        "roc_auc": float(cv_stats["roc_auc"]),
        "out_dir": str(out_dir),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run PCA12 Young/Old evaluation with a fixed selected-features list."
    )
    parser.add_argument("features_csv", help="Input feature table (sample_id rows).")
    parser.add_argument("selected_features_csv", help="CSV/TSV containing selected feature names.")
    parser.add_argument("out_dir", help="Output directory.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=5)
    parser.add_argument("--pseudotime-csv", default=None, help="Optional pseudotime CSV.")
    parser.add_argument("--pseudotime-palette", default=None, help="Optional pseudotime palette text file.")
    parser.add_argument(
        "--use-raw-two-features-as-pca",
        action="store_true",
        help="If exactly two features are used, treat them directly as PCA1/PCA2 coordinates.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    result = run_analysis(
        features_csv=Path(args.features_csv),
        selected_features_csv=Path(args.selected_features_csv),
        out_dir=Path(args.out_dir),
        random_state=int(args.random_state),
        cv_splits=int(args.cv_splits),
        cv_repeats=int(args.cv_repeats),
        pseudotime_csv=Path(args.pseudotime_csv) if args.pseudotime_csv else None,
        pseudotime_palette=Path(args.pseudotime_palette) if args.pseudotime_palette else None,
        use_raw_two_features_as_pca=bool(args.use_raw_two_features_as_pca),
    )
    print(f"Output dir: {result['out_dir']}")
    print(f"Used features: {result['used_feature_count']}")
    print(f"Missing features: {result['missing_feature_count']}")
    print(f"CV balanced accuracy: {result['balanced_accuracy']:.4f}")
    print(f"CV ROC AUC: {result['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
