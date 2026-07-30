"""Deterministic metadata parsing and alignment metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


ID_KEYWORDS = ("sample_id", "filename", "file", "path", "image", "dataset_image")
CATEGORICAL_HINTS = (
    "group",
    "label",
    "condition",
    "batch",
    "well",
    "timepoint",
    "time_point",
    "time",
    "mutation",
    "genotype",
    "phenotype",
    "treatment",
    "modality",
    "plate",
    "site",
)


@dataclass
class MetadataContext:
    """Resolved metadata context for one run."""

    dataframe: Optional[pd.DataFrame]
    categorical_fields: List[str]
    continuous_fields: List[str]
    ignored_fields: List[str]
    sample_id_prefix_counts: Dict[str, int]
    fallback_notes: List[str]
    has_explicit_metadata: bool

    def to_summary(self) -> Dict[str, Any]:
        return {
            "has_explicit_metadata": self.has_explicit_metadata,
            "categorical_fields": self.categorical_fields,
            "continuous_fields": self.continuous_fields,
            "ignored_fields": self.ignored_fields,
            "sample_id_prefix_counts": self.sample_id_prefix_counts,
            "fallback_notes": self.fallback_notes,
        }


def _sample_prefix_counts(sample_ids: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for sample_id in sample_ids:
        prefix = str(sample_id).split("_", 1)[0] if sample_id else "unknown"
        counts[prefix] = counts.get(prefix, 0) + 1
    return counts


def _looks_like_id_column(column_name: str) -> bool:
    name = str(column_name).lower()
    return any(keyword in name for keyword in ID_KEYWORDS)


def classify_metadata_fields(metadata_df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Split metadata columns into categorical, continuous, ignored."""

    categorical_fields: List[str] = []
    continuous_fields: List[str] = []
    ignored_fields: List[str] = []

    n_rows = max(len(metadata_df), 1)

    for column in metadata_df.columns:
        name = str(column)
        lower_name = name.lower()
        series = metadata_df[column]

        if _looks_like_id_column(lower_name):
            ignored_fields.append(name)
            continue

        if pd.api.types.is_numeric_dtype(series):
            unique_count = series.dropna().nunique()
            if any(hint in lower_name for hint in CATEGORICAL_HINTS):
                categorical_fields.append(name)
            elif unique_count <= max(12, n_rows // 10):
                categorical_fields.append(name)
            else:
                continuous_fields.append(name)
        else:
            categorical_fields.append(name)

    return categorical_fields, continuous_fields, ignored_fields


def load_metadata_context(
    sample_ids: List[str],
    metadata_path: Optional[Path],
) -> MetadataContext:
    """Load metadata if present; otherwise build conservative fallback context."""

    fallback_counts = _sample_prefix_counts(sample_ids)
    fallback_notes = [
        "No aggressive pseudo-metadata inference was applied.",
        "Sample ID prefixes are logged for context only and are not used for biological ranking.",
    ]

    if metadata_path is None or not metadata_path.exists():
        return MetadataContext(
            dataframe=None,
            categorical_fields=[],
            continuous_fields=[],
            ignored_fields=[],
            sample_id_prefix_counts=fallback_counts,
            fallback_notes=fallback_notes,
            has_explicit_metadata=False,
        )

    metadata_df = pd.read_csv(metadata_path)
    merged_df = _align_metadata_to_samples(sample_ids, metadata_df)
    categorical_fields, continuous_fields, ignored_fields = classify_metadata_fields(merged_df)
    return MetadataContext(
        dataframe=merged_df,
        categorical_fields=categorical_fields,
        continuous_fields=continuous_fields,
        ignored_fields=ignored_fields,
        sample_id_prefix_counts=fallback_counts,
        fallback_notes=[],
        has_explicit_metadata=True,
    )


def _align_metadata_to_samples(sample_ids: List[str], metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Align metadata rows to the feature sample order."""

    if "sample_id" in metadata_df.columns:
        merged = pd.DataFrame({"sample_id": sample_ids}).merge(
            metadata_df,
            on="sample_id",
            how="left",
        )
        return merged.drop(columns=["sample_id"])

    if len(metadata_df.columns) > 0:
        first_col = metadata_df.columns[0]
        metadata_values = set(metadata_df[first_col].astype(str))
        if set(map(str, sample_ids)) & metadata_values:
            merged = pd.DataFrame({"sample_id": sample_ids}).merge(
                metadata_df,
                left_on="sample_id",
                right_on=first_col,
                how="left",
            )
            return merged.drop(columns=["sample_id", first_col], errors="ignore")

    if len(metadata_df) != len(sample_ids):
        raise ValueError(
            f"Metadata row count ({len(metadata_df)}) does not match sample count ({len(sample_ids)})"
        )

    return metadata_df.reset_index(drop=True).copy()


def _mann_whitney_auc(feature_values: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC via Mann–Whitney U for a binary label (chance = 0.5)."""

    classes = pd.unique(labels)
    if len(classes) != 2:
        return 0.5
    left = feature_values[labels == classes[0]]
    right = feature_values[labels == classes[1]]
    if len(left) == 0 or len(right) == 0:
        return 0.5
    try:
        u_stat, _ = stats.mannwhitneyu(left, right, alternative="two-sided")
    except Exception:
        return 0.5
    auc = float(u_stat) / (len(left) * len(right))
    # Orientation-invariant: take the better of auc / 1-auc.
    return max(auc, 1.0 - auc)


def _linear_classifier_score(feature_values: np.ndarray, labels: np.ndarray) -> float:
    """Loose paired-metadata separability via a linear classifier (in-sample).

    With small n (e.g. 10 demo samples) this is intentionally permissive: any
    slight linear separation of metadata classes yields a score above chance.
    Returns a [0, 1] score where 0.5 ≈ chance and higher is better.
    """

    if len(feature_values) < 4 or len(np.unique(labels)) < 2:
        return 0.5
    if len(np.unique(feature_values)) < 2:
        return 0.5

    # Binary labels: exact rank AUC (stable, no sklearn fit needed).
    if len(np.unique(labels)) == 2:
        return _mann_whitney_auc(feature_values, labels)

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import LabelEncoder
    except Exception:
        # Multiclass fallback without sklearn: max pairwise Mann–Whitney AUC.
        classes = list(pd.unique(labels))
        best = 0.5
        for index, left_label in enumerate(classes):
            for right_label in classes[index + 1 :]:
                mask = np.isin(labels, [left_label, right_label])
                best = max(best, _mann_whitney_auc(feature_values[mask], labels[mask]))
        return best

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    x = feature_values.reshape(-1, 1)
    try:
        model = LogisticRegression(
            max_iter=200,
            solver="lbfgs",
            multi_class="auto",
            random_state=0,
        )
        model.fit(x, y)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x)
            if proba.shape[1] == 2:
                return float(roc_auc_score(y, proba[:, 1]))
            return float(roc_auc_score(y, proba, multi_class="ovr", average="macro"))
        scores = model.decision_function(x)
        if np.ndim(scores) == 1:
            return float(roc_auc_score(y, scores))
        return float(roc_auc_score(y, scores, multi_class="ovr", average="macro"))
    except Exception:
        return 0.5


def compute_metadata_alignment(
    values: pd.Series,
    metadata: MetadataContext,
) -> Dict[str, Any]:
    """Compute deterministic paired-metadata alignment metrics for one feature."""

    result = {
        "metadata_max_abs_spearman": 0.0,
        "metadata_best_spearman_field": None,
        "metadata_max_anova_f": 0.0,
        "metadata_best_anova_field": None,
        "metadata_max_eta_squared": 0.0,
        "metadata_best_eta_field": None,
        "metadata_max_classifier_auc": 0.5,
        "metadata_best_classifier_field": None,
        "metadata_alignment_score": 0.0,
    }

    if metadata.dataframe is None:
        return result

    data = metadata.dataframe
    if len(values) != len(data):
        raise ValueError("Feature values and metadata rows must be aligned.")

    for field in metadata.continuous_fields:
        field_values = pd.to_numeric(data[field], errors="coerce")
        mask = values.notna() & field_values.notna()
        if mask.sum() < 3:
            continue
        x = values[mask]
        y = field_values[mask]
        if x.nunique() < 2 or y.nunique() < 2:
            continue
        corr, _ = stats.spearmanr(x, y, nan_policy="omit")
        if pd.notna(corr) and abs(float(corr)) > result["metadata_max_abs_spearman"]:
            result["metadata_max_abs_spearman"] = abs(float(corr))
            result["metadata_best_spearman_field"] = field

    for field in metadata.categorical_fields:
        field_values = data[field]
        mask = values.notna() & field_values.notna()
        if mask.sum() < 3:
            continue
        grouped = []
        x = values[mask]
        y = field_values[mask]
        for _, group_values in x.groupby(y):
            if len(group_values) > 0:
                grouped.append(group_values.to_numpy())
        if len(grouped) < 2:
            continue
        try:
            f_stat, _ = stats.f_oneway(*grouped)
        except Exception:
            continue
        if pd.isna(f_stat):
            continue
        total_mean = float(np.mean(x))
        ss_between = 0.0
        ss_total = float(np.sum((x - total_mean) ** 2))
        if ss_total > 0:
            for group in grouped:
                ss_between += len(group) * (float(np.mean(group)) - total_mean) ** 2
            eta_sq = max(0.0, min(1.0, ss_between / ss_total))
        else:
            eta_sq = 0.0
        if float(f_stat) > result["metadata_max_anova_f"]:
            result["metadata_max_anova_f"] = float(f_stat)
            result["metadata_best_anova_field"] = field
        if eta_sq > result["metadata_max_eta_squared"]:
            result["metadata_max_eta_squared"] = eta_sq
            result["metadata_best_eta_field"] = field

        classifier_auc = _linear_classifier_score(x.to_numpy(dtype=float), y.to_numpy())
        if classifier_auc > result["metadata_max_classifier_auc"]:
            result["metadata_max_classifier_auc"] = float(classifier_auc)
            result["metadata_best_classifier_field"] = field

    # Map classifier AUC (chance=0.5) onto a [0, 1] alignment contribution.
    classifier_alignment = max(0.0, (float(result["metadata_max_classifier_auc"]) - 0.5) * 2.0)
    result["metadata_alignment_score"] = max(
        result["metadata_max_abs_spearman"],
        result["metadata_max_eta_squared"],
        classifier_alignment,
    )
    return result
