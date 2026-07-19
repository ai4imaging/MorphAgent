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


def compute_metadata_alignment(
    values: pd.Series,
    metadata: MetadataContext,
) -> Dict[str, Any]:
    """Compute deterministic metadata alignment metrics for one feature."""

    result = {
        "metadata_max_abs_spearman": 0.0,
        "metadata_best_spearman_field": None,
        "metadata_max_anova_f": 0.0,
        "metadata_best_anova_field": None,
        "metadata_max_eta_squared": 0.0,
        "metadata_best_eta_field": None,
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

    result["metadata_alignment_score"] = max(
        result["metadata_max_abs_spearman"],
        result["metadata_max_eta_squared"],
    )
    return result
