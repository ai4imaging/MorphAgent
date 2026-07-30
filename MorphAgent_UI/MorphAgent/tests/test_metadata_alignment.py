from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from unittest.mock import patch

from validation.metadata import compute_metadata_alignment, load_metadata_context


def _executor_without_llm():
    """Build ValidationExecutor without constructing a live ChatOpenAI client."""
    with patch("validation.executor.LLMRedundancyReviewer", return_value=object()):
        from validation.executor import ValidationExecutor

        return ValidationExecutor()


class PairedMetadataAlignmentTests(unittest.TestCase):
    def test_categorical_group_uses_eta_and_classifier_not_only_spearman(self) -> None:
        sample_ids = [f"WT_{i}" for i in range(1, 6)] + [f"MU_{i}" for i in range(1, 6)]
        # Feature that weakly but clearly separates WT vs MU.
        values = pd.Series([1.0, 1.1, 0.9, 1.2, 1.0, 2.0, 2.1, 1.9, 2.2, 2.0])
        with tempfile.TemporaryDirectory() as raw:
            meta_path = Path(raw) / "metadata.csv"
            meta_path.write_text(
                "sample_id,group,genotype\n"
                + "\n".join(
                    f"{sid},{'WT' if sid.startswith('WT') else 'MU'},"
                    f"{'wild_type' if sid.startswith('WT') else 'mutant'}"
                    for sid in sample_ids
                )
                + "\n",
                encoding="utf-8",
            )
            context = load_metadata_context(sample_ids, meta_path)
            self.assertIn("group", context.categorical_fields)
            metrics = compute_metadata_alignment(values, context)

        self.assertEqual(metrics["metadata_max_abs_spearman"], 0.0)
        self.assertGreater(metrics["metadata_max_eta_squared"], 0.5)
        self.assertGreaterEqual(metrics["metadata_max_classifier_auc"], 0.9)
        self.assertGreater(metrics["metadata_alignment_score"], 0.05)

    def test_hard_filter_is_loose_for_paired_metadata(self) -> None:
        executor = _executor_without_llm()
        self.assertLessEqual(executor.min_metadata_alignment, 0.05)
        self.assertLessEqual(executor.min_metadata_classifier_auc, 0.55)

        sample_ids = [f"WT_{i}" for i in range(1, 6)] + [f"MU_{i}" for i in range(1, 6)]
        # Tiny but consistent WT/MU separation — should still pass the loose screen.
        values = pd.Series([1.00, 1.02, 0.98, 1.04, 1.01, 1.10, 1.12, 1.08, 1.14, 1.11])
        with tempfile.TemporaryDirectory() as raw:
            meta_path = Path(raw) / "metadata.csv"
            rows = ["sample_id,group"]
            rows.extend(f"{sid},{'WT' if sid.startswith('WT') else 'MU'}" for sid in sample_ids)
            meta_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            context = load_metadata_context(sample_ids, meta_path)
            metrics = compute_metadata_alignment(values, context)

            from validation.records import FeatureMetrics, FeatureRecord

            record = FeatureRecord(
                feature_id="round_1:weak_sep",
                round_number=1,
                name="weak_sep",
                canonical_name="weak_sep",
                method="code",
                description="",
                category="",
                values=values.tolist(),
            )
            feature_metrics = FeatureMetrics(
                availability_rate=1.0,
                finite_rate=1.0,
                n_valid=10,
                n_unique=int(values.nunique()),
                mean=float(values.mean()),
                std=float(values.std(ddof=0)),
                cv=0.2,
                median=float(values.median()),
                iqr=0.1,
                robust_dispersion=0.2,
                zero_fraction=0.0,
                dynamic_range=float(values.max() - values.min()),
                unsupervised_signal=0.2,
                metadata_max_abs_spearman=metrics["metadata_max_abs_spearman"],
                metadata_max_eta_squared=metrics["metadata_max_eta_squared"],
                metadata_max_classifier_auc=metrics["metadata_max_classifier_auc"],
                metadata_alignment_score=metrics["metadata_alignment_score"],
                base_validation_score=0.5,
                validation_score=0.5,
            )
            decision = executor._decision_after_hard_filters(
                record, feature_metrics, metadata_available=True
            )

        self.assertEqual(decision.status, "retained")
        self.assertNotIn("low_paired_metadata_alignment", decision.reason_codes)
        self.assertNotIn("low_transcriptome_alignment", decision.reason_codes)

    def test_constant_feature_still_fails_loose_metadata_screen(self) -> None:
        executor = _executor_without_llm()
        sample_ids = [f"WT_{i}" for i in range(1, 6)] + [f"MU_{i}" for i in range(1, 6)]
        values = pd.Series([1.0] * 10)
        with tempfile.TemporaryDirectory() as raw:
            meta_path = Path(raw) / "metadata.csv"
            rows = ["sample_id,group"]
            rows.extend(f"{sid},{'WT' if sid.startswith('WT') else 'MU'}" for sid in sample_ids)
            meta_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            context = load_metadata_context(sample_ids, meta_path)
            metrics = compute_metadata_alignment(values, context)

            from validation.records import FeatureMetrics, FeatureRecord

            record = FeatureRecord(
                feature_id="round_1:constant",
                round_number=1,
                name="constant",
                canonical_name="constant",
                method="code",
                description="",
                category="",
                values=values.tolist(),
            )
            feature_metrics = FeatureMetrics(
                availability_rate=1.0,
                finite_rate=1.0,
                n_valid=10,
                n_unique=1,
                mean=1.0,
                std=0.0,
                cv=0.0,
                median=1.0,
                iqr=0.0,
                robust_dispersion=0.0,
                zero_fraction=0.0,
                dynamic_range=0.0,
                unsupervised_signal=0.0,
                metadata_max_abs_spearman=metrics["metadata_max_abs_spearman"],
                metadata_max_eta_squared=metrics["metadata_max_eta_squared"],
                metadata_max_classifier_auc=metrics["metadata_max_classifier_auc"],
                metadata_alignment_score=metrics["metadata_alignment_score"],
                base_validation_score=0.2,
                validation_score=0.2,
            )
            decision = executor._decision_after_hard_filters(
                record, feature_metrics, metadata_available=True
            )

        self.assertEqual(decision.status, "dropped")
        self.assertIn("low_paired_metadata_alignment", decision.reason_codes)


if __name__ == "__main__":
    unittest.main()
