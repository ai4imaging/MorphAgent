"""Deterministic round validator for MorphAgent."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .metadata import MetadataContext, compute_metadata_alignment, load_metadata_context
from .records import (
    FeatureDecision,
    FeatureMetrics,
    FeatureRecord,
    RegistryEntry,
    ValidationResult,
)
from .review import LLMRedundancyReviewer


TIMESTAMP_SUFFIX_RE = re.compile(r"_new_\d{8}_\d{6}$")


def _canonical_name(name: str) -> str:
    base = TIMESTAMP_SUFFIX_RE.sub("", name or "")
    base = base.lower().strip()
    base = re.sub(r"[^a-z0-9]+", "_", base)
    return re.sub(r"_+", "_", base).strip("_")


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_json_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def _row_signature(values: pd.Series) -> Tuple[Any, ...]:
    signature: List[Any] = []
    for value in values.tolist():
        if pd.isna(value):
            signature.append("__nan__")
        else:
            signature.append(round(float(value), 12))
    return tuple(signature)


def _tokenize_concept(text: str) -> Set[str]:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    stopwords = {"mean", "median", "std", "ratio", "score", "index", "feature", "tau", "vlm"}
    return {token for token in tokens if token not in stopwords and len(token) > 2}


class ValidationExecutor:
    """Primary deterministic round validator."""

    def __init__(self) -> None:
        self.cv_threshold = 0.10
        self.min_availability_rate = 0.7
        self.min_unique_values = 5
        self.deterministic_corr_threshold = 0.98
        self.ambiguous_corr_threshold = 0.95
        # Very loose paired-metadata screen: any slight Spearman / eta² /
        # linear-classifier signal is enough to pass (not a transcriptome assay).
        self.min_metadata_alignment = 0.05
        self.min_metadata_classifier_auc = 0.55
        self.min_base_validation_score = 0.35
        self.llm_reviewer = LLMRedundancyReviewer()
        # Backward-compatible alias used by older summaries / callers.
        self.min_transcriptome_alignment = self.min_metadata_alignment

    def validate_round(
        self,
        raw_features_csv: Path,
        feature_plan: Dict[str, Any],
        prior_registry: Optional[Dict[str, Any]],
        metadata_path: Optional[Path],
        round_dir: Path,
        run_dir: Path,
    ) -> ValidationResult:
        """Validate the current round and update persistent registry artifacts."""

        features_df = pd.read_csv(raw_features_csv)
        if "sample_id" not in features_df.columns:
            raise ValueError("Raw features CSV must include a sample_id column.")

        round_number = self._infer_round_number(round_dir)
        sample_ids = features_df["sample_id"].astype(str).tolist()
        planned_features = feature_plan.get("features", []) or []
        prior_registry = prior_registry or {}
        previous_raw_names = set(prior_registry.get("all_raw_feature_names", []))
        previous_live_ids = set(prior_registry.get("live_feature_ids", []))

        current_feature_columns = [
            column
            for column in features_df.columns
            if column != "sample_id" and column not in previous_raw_names
        ]
        if not current_feature_columns and round_number == 1:
            current_feature_columns = [column for column in features_df.columns if column != "sample_id"]

        records = self._build_feature_records(
            current_feature_columns=current_feature_columns,
            planned_features=planned_features,
            features_df=features_df,
            round_number=round_number,
            round_dir=round_dir,
        )
        metadata = load_metadata_context(sample_ids, metadata_path if metadata_path and metadata_path.exists() else None)
        record_rank_by_id = {
            record.feature_id: rank
            for rank, record in enumerate(records)
        }
        metrics_by_id = {
            record.feature_id: self._compute_feature_metrics(pd.Series(record.values, dtype=float), metadata)
            for record in records
        }

        registry_entries = {
            entry["feature_id"]: RegistryEntry(**entry)
            for entry in prior_registry.get("entries", [])
        }
        live_entries = {
            feature_id: registry_entries[feature_id]
            for feature_id in previous_live_ids
            if feature_id in registry_entries and registry_entries[feature_id].live
        }
        historical_live_records = self._build_historical_live_records(features_df, live_entries)
        for feature_id, live_record in historical_live_records.items():
            metrics_by_id.setdefault(
                feature_id,
                self._compute_feature_metrics(pd.Series(live_record.values, dtype=float), metadata),
            )

        decisions_by_id: Dict[str, FeatureDecision] = {}
        exact_signatures: Dict[Tuple[Any, ...], str] = {}
        redundancy_components: Dict[str, Set[str]] = defaultdict(set)
        deterministic_groups: List[Dict[str, Any]] = []
        ambiguous_pairs: List[Dict[str, Any]] = []
        superseded_live_ids: Set[str] = set()

        for feature_id, live_record in historical_live_records.items():
            exact_signatures[_row_signature(pd.Series(live_record.values, dtype=float))] = feature_id

        # Hard filters and exact duplicate bookkeeping.
        for record in records:
            metrics = metrics_by_id[record.feature_id]
            decision = self._decision_after_hard_filters(
                record, metrics, metadata_available=metadata.has_explicit_metadata
            )
            decisions_by_id[record.feature_id] = decision

            if decision.status == "dropped":
                continue

            signature = _row_signature(pd.Series(record.values, dtype=float))
            if signature in exact_signatures:
                winner_id = exact_signatures[signature]
                kept_id, dropped_id = self._choose_stronger_feature(
                    winner_id,
                    record.feature_id,
                    metrics_by_id,
                    live_entries,
                    registry_entries,
                    record_rank_by_id,
                )
                exact_signatures[signature] = kept_id
                self._mark_deterministic_drop(
                    kept_id,
                    dropped_id,
                    decisions_by_id,
                    metrics_by_id,
                    deterministic_groups,
                    reason_code="duplicate_exact_vector",
                    superseded_live_ids=superseded_live_ids,
                    live_entries=live_entries,
                )
            else:
                exact_signatures[signature] = record.feature_id

        # Deterministic and ambiguous redundancy versus previous live features and new survivors.
        surviving_ids = [
            record.feature_id
            for record in records
            if decisions_by_id[record.feature_id].status != "dropped"
        ]
        historical_ids = list(live_entries.keys())
        for feature_id in surviving_ids + historical_ids:
            redundancy_components[feature_id].add(feature_id)

        for index, left_id in enumerate(surviving_ids):
            intra_round_right_ids = surviving_ids[index + 1 :]
            cross_round_right_ids = historical_ids
            for right_id in intra_round_right_ids + cross_round_right_ids:
                left_record = historical_live_records.get(left_id) or self._lookup_record(left_id, records, registry_entries)
                right_record = historical_live_records.get(right_id) or self._lookup_record(right_id, records, registry_entries)
                if left_record is None or right_record is None:
                    continue
                if left_id in decisions_by_id and decisions_by_id[left_id].status == "dropped":
                    continue
                if right_id in decisions_by_id and decisions_by_id[right_id].status == "dropped":
                    continue
                corr = self._pairwise_correlation(left_record.values, right_record.values)
                same_name = left_record.canonical_name == right_record.canonical_name
                semantic_similarity = self._semantic_similarity(left_record, right_record)

                if same_name or (corr is not None and abs(corr) >= self.deterministic_corr_threshold):
                    kept_id, dropped_id = self._choose_stronger_feature(
                        left_id,
                        right_id,
                        metrics_by_id,
                        live_entries,
                        registry_entries,
                        record_rank_by_id,
                    )
                    reason_code = "duplicate_canonical_name" if same_name else "duplicate_high_correlation"
                    self._mark_deterministic_drop(
                        kept_id,
                        dropped_id,
                        decisions_by_id,
                        metrics_by_id,
                        deterministic_groups,
                        reason_code=reason_code,
                        corr=corr,
                        superseded_live_ids=superseded_live_ids,
                        live_entries=live_entries,
                    )
                    redundancy_components[kept_id].add(dropped_id)
                    redundancy_components[dropped_id].add(kept_id)
                    continue

                if corr is not None and abs(corr) >= self.ambiguous_corr_threshold:
                    ambiguous_pairs.append(
                        self._build_ambiguous_pair(left_record, right_record, corr, "ambiguous_high_correlation")
                    )
                    redundancy_components[left_id].add(right_id)
                    redundancy_components[right_id].add(left_id)
                elif semantic_similarity >= 0.65:
                    ambiguous_pairs.append(
                        self._build_ambiguous_pair(left_record, right_record, corr, "ambiguous_semantic_similarity")
                    )
                    redundancy_components[left_id].add(right_id)
                    redundancy_components[right_id].add(left_id)

        ambiguous_groups = self._group_ambiguous_pairs(ambiguous_pairs)
        llm_review_result = self.llm_reviewer.review_groups(ambiguous_groups) if ambiguous_groups else {
            "success": True,
            "groups": [],
            "chunks": [],
        }
        redundancy_resolutions = self._apply_llm_review(
            ambiguous_groups=ambiguous_groups,
            llm_review_result=llm_review_result,
            decisions_by_id=decisions_by_id,
            metrics_by_id=metrics_by_id,
            live_entries=live_entries,
            registry_entries=registry_entries,
            record_rank_by_id=record_rank_by_id,
            superseded_live_ids=superseded_live_ids,
        )

        # Final scoring pass.
        for decision in decisions_by_id.values():
            metrics = metrics_by_id.get(decision.feature_id)
            if metrics is None:
                continue
            metrics.validation_score = max(0.0, metrics.base_validation_score - metrics.redundancy_penalty)
            decision.validation_score = metrics.validation_score

        retained_ids = [
            record.feature_id
            for record in records
            if decisions_by_id[record.feature_id].status == "retained"
        ]
        dropped_ids = [
            record.feature_id
            for record in records
            if decisions_by_id[record.feature_id].status == "dropped"
        ]

        updated_registry = self._update_registry(
            prior_registry=prior_registry,
            registry_entries=registry_entries,
            records=records,
            decisions_by_id=decisions_by_id,
            round_number=round_number,
            current_raw_names=[column for column in features_df.columns if column != "sample_id"],
            superseded_live_ids=superseded_live_ids,
        )

        retained_df = self._build_retained_features_df(features_df, updated_registry)
        retained_csv_path = run_dir / "retained_features.csv"
        retained_df.to_csv(retained_csv_path, index=False, encoding="utf-8")

        decisions_df = self._build_decisions_df(records, metrics_by_id, decisions_by_id)
        decisions_path = round_dir / "validation_decisions.csv"
        decisions_df.to_csv(decisions_path, index=False, encoding="utf-8")

        summary = self._build_summary(
            records=records,
            decisions_by_id=decisions_by_id,
            metrics_by_id=metrics_by_id,
            metadata=metadata,
            retained_df=retained_df,
            retained_ids=retained_ids,
            dropped_ids=dropped_ids,
            redundancy_resolutions=redundancy_resolutions,
        )
        summary_path = round_dir / "validation_summary.json"
        redundancy_path = round_dir / "validation_redundancy.json"
        llm_review_path = round_dir / "validation_llm_review.json"
        registry_path = run_dir / "feature_registry.json"

        self._dump_json(summary_path, summary)
        self._dump_json(
            redundancy_path,
            {
                "deterministic_groups": deterministic_groups,
                "ambiguous_groups": ambiguous_groups,
                "resolutions": redundancy_resolutions,
            },
        )
        self._dump_json(llm_review_path, llm_review_result)
        self._dump_json(registry_path, updated_registry)

        planner_feedback = self._build_planner_feedback(
            records=records,
            decisions_by_id=decisions_by_id,
            summary=summary,
            updated_registry=updated_registry,
            redundancy_resolutions=redundancy_resolutions,
        )

        return ValidationResult(
            retained_feature_names=[
                updated_registry["feature_id_to_column"][feature_id]
                for feature_id in retained_ids
                if feature_id in updated_registry["feature_id_to_column"]
            ],
            dropped_feature_names=[record.name for record in records if record.feature_id in dropped_ids],
            decisions=list(decisions_by_id.values()),
            summary=summary,
            updated_registry=updated_registry,
            planner_feedback=planner_feedback,
        )

    def _infer_round_number(self, round_dir: Path) -> int:
        match = re.search(r"round_(\d+)", round_dir.name)
        if not match:
            return 1
        return int(match.group(1))

    def _build_feature_records(
        self,
        current_feature_columns: Sequence[str],
        planned_features: Sequence[Dict[str, Any]],
        features_df: pd.DataFrame,
        round_number: int,
        round_dir: Path,
    ) -> List[FeatureRecord]:
        plan_lookup = {feature.get("name", ""): feature for feature in planned_features}
        records: List[FeatureRecord] = []
        for column in current_feature_columns:
            plan_feature = plan_lookup.get(column) or plan_lookup.get(TIMESTAMP_SUFFIX_RE.sub("", column), {})
            record = FeatureRecord(
                feature_id=f"round_{round_number}:{column}",
                round_number=round_number,
                name=column,
                canonical_name=_canonical_name(column),
                method=str(plan_feature.get("method", "unknown")),
                description=str(plan_feature.get("description", "")),
                category=str(plan_feature.get("category", "other")),
                values=features_df[column].astype(float).tolist(),
                source_paths={
                    "round_dir": str(round_dir),
                    "feature_plan_path": str(round_dir / "feature_plan.json"),
                    "raw_features_csv": str(round_dir.parent / "features.csv"),
                },
            )
            records.append(record)
        return records

    def _compute_feature_metrics(self, values: pd.Series, metadata: MetadataContext) -> FeatureMetrics:
        finite_mask = values.replace([np.inf, -np.inf], np.nan).notna()
        valid_values = pd.to_numeric(values[finite_mask], errors="coerce").dropna()
        availability_rate = float(finite_mask.mean()) if len(values) else 0.0
        finite_rate = availability_rate
        n_valid = int(len(valid_values))
        n_unique = int(valid_values.nunique())

        mean = _safe_float(valid_values.mean()) if n_valid else None
        std = _safe_float(valid_values.std(ddof=0)) if n_valid else None
        median = _safe_float(valid_values.median()) if n_valid else None
        q1 = _safe_float(valid_values.quantile(0.25)) if n_valid else None
        q3 = _safe_float(valid_values.quantile(0.75)) if n_valid else None
        iqr = _safe_float((q3 - q1) if q1 is not None and q3 is not None else None)
        dynamic_range = _safe_float(valid_values.max() - valid_values.min()) if n_valid else None
        zero_fraction = _safe_float(float((valid_values == 0).mean())) if n_valid else None

        cv = None
        if mean is not None and std is not None and abs(mean) > 1e-12:
            cv = _safe_float(std / abs(mean))

        robust_dispersion = None
        if median is not None and iqr is not None:
            robust_dispersion = _safe_float(iqr / (abs(median) + 1e-12))

        unsupervised_signal = max(cv or 0.0, robust_dispersion or 0.0)
        metadata_metrics = compute_metadata_alignment(values, metadata)
        coverage_score = availability_rate
        variability_score = min(1.0, unsupervised_signal / 0.2) if unsupervised_signal > 0 else 0.0
        metadata_score = metadata_metrics["metadata_alignment_score"]
        base_validation_score = (0.45 * coverage_score) + (0.35 * variability_score) + (0.20 * metadata_score)

        return FeatureMetrics(
            availability_rate=availability_rate,
            finite_rate=finite_rate,
            n_valid=n_valid,
            n_unique=n_unique,
            mean=mean,
            std=std,
            cv=cv,
            median=median,
            iqr=iqr,
            robust_dispersion=robust_dispersion,
            zero_fraction=zero_fraction,
            dynamic_range=dynamic_range,
            unsupervised_signal=unsupervised_signal,
            metadata_max_abs_spearman=metadata_metrics["metadata_max_abs_spearman"],
            metadata_best_spearman_field=metadata_metrics["metadata_best_spearman_field"],
            metadata_max_anova_f=metadata_metrics["metadata_max_anova_f"],
            metadata_best_anova_field=metadata_metrics["metadata_best_anova_field"],
            metadata_max_eta_squared=metadata_metrics["metadata_max_eta_squared"],
            metadata_best_eta_field=metadata_metrics["metadata_best_eta_field"],
            metadata_max_classifier_auc=metadata_metrics["metadata_max_classifier_auc"],
            metadata_best_classifier_field=metadata_metrics["metadata_best_classifier_field"],
            metadata_alignment_score=metadata_metrics["metadata_alignment_score"],
            redundancy_penalty=0.0,
            base_validation_score=base_validation_score,
            validation_score=base_validation_score,
        )

    def _decision_after_hard_filters(
        self,
        record: FeatureRecord,
        metrics: FeatureMetrics,
        metadata_available: bool = False,
    ) -> FeatureDecision:
        reason_codes: List[str] = []
        explanation_parts: List[str] = []

        if metrics.n_valid == 0:
            reason_codes.append("all_nan_or_non_finite")
            explanation_parts.append("All values are NaN or non-finite.")
        if metrics.availability_rate < self.min_availability_rate:
            reason_codes.append("low_availability")
            explanation_parts.append(
                f"Availability rate {metrics.availability_rate:.3f} is below {self.min_availability_rate:.2f}."
            )
        if metrics.n_unique < self.min_unique_values:
            reason_codes.append("low_unique_values")
            explanation_parts.append(
                f"Only {metrics.n_unique} unique valid values were observed."
            )
        if metrics.unsupervised_signal < self.cv_threshold:
            reason_codes.append("low_unsupervised_variability")
            explanation_parts.append(
                f"Variability signal {metrics.unsupervised_signal:.3f} is below {self.cv_threshold:.2f}."
            )
        if metadata_available:
            # Combined paired-metadata score (Spearman OR categorical eta² OR
            # classifier AUC). Categorical labels (e.g. WT/MU) are gated mainly
            # by AUC; continuous covariates mainly by Spearman/eta² score.
            # Previously only Spearman was checked, which dropped every feature
            # when metadata was categorical.
            passes_alignment = (
                metrics.metadata_alignment_score >= self.min_metadata_alignment
                or metrics.metadata_max_classifier_auc >= self.min_metadata_classifier_auc
            )
            if not passes_alignment:
                reason_codes.append("low_paired_metadata_alignment")
                explanation_parts.append(
                    "Paired-metadata alignment is below the loose screen "
                    f"(score={metrics.metadata_alignment_score:.3f} < {self.min_metadata_alignment:.2f}, "
                    f"classifier AUC={metrics.metadata_max_classifier_auc:.3f} < "
                    f"{self.min_metadata_classifier_auc:.2f}). "
                    "Only features with essentially no ability to track paired metadata are dropped."
                )
        if metrics.base_validation_score < self.min_base_validation_score:
            reason_codes.append("low_validation_score")
            explanation_parts.append(
                f"Base validation score {metrics.base_validation_score:.3f} "
                f"is below {self.min_base_validation_score:.2f}."
            )

        status = "dropped" if reason_codes else "retained"
        explanation = " ".join(explanation_parts) if explanation_parts else "Passed deterministic hard filters."
        return FeatureDecision(
            feature_id=record.feature_id,
            feature_name=record.name,
            status=status,
            reason_codes=reason_codes if reason_codes else ["passed_hard_filters"],
            explanation=explanation,
            compared_feature_ids=[],
            validation_score=metrics.validation_score,
            actual_column_name=record.name,
            llm_reviewed=False,
        )

    def _lookup_record(
        self,
        feature_id: str,
        records: Sequence[FeatureRecord],
        registry_entries: Dict[str, RegistryEntry],
    ) -> Optional[FeatureRecord]:
        for record in records:
            if record.feature_id == feature_id:
                return record
        entry = registry_entries.get(feature_id)
        if entry is None:
            return None
        return FeatureRecord(
            feature_id=entry.feature_id,
            round_number=entry.latest_round,
            name=entry.actual_column_name,
            canonical_name=entry.canonical_name,
            method=entry.method,
            description=entry.description,
            category=entry.category,
            values=[],
            source_paths=entry.source_paths,
        )

    def _build_historical_live_records(
        self,
        features_df: pd.DataFrame,
        live_entries: Dict[str, RegistryEntry],
    ) -> Dict[str, FeatureRecord]:
        historical_records: Dict[str, FeatureRecord] = {}
        for feature_id, entry in live_entries.items():
            column_name = entry.actual_column_name
            if column_name not in features_df.columns:
                continue
            historical_records[feature_id] = FeatureRecord(
                feature_id=entry.feature_id,
                round_number=entry.latest_round,
                name=entry.actual_column_name,
                canonical_name=entry.canonical_name,
                method=entry.method,
                description=entry.description,
                category=entry.category,
                values=features_df[column_name].astype(float).tolist(),
                source_paths=entry.source_paths,
            )
        return historical_records

    def _choose_stronger_feature(
        self,
        left_id: str,
        right_id: str,
        metrics_by_id: Dict[str, FeatureMetrics],
        live_entries: Dict[str, RegistryEntry],
        registry_entries: Dict[str, RegistryEntry],
        record_rank_by_id: Dict[str, int],
    ) -> Tuple[str, str]:
        left_metrics = metrics_by_id.get(left_id)
        right_metrics = metrics_by_id.get(right_id)
        left_score = left_metrics.base_validation_score if left_metrics else 0.0
        right_score = right_metrics.base_validation_score if right_metrics else 0.0
        if right_score > left_score:
            return right_id, left_id
        if left_score > right_score:
            return left_id, right_id

        left_availability = left_metrics.availability_rate if left_metrics else 0.0
        right_availability = right_metrics.availability_rate if right_metrics else 0.0
        if right_availability > left_availability:
            return right_id, left_id
        if left_availability > right_availability:
            return left_id, right_id

        left_live = left_id in live_entries
        right_live = right_id in live_entries
        if left_live and not right_live:
            return left_id, right_id
        if right_live and not left_live:
            return right_id, left_id

        left_round = self._feature_origin_round(left_id, registry_entries)
        right_round = self._feature_origin_round(right_id, registry_entries)
        if left_round != right_round:
            return (left_id, right_id) if left_round < right_round else (right_id, left_id)

        left_rank = record_rank_by_id.get(left_id)
        right_rank = record_rank_by_id.get(right_id)
        if left_rank is not None and right_rank is not None and left_rank != right_rank:
            return (left_id, right_id) if left_rank < right_rank else (right_id, left_id)

        return left_id, right_id

    def _feature_origin_round(
        self,
        feature_id: str,
        registry_entries: Dict[str, RegistryEntry],
    ) -> int:
        if feature_id in registry_entries:
            return registry_entries[feature_id].first_round
        match = re.search(r"round_(\d+):", feature_id)
        if match:
            return int(match.group(1))
        return 10 ** 9

    def _mark_deterministic_drop(
        self,
        kept_id: str,
        dropped_id: str,
        decisions_by_id: Dict[str, FeatureDecision],
        metrics_by_id: Dict[str, FeatureMetrics],
        deterministic_groups: List[Dict[str, Any]],
        reason_code: str,
        superseded_live_ids: Set[str],
        live_entries: Dict[str, RegistryEntry],
        corr: Optional[float] = None,
    ) -> None:
        if dropped_id in decisions_by_id:
            decisions_by_id[dropped_id].status = "dropped"
            decisions_by_id[dropped_id].reason_codes = [reason_code]
            decisions_by_id[dropped_id].compared_feature_ids = [kept_id]
            decisions_by_id[dropped_id].explanation = (
                f"Dropped in favor of {kept_id} due to {reason_code}."
            )
            metrics_by_id[dropped_id].redundancy_penalty = max(
                metrics_by_id[dropped_id].redundancy_penalty,
                0.35,
            )
        elif dropped_id in live_entries:
            superseded_live_ids.add(dropped_id)
        deterministic_groups.append(
            {
                "kept_feature_id": kept_id,
                "dropped_feature_id": dropped_id,
                "reason_code": reason_code,
                "correlation": corr,
            }
        )

    def _pairwise_correlation(self, left_values: Sequence[Optional[float]], right_values: Sequence[Optional[float]]) -> Optional[float]:
        if not left_values or not right_values:
            return None
        left = pd.Series(left_values, dtype=float)
        right = pd.Series(right_values, dtype=float)
        pair = pd.DataFrame({"left": left, "right": right}).dropna()
        if len(pair) < 10:
            return None
        if pair["left"].nunique() < 2 or pair["right"].nunique() < 2:
            return None
        corr = pair["left"].corr(pair["right"])
        return _safe_float(corr)

    def _semantic_similarity(self, left: FeatureRecord, right: FeatureRecord) -> float:
        left_tokens = _tokenize_concept(left.name + " " + left.description)
        right_tokens = _tokenize_concept(right.name + " " + right.description)
        if not left_tokens or not right_tokens:
            return 0.0
        intersection = len(left_tokens & right_tokens)
        union = len(left_tokens | right_tokens)
        return intersection / union if union else 0.0

    def _build_ambiguous_pair(
        self,
        left_record: FeatureRecord,
        right_record: FeatureRecord,
        corr: Optional[float],
        trigger: str,
    ) -> Dict[str, Any]:
        return {
            "left_feature_id": left_record.feature_id,
            "right_feature_id": right_record.feature_id,
            "left_name": left_record.name,
            "right_name": right_record.name,
            "left_round": left_record.round_number,
            "right_round": right_record.round_number,
            "correlation": corr,
            "trigger": trigger,
            "left_description": left_record.description,
            "right_description": right_record.description,
        }

    def _group_ambiguous_pairs(self, ambiguous_pairs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not ambiguous_pairs:
            return []
        adjacency: Dict[str, Set[str]] = defaultdict(set)
        pair_lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for pair in ambiguous_pairs:
            left_id = pair["left_feature_id"]
            right_id = pair["right_feature_id"]
            adjacency[left_id].add(right_id)
            adjacency[right_id].add(left_id)
            pair_lookup[tuple(sorted((left_id, right_id)))] = pair

        visited: Set[str] = set()
        groups: List[Dict[str, Any]] = []
        for node in list(adjacency):
            if node in visited:
                continue
            stack = [node]
            component: Set[str] = set()
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                stack.extend(adjacency[current] - visited)
            if len(component) < 2:
                continue
            sorted_ids = sorted(component)
            group_pairs = []
            for left_index, left_id in enumerate(sorted_ids):
                for right_id in sorted_ids[left_index + 1 :]:
                    pair_key = tuple(sorted((left_id, right_id)))
                    if pair_key in pair_lookup:
                        group_pairs.append(pair_lookup[pair_key])
            groups.append(
                {
                    "group_id": f"group_{len(groups) + 1}",
                    "feature_ids": sorted_ids,
                    "pairs": group_pairs,
                }
            )
        return groups

    def _apply_llm_review(
        self,
        ambiguous_groups: Sequence[Dict[str, Any]],
        llm_review_result: Dict[str, Any],
        decisions_by_id: Dict[str, FeatureDecision],
        metrics_by_id: Dict[str, FeatureMetrics],
        live_entries: Dict[str, RegistryEntry],
        registry_entries: Dict[str, RegistryEntry],
        record_rank_by_id: Dict[str, int],
        superseded_live_ids: Set[str],
    ) -> List[Dict[str, Any]]:
        resolutions: List[Dict[str, Any]] = []
        if not ambiguous_groups:
            return resolutions

        parsed_by_group = {
            group_result.get("group_id"): group_result
            for group_result in llm_review_result.get("groups", [])
            if isinstance(group_result, dict)
        }

        for group in ambiguous_groups:
            feature_ids = [
                feature_id
                for feature_id in group["feature_ids"]
                if feature_id not in decisions_by_id or decisions_by_id[feature_id].status != "dropped"
            ]
            if len(feature_ids) < 2:
                continue
            parsed = parsed_by_group.get(group["group_id"]) if llm_review_result.get("success") else None
            if parsed:
                keep_ids = [
                    feature_id
                    for feature_id in parsed.get("keep_feature_ids", [])
                    if feature_id in feature_ids
                ]
                if not keep_ids:
                    keep_ids = [
                        self._fallback_keep_id(
                            feature_ids,
                            metrics_by_id,
                            live_entries,
                            registry_entries,
                            record_rank_by_id,
                        )
                    ]
                decision_label = parsed.get("decision", "keep_old")
                if decision_label == "keep_both" and self._group_allows_keep_both(group):
                    keep_ids = [feature_id for feature_id in feature_ids if feature_id in keep_ids]
                elif decision_label == "keep_new":
                    keep_ids = keep_ids[:1]
                else:
                    keep_ids = keep_ids[:1]
                rationale = str(parsed.get("rationale", "Resolved by LLM review."))
                success = True
            else:
                keep_ids = [
                    self._fallback_keep_id(
                        feature_ids,
                        metrics_by_id,
                        live_entries,
                        registry_entries,
                        record_rank_by_id,
                    )
                ]
                decision_label = "keep_old"
                rationale = "LLM review unavailable; applied deterministic conservative fallback."
                success = False

            if decision_label == "keep_both" and len(keep_ids) < 2:
                decision_label = "keep_old"
                rationale = (
                    rationale
                    + " Keep-both was overridden because the group did not satisfy conservative complementarity rules."
                )

            drop_ids = [feature_id for feature_id in feature_ids if feature_id not in keep_ids]
            for feature_id in feature_ids:
                if feature_id not in decisions_by_id:
                    if feature_id in drop_ids and feature_id in live_entries:
                        superseded_live_ids.add(feature_id)
                    continue
                metrics_by_id[feature_id].redundancy_penalty = max(
                    metrics_by_id[feature_id].redundancy_penalty,
                    0.10 if feature_id in keep_ids and len(keep_ids) > 1 else 0.25 if feature_id in drop_ids else 0.0,
                )
                if feature_id in drop_ids:
                    decisions_by_id[feature_id].status = "dropped"
                    decisions_by_id[feature_id].reason_codes = ["reviewed_keep_other"]
                    decisions_by_id[feature_id].explanation = rationale
                    decisions_by_id[feature_id].compared_feature_ids = [candidate for candidate in keep_ids]
                    decisions_by_id[feature_id].llm_reviewed = success
                elif len(keep_ids) > 1:
                    # Keep both still counts as retained; nuance lives in reason_codes.
                    decisions_by_id[feature_id].status = "retained"
                    decisions_by_id[feature_id].reason_codes = ["reviewed_complementary_keep_both"]
                    decisions_by_id[feature_id].explanation = rationale
                    decisions_by_id[feature_id].compared_feature_ids = [candidate for candidate in feature_ids if candidate != feature_id]
                    decisions_by_id[feature_id].llm_reviewed = success
                else:
                    decisions_by_id[feature_id].status = "retained"
                    decisions_by_id[feature_id].reason_codes = [
                        "reviewed_keep_old" if feature_id in live_entries else "reviewed_keep_new"
                    ]
                    decisions_by_id[feature_id].explanation = rationale
                    decisions_by_id[feature_id].compared_feature_ids = drop_ids
                    decisions_by_id[feature_id].llm_reviewed = success

            resolutions.append(
                {
                    "group_id": group["group_id"],
                    "success": success,
                    "decision": decision_label,
                    "kept_feature_ids": keep_ids,
                    "dropped_feature_ids": drop_ids,
                    "rationale": rationale,
                }
            )

        return resolutions

    def _fallback_keep_id(
        self,
        feature_ids: Sequence[str],
        metrics_by_id: Dict[str, FeatureMetrics],
        live_entries: Dict[str, RegistryEntry],
        registry_entries: Dict[str, RegistryEntry],
        record_rank_by_id: Dict[str, int],
    ) -> str:
        keep_id = feature_ids[0]
        for candidate in feature_ids[1:]:
            keep_id, _ = self._choose_stronger_feature(
                keep_id,
                candidate,
                metrics_by_id,
                live_entries,
                registry_entries,
                record_rank_by_id,
            )
        return keep_id

    def _group_allows_keep_both(self, group: Dict[str, Any]) -> bool:
        for pair in group.get("pairs", []):
            corr = pair.get("correlation")
            if corr is not None and abs(float(corr)) >= self.deterministic_corr_threshold:
                return False
        return True

    def _update_registry(
        self,
        prior_registry: Dict[str, Any],
        registry_entries: Dict[str, RegistryEntry],
        records: Sequence[FeatureRecord],
        decisions_by_id: Dict[str, FeatureDecision],
        round_number: int,
        current_raw_names: Sequence[str],
        superseded_live_ids: Set[str],
    ) -> Dict[str, Any]:
        live_feature_ids = set(prior_registry.get("live_feature_ids", []))
        feature_id_to_column = dict(prior_registry.get("feature_id_to_column", {}))

        for superseded_id in superseded_live_ids:
            if superseded_id in registry_entries:
                registry_entries[superseded_id].live = False
                registry_entries[superseded_id].current_status = "dropped"
                registry_entries[superseded_id].decision_history.append(
                    {
                        "feature_id": superseded_id,
                        "feature_name": registry_entries[superseded_id].actual_column_name,
                        "status": "dropped",
                        "reason_codes": ["superseded_by_new_feature"],
                        "explanation": "A newer feature replaced this live retained feature during redundancy resolution.",
                        "compared_feature_ids": [],
                        "validation_score": 0.0,
                        "actual_column_name": registry_entries[superseded_id].actual_column_name,
                        "llm_reviewed": False,
                    }
                )
                live_feature_ids.discard(superseded_id)

        for record in records:
            decision = decisions_by_id[record.feature_id]
            entry = registry_entries.get(record.feature_id)
            if entry is None:
                entry = RegistryEntry(
                    feature_id=record.feature_id,
                    name=record.name,
                    canonical_name=record.canonical_name,
                    method=record.method,
                    description=record.description,
                    category=record.category,
                    first_round=round_number,
                    latest_round=round_number,
                    current_status=decision.status,
                    live=decision.status != "dropped",
                    actual_column_name=record.name,
                    source_paths=record.source_paths,
                    decision_history=[],
                )
                registry_entries[record.feature_id] = entry
            entry.latest_round = round_number
            entry.current_status = decision.status
            entry.live = decision.status != "dropped"
            entry.actual_column_name = decision.actual_column_name or record.name
            entry.decision_history.append(decision.to_dict())
            feature_id_to_column[record.feature_id] = entry.actual_column_name
            if entry.live:
                live_feature_ids.add(record.feature_id)
            else:
                live_feature_ids.discard(record.feature_id)

        return {
            "version": 1,
            "entries": [entry.to_dict() for entry in registry_entries.values()],
            "live_feature_ids": sorted(live_feature_ids),
            "feature_id_to_column": feature_id_to_column,
            "all_raw_feature_names": list(current_raw_names),
            "all_historical_feature_names": sorted(
                {
                    entry.actual_column_name
                    for entry in registry_entries.values()
                }
            ),
        }

    def _build_retained_features_df(self, raw_features_df: pd.DataFrame, updated_registry: Dict[str, Any]) -> pd.DataFrame:
        live_columns = []
        for feature_id in updated_registry.get("live_feature_ids", []):
            column = updated_registry.get("feature_id_to_column", {}).get(feature_id)
            if column and column in raw_features_df.columns and column not in live_columns:
                live_columns.append(column)
        retained_columns = ["sample_id"] + live_columns
        return raw_features_df[retained_columns].copy()

    def _build_decisions_df(
        self,
        records: Sequence[FeatureRecord],
        metrics_by_id: Dict[str, FeatureMetrics],
        decisions_by_id: Dict[str, FeatureDecision],
    ) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for record in records:
            metrics = metrics_by_id[record.feature_id]
            decision = decisions_by_id[record.feature_id]
            row = {
                "feature_id": record.feature_id,
                "feature_name": record.name,
                "canonical_name": record.canonical_name,
                "method": record.method,
                "category": record.category,
                "status": decision.status,
                "reason_codes": "|".join(decision.reason_codes),
                "explanation": decision.explanation,
                "compared_feature_ids": "|".join(decision.compared_feature_ids),
                "llm_reviewed": decision.llm_reviewed,
            }
            row.update(metrics.to_dict())
            rows.append(row)
        return pd.DataFrame(rows)

    def _build_summary(
        self,
        records: Sequence[FeatureRecord],
        decisions_by_id: Dict[str, FeatureDecision],
        metrics_by_id: Dict[str, FeatureMetrics],
        metadata: MetadataContext,
        retained_df: pd.DataFrame,
        retained_ids: Sequence[str],
        dropped_ids: Sequence[str],
        redundancy_resolutions: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        retained_records = [record for record in records if record.feature_id in retained_ids]
        dropped_records = [record for record in records if record.feature_id in dropped_ids]
        reason_counts = Counter()
        for decision in decisions_by_id.values():
            reason_counts.update(decision.reason_codes)

        top_retained = sorted(
            retained_records,
            key=lambda record: metrics_by_id[record.feature_id].validation_score,
            reverse=True,
        )[:10]
        summary = {
            "thresholds": {
                "availability_rate": self.min_availability_rate,
                "min_unique_values": self.min_unique_values,
                "cv_floor": self.cv_threshold,
                "deterministic_corr_threshold": self.deterministic_corr_threshold,
                "ambiguous_corr_threshold": self.ambiguous_corr_threshold,
                "min_metadata_alignment": self.min_metadata_alignment,
                "min_metadata_classifier_auc": self.min_metadata_classifier_auc,
                # Legacy key kept for older readers of validation_summary.json.
                "min_transcriptome_alignment": self.min_metadata_alignment,
                "min_base_validation_score": self.min_base_validation_score,
            },
            "metadata": metadata.to_summary(),
            "counts": {
                "round_features": len(records),
                "retained": len(retained_records),
                "dropped": len(dropped_records),
                "retained_total_live": max(retained_df.shape[1] - 1, 0),
            },
            "retained_features": [record.name for record in retained_records],
            "dropped_features": [record.name for record in dropped_records],
            "top_reason_codes": reason_counts.most_common(10),
            "top_retained_features": [
                {
                    "feature_name": record.name,
                    "validation_score": metrics_by_id[record.feature_id].validation_score,
                    "availability_rate": metrics_by_id[record.feature_id].availability_rate,
                    "unsupervised_signal": metrics_by_id[record.feature_id].unsupervised_signal,
                }
                for record in top_retained
            ],
            "redundancy_resolutions": list(redundancy_resolutions),
        }
        return _normalize_json_value(summary)

    def _build_planner_feedback(
        self,
        records: Sequence[FeatureRecord],
        decisions_by_id: Dict[str, FeatureDecision],
        summary: Dict[str, Any],
        updated_registry: Dict[str, Any],
        redundancy_resolutions: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        retained_records = [
            record
            for record in records
            if decisions_by_id[record.feature_id].status == "retained"
        ]
        dropped_records = [
            record
            for record in records
            if decisions_by_id[record.feature_id].status == "dropped"
        ]
        return {
            "retained_features": [
                {
                    "name": record.name,
                    "status": decisions_by_id[record.feature_id].status,
                    "score": decisions_by_id[record.feature_id].validation_score,
                }
                for record in retained_records
            ],
            "dropped_features": [
                {
                    "name": record.name,
                    "status": decisions_by_id[record.feature_id].status,
                    "reasons": decisions_by_id[record.feature_id].reason_codes,
                }
                for record in dropped_records
            ],
            "redundancy_resolutions": list(redundancy_resolutions),
            "top_reason_codes": summary.get("top_reason_codes", []),
            "all_historical_feature_names": updated_registry.get("all_historical_feature_names", []),
        }

    def _dump_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(_normalize_json_value(payload), handle, indent=2, ensure_ascii=False)
