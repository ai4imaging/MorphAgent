"""Typed records for deterministic validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FeatureRecord:
    """Runtime description of one extracted feature."""

    feature_id: str
    round_number: int
    name: str
    canonical_name: str
    method: str
    description: str
    category: str
    values: List[Optional[float]]
    source_paths: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureMetrics:
    """Deterministic quality metrics for one feature."""

    availability_rate: float
    finite_rate: float
    n_valid: int
    n_unique: int
    mean: Optional[float]
    std: Optional[float]
    cv: Optional[float]
    median: Optional[float]
    iqr: Optional[float]
    robust_dispersion: Optional[float]
    zero_fraction: Optional[float]
    dynamic_range: Optional[float]
    unsupervised_signal: float
    metadata_max_abs_spearman: float = 0.0
    metadata_best_spearman_field: Optional[str] = None
    metadata_max_anova_f: float = 0.0
    metadata_best_anova_field: Optional[str] = None
    metadata_max_eta_squared: float = 0.0
    metadata_best_eta_field: Optional[str] = None
    metadata_alignment_score: float = 0.0
    redundancy_penalty: float = 0.0
    base_validation_score: float = 0.0
    validation_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureDecision:
    """Decision taken for a feature in a round."""

    feature_id: str
    feature_name: str
    status: str
    reason_codes: List[str]
    explanation: str
    compared_feature_ids: List[str] = field(default_factory=list)
    validation_score: float = 0.0
    actual_column_name: Optional[str] = None
    llm_reviewed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RegistryEntry:
    """Persistent lifecycle state for one feature."""

    feature_id: str
    name: str
    canonical_name: str
    method: str
    description: str
    category: str
    first_round: int
    latest_round: int
    current_status: str
    live: bool
    actual_column_name: str
    source_paths: Dict[str, str] = field(default_factory=dict)
    decision_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    """Return value for round validation."""

    retained_feature_names: List[str]
    dropped_feature_names: List[str]
    decisions: List[FeatureDecision]
    summary: Dict[str, Any]
    updated_registry: Dict[str, Any]
    planner_feedback: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retained_feature_names": self.retained_feature_names,
            "dropped_feature_names": self.dropped_feature_names,
            "decisions": [decision.to_dict() for decision in self.decisions],
            "summary": self.summary,
            "updated_registry": self.updated_registry,
            "planner_feedback": self.planner_feedback,
        }
