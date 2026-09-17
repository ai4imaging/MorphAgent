"""Qt-free workload and elapsed-time estimates (approximate, not deadlines)."""
from dataclasses import dataclass

from .models import DatasetSummary, RunConfig


def estimate_run_seconds(config: RunConfig, dataset: DatasetSummary | None) -> int:
    """Rough initial runtime estimate from rounds, features, images, and routes."""

    rounds = max(1, int(config.num_rounds))
    features = max(1, int(config.features_per_iteration))
    if dataset is None:
        images = 1
    else:
        # Primary and VLM sources often refer to the same image, so avoid summing.
        images = max(
            1,
            int(dataset.sample_count),
            int(dataset.primary_image_count),
            int(dataset.vlm_source_count),
        )

    fixed_seconds = 75.0 + images * 2.0
    knowledge_sources = sum((
        bool(config.enable_expert_knowledge),
        bool(config.enable_deep_research),
        bool(config.enable_rag),
    ))
    fixed_seconds += knowledge_sources * 90.0

    per_round = 45.0 + 30.0 + features * 1.5  # planning + validation
    if config.method in {"code", "both"}:
        workers = max(1, int(config.code_parallel_workers))
        per_round += features * (12.0 + images * 0.5) / workers
    if config.method in {"vlm", "both"}:
        concurrency = max(1, int(config.vlm_online_concurrency))
        per_round += features * images * 4.0 / concurrency
    return max(120, int(round(fixed_seconds + rounds * per_round)))


@dataclass
class DynamicEta:
    """Blend the initial workload estimate with observed progress and elapsed time."""

    initial_total_seconds: int
    num_rounds: int
    progress_percent: int = 5
    completed_rounds: int = 0

    def update_progress(self, percent: int) -> None:
        self.progress_percent = max(self.progress_percent, min(100, int(percent)))

    def update_completed_rounds(self, count: int) -> None:
        self.completed_rounds = max(self.completed_rounds, max(0, int(count)))

    @property
    def effective_progress(self) -> int:
        # Completed rounds provide useful progress while the CLI remains in Quantify.
        round_progress = 0
        if self.num_rounds > 0 and self.completed_rounds > 0:
            round_progress = 40 + int(
                42 * min(self.completed_rounds, self.num_rounds) / self.num_rounds
            )
        return max(self.progress_percent, round_progress)

    def remaining_seconds(self, elapsed_seconds: int) -> int:
        elapsed = max(0, int(elapsed_seconds))
        if self.effective_progress >= 100:
            return 0
        fraction = max(0.01, self.effective_progress / 100.0)
        if elapsed < 15:
            predicted_total = float(self.initial_total_seconds)
        else:
            observed_total = elapsed / fraction
            # Increase trust in observed speed as the run advances.
            observed_weight = min(0.75, 0.15 + fraction * 0.5)
            predicted_total = (
                self.initial_total_seconds * (1.0 - observed_weight)
                + observed_total * observed_weight
            )
        predicted_total = max(predicted_total, elapsed + 5.0)
        return max(0, int(round(predicted_total - elapsed)))
