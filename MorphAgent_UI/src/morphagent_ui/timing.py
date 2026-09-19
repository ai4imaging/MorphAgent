"""Qt-free workload and elapsed-time estimates (approximate, not deadlines)."""
from dataclasses import dataclass

from .models import DatasetSummary, RunConfig


# Calibrated against the BBBC021 run described in the tutorial: over its 3,552
# samples, 438 code features took about two hours across sixteen processes and
# the batched vision calls took about fourteen hours.
PREPARE_SECONDS_PER_IMAGE = 0.05
AUTHOR_SECONDS_PER_FEATURE = 35.0
MERGE_SECONDS_PER_ROUND = 30.0
EXTRACT_SECONDS_PER_FEATURE_IMAGE = 0.05
# Replaying saved code merges the selection into one sandbox per sample, so each
# sample pays for a process start and an image decode on top of the features.
REPLAY_SECONDS_PER_SAMPLE = 0.6
VLM_SECONDS_PER_SAMPLE = 6.0
VLM_SECONDS_PER_SAMPLE_FEATURE = 0.28
# Planners have split "both" runs from 6% VLM (BBBC021) to 21% (Tau).
VLM_SHARE_OF_BOTH = 0.25


def estimate_run_seconds(config: RunConfig, dataset: DatasetSummary | None) -> int:
    """Rough initial runtime estimate from rounds, features, images, and routes.

    Authoring a feature costs the same on ten images as on ten thousand: the
    generated code is tested on a single sample, and the round's features are
    then merged into one pass over the dataset. So the image count multiplies
    only that pass and the per-sample vision calls, not the feature count.
    """

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

    fixed_seconds = 75.0 + images * PREPARE_SECONDS_PER_IMAGE
    knowledge_sources = sum((
        bool(config.enable_expert_knowledge),
        bool(config.enable_deep_research),
        bool(config.enable_rag),
    ))
    fixed_seconds += knowledge_sources * 90.0

    if config.method == "code":
        vlm_features, code_features = 0.0, float(features)
    elif config.method == "vlm":
        vlm_features, code_features = float(features), 0.0
    else:
        vlm_features = features * VLM_SHARE_OF_BOTH
        code_features = features - vlm_features

    per_round = 45.0 + 30.0 + features * 1.5  # planning + validation
    if code_features:
        authors = max(1, int(config.code_gen_workers))
        extractors = max(1, int(config.code_parallel_workers))
        per_round += code_features * AUTHOR_SECONDS_PER_FEATURE / authors
        per_round += MERGE_SECONDS_PER_ROUND
        per_round += (
            images * code_features * EXTRACT_SECONDS_PER_FEATURE_IMAGE / extractors
        )
    if vlm_features:
        # One batched call per sample covers every VLM feature of the round.
        concurrency = max(1, int(config.vlm_online_concurrency))
        per_sample = (
            VLM_SECONDS_PER_SAMPLE + VLM_SECONDS_PER_SAMPLE_FEATURE * vlm_features
        )
        per_round += images * per_sample / concurrency
    return max(120, int(round(fixed_seconds + rounds * per_round)))


def estimate_reuse_seconds(code_features: int, vlm_features: int, samples: int,
                           vlm_concurrency: int) -> int:
    """Initial runtime estimate for replaying selected features on a dataset.

    Nothing is authored here, so the cost is local execution of saved code plus
    one batched vision call per sample when VLM features are selected.
    """

    samples = max(0, int(samples))
    code_features = max(0, int(code_features))
    seconds = 20.0
    if code_features:
        seconds += samples * (
            REPLAY_SECONDS_PER_SAMPLE
            + code_features * EXTRACT_SECONDS_PER_FEATURE_IMAGE
        )
    if vlm_features > 0:
        per_sample = (
            VLM_SECONDS_PER_SAMPLE + VLM_SECONDS_PER_SAMPLE_FEATURE * int(vlm_features)
        )
        seconds += samples * per_sample / max(1, int(vlm_concurrency))
    return max(30, int(round(seconds)))


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
