"""Lite Deep Research: prepared txt or one configured-model synthesis call."""

from pathlib import Path
from typing import Optional

from knowledge.precomputed_lite import load_or_generate_summary


def extract_deep_research(
    project_root: Path,
    enable_deep_research: bool = True,
    device: str = "gpu:0",
    user_query: str = "",
    dataset_description: str = "",
) -> Optional[str]:
    """Return prepared Deep Research, or synthesize it from the biological question."""

    del device  # CLI compatibility; Lite never runs PDF/OCR.
    if not enable_deep_research:
        return None
    return load_or_generate_summary(
        project_root,
        "deep_research",
        user_query,
        dataset_description,
    )
