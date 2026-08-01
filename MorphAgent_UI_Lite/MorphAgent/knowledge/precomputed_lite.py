"""Lite: load prepared knowledge summaries (txt) for prompt injection.

Skips PDF parsing, PubMed fetch, and online deep-research generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# Filename under demo/precomputed/ (sibling of demo/data/).
PRECOMPUTED_NAMES = {
    "expert": "expert_knowledge_summary.txt",
    "deep_research": "deep_research_summary.txt",
    "rag": "rag_knowledge_summary.txt",
}


def precomputed_dirs(project_root: Path) -> list[Path]:
    """Candidate folders that may hold prepared knowledge txt files."""

    root = Path(project_root).expanduser().resolve()
    return [
        root / "precomputed",
        root.parent / "precomputed",
        root / ".knowledge_precomputed",
    ]


def load_precomputed_summary(project_root: Path, kind: str) -> Optional[str]:
    """Return prepared summary text for kind in {expert, deep_research, rag}."""

    filename = PRECOMPUTED_NAMES.get(kind)
    if not filename:
        return None
    for directory in precomputed_dirs(project_root):
        path = directory / filename
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text and not text.startswith("[ERROR]"):
            print(f"  [{kind}] Using precomputed summary: {path}")
            return text
    return None
