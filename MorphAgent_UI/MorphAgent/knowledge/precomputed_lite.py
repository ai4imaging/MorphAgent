"""Load or generate lightweight knowledge summaries for prompt injection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

# Filename under demo/precomputed/ (sibling of demo/data/).
PRECOMPUTED_NAMES = {
    "expert": "expert_knowledge_summary.txt",
    "deep_research": "deep_research_summary.txt",
    "rag": "rag_knowledge_summary.txt",
}

DEEP_RESEARCH_SYSTEM_PROMPT = """You are a biomedical deep-research analyst helping design quantitative microscopy features.

Given only a biological question and a dataset description, produce a compact, structured research brief from your established scientific knowledge. Focus on:
1. biological mechanisms, compartments, structures, and phenotypes relevant to the question;
2. expected morphological changes and spatial relationships visible in microscopy;
3. measurable image features that Code or VLM routes can quantify;
4. confounders, controls, and interpretation pitfalls;
5. concrete feature-design recommendations.

Do not claim that you searched the web, accessed papers, or verified current literature. Do not invent citations, paper titles, authors, identifiers, or numerical findings. Clearly label uncertain or context-dependent statements. Write in English and make the result directly usable as background knowledge in a feature-planning prompt."""

LITERATURE_SUMMARY_SYSTEM_PROMPT = """You are a scientific literature-synthesis assistant helping design quantitative microscopy features.

Given only a biological question and a dataset description, create a literature-style synthesis from your established scientific knowledge. Organize it as:
1. consensus biological concepts relevant to the question;
2. commonly reported microscopy phenotypes and their visual appearance;
3. established or plausible quantitative morphology measurements;
4. competing interpretations, limitations, and validation checks;
5. a concise list of literature-grounded feature hypotheses for Code and VLM scoring.

This is not live retrieval. Do not state that PubMed, the web, or specific documents were searched. Do not fabricate citations, titles, authors, DOIs, PMIDs, or exact statistics. Distinguish broad consensus from hypotheses. Write in English and make the result directly usable in a downstream feature-planning prompt."""

SYSTEM_PROMPTS = {
    "deep_research": DEEP_RESEARCH_SYSTEM_PROMPT,
    "rag": LITERATURE_SUMMARY_SYSTEM_PROMPT,
}


def precomputed_dirs(project_root: Path) -> list[Path]:
    """Candidate folders that may hold prepared knowledge txt files."""

    root = Path(project_root).expanduser().resolve()
    return [
        root / "precomputed",
        root.parent / "precomputed",
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


def generated_summary_path(project_root: Path, kind: str) -> Optional[Path]:
    """Return the project-local cache path for a generated summary."""

    filename = PRECOMPUTED_NAMES.get(kind)
    if not filename:
        return None
    return Path(project_root).expanduser().resolve() / ".knowledge_precomputed" / filename


def _summary_fingerprint(user_query: str, dataset_description: str) -> str:
    payload = f"{user_query.strip()}\n\n{dataset_description.strip()}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_generated_summary(
    project_root: Path,
    kind: str,
    user_query: str,
    dataset_description: str = "",
) -> Optional[str]:
    """Reuse a generated cache only when its question/context fingerprint matches."""

    path = generated_summary_path(project_root, kind)
    if path is None:
        return None
    metadata_path = path.with_suffix(path.suffix + ".meta.json")
    if not path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = _summary_fingerprint(user_query, dataset_description)
        if metadata.get("fingerprint") != expected:
            return None
        text = path.read_text(encoding="utf-8").strip()
    except (OSError, ValueError, TypeError):
        return None
    if text and not text.startswith("[ERROR]"):
        print(f"  [{kind}] Reusing generated summary: {path}")
        return text
    return None


def generate_summary_from_question(
    project_root: Path,
    kind: str,
    user_query: str,
    dataset_description: str = "",
) -> Optional[str]:
    """Generate and cache a Deep Research or literature summary with the configured LLM."""

    system_prompt = SYSTEM_PROMPTS.get(kind)
    cache_path = generated_summary_path(project_root, kind)
    query = (user_query or "").strip()
    if not system_prompt or cache_path is None or not query:
        print(f"  [{kind}] Biological question is empty; skipping")
        return None

    from config import make_chat_llm, settings

    dataset_context = (dataset_description or "").strip()
    if len(dataset_context) > 30_000:
        dataset_context = dataset_context[:30_000] + "\n[Dataset description truncated]"
    human_prompt = (
        f"Biological question:\n{query}\n\n"
        f"Dataset description:\n{dataset_context or 'No additional dataset description was provided.'}\n\n"
        "Generate the requested summary now."
    )
    print(f"  [{kind}] No prepared txt found; generating summary with model {settings.llm_model}")
    try:
        response = make_chat_llm(
            temperature=0,
            max_tokens=min(int(settings.llm_max_tokens), 8_000),
        ).invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ])
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = "\n".join(
                str(item.get("text", item)) if isinstance(item, dict) else str(item)
                for item in content
            )
        summary = str(content or "").strip()
    except Exception as exc:
        print(f"  [{kind}] Model summary generation failed; skipping: {exc}")
        return None

    if not summary or summary.startswith("[ERROR]"):
        print(f"  [{kind}] Model returned an empty/error summary; skipping")
        return None

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        temporary.write_text(summary + "\n", encoding="utf-8")
        temporary.replace(cache_path)
        metadata_path = cache_path.with_suffix(cache_path.suffix + ".meta.json")
        metadata_path.write_text(
            json.dumps(
                {
                    "kind": kind,
                    "fingerprint": _summary_fingerprint(query, dataset_description),
                    "model": str(settings.llm_model),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"  [{kind}] Generated summary saved: {cache_path}")
    except OSError as exc:
        # The run can continue because the summary is already in memory.
        print(f"  [{kind}] Could not cache generated summary (continuing): {exc}")
    return summary


def load_or_generate_summary(
    project_root: Path,
    kind: str,
    user_query: str,
    dataset_description: str = "",
) -> Optional[str]:
    """Prefer a prepared/cached txt, otherwise generate it from the question."""

    prepared = load_precomputed_summary(project_root, kind)
    if prepared:
        return prepared
    generated = load_generated_summary(
        project_root,
        kind,
        user_query,
        dataset_description,
    )
    if generated:
        return generated
    return generate_summary_from_question(
        project_root,
        kind,
        user_query,
        dataset_description,
    )
