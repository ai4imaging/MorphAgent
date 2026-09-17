"""Automatic Deep Research report generation (single API call).

This build does NOT deploy a local deep-research agent. Instead it makes a
single call to an OpenAI-compatible endpoint (configurable via the
``DEEP_RESEARCH_*`` settings) to produce a literature-grounded research report,
which is saved as markdown into the ``deep_research/`` folder. The existing
``extract_deep_research`` step then reads that markdown directly (no PaddleX
needed for the generated report).

For best results point ``DEEP_RESEARCH_MODEL`` at a web-search / research-capable
model (e.g. Perplexity ``sonar`` / ``sonar-deep-research``, OpenAI
``gpt-4o-search-preview``). Any strong chat model also works, producing a report
grounded in the model's parametric knowledge.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from config import settings, make_chat_llm


_SYSTEM_PROMPT = """You are a senior research scientist writing a focused
literature review to guide QUANTITATIVE IMAGE FEATURE ENGINEERING for a
microscopy dataset. Your report will be read by an automated agent that designs
code-based and vision-model-based image features, so it must be concrete and
actionable.

Write a structured markdown report with these sections:

1. **Background** — the biological system and why its morphology matters.
2. **Key morphological / visual phenotypes** — for each, describe in DETAIL its
   visual appearance in images (shape, size, texture, intensity pattern, spatial
   distribution, sub-cellular localization). These descriptions drive
   segmentation, coding, and vision-model scoring.
3. **Quantitative image features & analysis methods** — concrete measurable
   features (e.g. area fraction, count, elongation, straightness, texture/Haralick,
   intensity heterogeneity, nuclear vs cytoplasmic ratio) and the biological
   property each reflects.
4. **Relationships & mechanisms** — how biology explains observable image
   differences between conditions.
5. **Recommendations & pitfalls** — practical advice for feature extraction.

Rules:
- Be specific and quantitative wherever possible.
- Label the evidence strength of each major claim as (Strong / Moderate / Weak).
- If you cite literature, include concise inline references.
- Output must be in English."""


def build_user_prompt(query: str, dataset_description: Optional[str]) -> str:
    parts = [f"Research task / analysis goal:\n{query}\n"]
    if dataset_description and dataset_description.strip():
        parts.append(
            "Dataset description (imaging modality, channels, structures present):\n"
            f"{dataset_description.strip()[:4000]}\n"
        )
    parts.append(
        "Write the deep-research report now, focusing on morphological / image "
        "features that can be quantified from these images."
    )
    return "\n".join(parts)


def generate_deep_research_report(
    query: str,
    out_dir: Path,
    dataset_description: Optional[str] = None,
    filename: Optional[str] = None,
) -> Optional[Path]:
    """Generate a deep-research report with one API call and save it as markdown.

    Args:
        query: research task / keywords (typically the user query).
        out_dir: the ``deep_research`` folder to write the report into.
        dataset_description: optional dataset context to ground the report.
        filename: optional output filename (defaults to a timestamped name).

    Returns:
        Path to the written markdown report, or None on failure.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use the deep-research model/endpoint (falls back to the main LLM).
    llm = make_chat_llm(
        model=settings.deep_research_model,
        base_url=settings.deep_research_base_url,
        api_key=settings.deep_research_api_key,
        temperature=0.2,
        max_tokens=settings.deep_research_max_tokens,
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=build_user_prompt(query, dataset_description)),
    ]

    print(f"\n[Deep Research] Generating report with model '{settings.deep_research_model}' (single call)...")
    try:
        response = llm.invoke(messages)
        report = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        print(f"  [Deep Research] Report generation failed: {e}")
        return None

    if not report or not report.strip():
        print("  [Deep Research] Model returned an empty report.")
        return None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = filename or f"deep_research_report_{ts}.md"
    out_path = out_dir / fname
    header = (
        f"# Deep Research Report\n\n"
        f"- Query: {query}\n"
        f"- Model: {settings.deep_research_model}\n"
        f"- Generated: {ts}\n\n---\n\n"
    )
    out_path.write_text(header + report, encoding="utf-8")
    print(f"  [Deep Research] Report saved: {out_path} ({len(report)} chars)")
    return out_path


if __name__ == "__main__":  # simple manual smoke test
    import argparse
    ap = argparse.ArgumentParser(description="Generate a deep-research markdown report with one API call.")
    ap.add_argument("query")
    ap.add_argument("--out", default="./deep_research")
    args = ap.parse_args()
    generate_deep_research_report(args.query, Path(args.out))
