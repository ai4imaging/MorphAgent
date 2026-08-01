"""Deep Research processing module - extract PDF information from the deep_research folder"""
from pathlib import Path
from typing import Optional, List
import os

from langchain_core.messages import HumanMessage, SystemMessage

from config import settings, make_chat_llm

from .pdf_text import extract_text_from_pdf


def summarize_deep_research_content(
    pdf_texts: List[str],
    pdf_names: List[str]
) -> str:
    """Summarize Deep Research content using an LLM

    Args:
        pdf_texts: List of PDF text contents
        pdf_names: List of PDF file names

    Returns:
        Summarized Deep Research information
    """
    llm = make_chat_llm(
        temperature=0,
        max_tokens=settings.llm_max_tokens,
    )
    
    system_prompt = """You are a scientific literature analysis expert. Your task is to extract key information from the Deep Research PDF documents; this information will be used to guide image feature extraction.

Please extract the following from the given PDF documents:

1. **Important research findings and conclusions**
   - Key biological findings
   - Scientific conclusions related to image features

2. **Important image features and morphological indicators**
   - Which features have been proven important in the research
   - The relationship between features and biological processes

3. **Appearance descriptions of important features** (This is critical!)
   - What these important features should look like in images
   - Their visual features, morphological features, and spatial distribution features
   - How to identify these features (color, shape, size, location, texture, etc.)
   - These descriptions will be used to:
     * Guide segmentation: how to segment out these structures
     * Guide coding: how to detect and quantify these features with code
     * Guide VLM: how to enable visual models to recognize and score these features

4. **Research methods and technical details**
   - Relevant experimental methods
   - Technical details of image analysis

5. **Feature extraction recommendations and considerations**
   - Research-based feature extraction recommendations
   - Pitfalls and considerations to be aware of

Output should be clear, structured, and particularly detailed in describing the appearance of important features, to facilitate subsequent feature generation, segmentation, and VLM scoring. Use English to answer. All output must be in English."""
    
    user_prompt_parts = []
    
    for pdf_name, pdf_text in zip(pdf_names, pdf_texts):
        # Truncate overly long text (to avoid exceeding token limits)
        if len(pdf_text) > 50000:
            pdf_text = pdf_text[:50000] + "\n\n[Text truncated, showing first 50000 characters only]"
        
        user_prompt_parts.append(f"=== PDF Document: {pdf_name} ===\n{pdf_text}\n")
    
    user_prompt = "\n".join(user_prompt_parts)
    user_prompt += "\n\nPlease integrate the information from all the above PDF documents to generate a unified Deep Research summary for guiding image feature extraction, segmentation, and feature generation.\n\n**Very Important**: Please describe in detail the appearance of important features, including their visual appearance in images, morphological features, spatial distribution, etc. These descriptions will be used to guide segmentation, coding, and VLM feature extraction."
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    return response.content if hasattr(response, 'content') else str(response)


def extract_deep_research(
    project_root: Path,
    enable_deep_research: bool = True,
    device: str = "gpu:0"
) -> Optional[str]:
    """Extract Deep Research information from the deep_research folder under the project root

    Args:
        project_root: Path to the project root directory (parent directory containing dataset and deep_research)
        enable_deep_research: Whether to enable Deep Research extraction
        device: unused by lite PDF extraction; kept for CLI compatibility

    Returns:
        Deep Research summary text, or None if not enabled or not found
    """
    if not enable_deep_research:
        return None

    # Lite: only inject prepared txt; never PDF OCR / LLM re-summarization.
    from knowledge.precomputed_lite import load_precomputed_summary

    precomputed = load_precomputed_summary(project_root, "deep_research")
    if precomputed:
        return precomputed
    print("  [Deep Research] No precomputed summary; skipping (Lite)")
    return None
