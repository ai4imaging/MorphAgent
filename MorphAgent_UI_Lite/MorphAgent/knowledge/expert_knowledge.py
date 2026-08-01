"""Expert Knowledge processing module - extract and organize expert knowledge from the expert_knowledge folder"""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import mimetypes

from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image
import tifffile
import numpy as np

from config import settings, make_chat_llm


# Supported image formats
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif', '.webp'}
# Supported text formats
TEXT_EXTENSIONS = {'.txt', '.md', '.markdown', '.csv', '.json'}
# Supported PDF formats
PDF_EXTENSIONS = {'.pdf'}


def get_file_type(file_path: Path) -> str:
    """Determine the file type"""
    suffix = file_path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return 'image'
    elif suffix in TEXT_EXTENSIONS:
        return 'text'
    elif suffix in PDF_EXTENSIONS:
        return 'pdf'
    else:
        return 'unknown'


def load_text_file(file_path: Path) -> str:
    """Load a text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Try other encodings
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                return f.read()
        except:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()


def process_text_with_llm(text: str, file_path: Path) -> str:
    """Process a text file with an LLM to extract key information"""
    llm = make_chat_llm(
        temperature=0,
        max_tokens=settings.llm_max_tokens,
    )
    
    system_prompt = """You are an expert knowledge extraction assistant. Your task is to extract key information from documents provided by experts; this information will be used to guide image feature extraction, segmentation, and feature generation.

Please extract the following information from the given document:

1. **Important image features and morphological indicators**
   - Which features should be focused on and which can be ignored
   - The priority of feature importance

2. **Appearance descriptions of important features** (This is critical!)
   - What these important features should look like in images
   - Their visual features, morphological features, and spatial distribution features
   - How to identify these features (color, shape, size, location, texture, etc.)
   - These descriptions will be used to:
     * Guide segmentation: how to segment out these structures
     * Guide coding: how to detect and quantify these features with code
     * Guide VLM: how to enable visual models to recognize and score these features

3. **Biological background knowledge and related concepts**
   - Relevant biological mechanisms and principles

4. **Feature extraction recommendations and considerations**
   - Specific recommendations for feature extraction
   - Common pitfalls and considerations

Output should be clear, structured, and particularly detailed in describing the appearance of important features, to facilitate subsequent feature generation, segmentation, and VLM scoring. Use English to answer. All output must be in English."""
    
    user_prompt = f"""Please analyze the following expert knowledge document (from file: {file_path.name}) and extract key information:

{text}

Please summarize the key information in the document, with particular attention to:
- Important image features and morphological indicators
- **Appearance descriptions of important features** (detailed description of the visual appearance of these features in images, including shape, size, color, distribution, texture, etc.)
- Priority of feature importance
- Biological background knowledge
- Feature extraction recommendations

**Very Important**: Please describe in detail the appearance of important features. These descriptions will be used to guide segmentation, coding, and VLM feature extraction."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    return response.content if hasattr(response, 'content') else str(response)


def process_image_with_vlm(image_path: Path) -> str:
    """Process an image file with a VLM to extract key information"""
    # TODO: Implement VLM calling
    # For now return a placeholder; the actual implementation depends on the VLM interface
    # Can use Qwen3-VL or other VLM models

    # Temporary implementation: use an LLM to describe the image (should actually use a VLM)
    # Read basic image information
    try:
        if image_path.suffix.lower() in ['.tif', '.tiff']:
            img = tifffile.imread(str(image_path))
            shape = img.shape
            dtype = img.dtype
            info = f"TIFF image, shape: {shape}, data type: {dtype}"
        else:
            img = Image.open(image_path)
            info = f"Image format: {img.format}, size: {img.size}, mode: {img.mode}"
    except Exception as e:
        info = f"Unable to read image information: {e}"
    
    # Use an LLM to generate a description (should actually use a VLM)
    llm = make_chat_llm(
        temperature=0,
        max_tokens=settings.llm_max_tokens,
    )
    
    system_prompt = """You are an expert knowledge extraction assistant. The user will provide information about image files from the expert knowledge folder, and you need to help understand the important information these images may contain.

Based on the image information, infer the possible expert knowledge content and extract key information, particularly describing the appearance of important features. Use English to answer. All output must be in English."""
    
    user_prompt = f"""The expert knowledge folder contains an image file: {image_path.name}
Image information: {info}

Please infer the possible expert knowledge content that this image may contain, with particular attention to:

1. **Important Image Feature Examples**
   - What important features or structures are shown in the image

2. **Appearance Descriptions of Important Features** (This is critical!)
   - The visual appearance of these features in the image
   - Visual features such as shape, size, color, distribution, texture, etc.
   - These descriptions will be used to guide segmentation, coding, and VLM feature extraction

3. **Reference Standards for Feature Extraction**
   - What constitutes good feature extraction results
   - What should be avoided

4. **Morphological Patterns to Note**
   - Important morphological features and patterns

Please summarize the possible key information, with particular emphasis on detailed descriptions of the appearance of important features."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    return response.content if hasattr(response, 'content') else str(response)


def process_pdf_file(pdf_path: Path) -> str:
    """Process a PDF file (uses text extraction for now, can be improved later)"""
    # TODO: Implement PDF text extraction
    # Can use libraries such as PyPDF2, pdfplumber, etc.
    try:
        import PyPDF2
        text = ""
        with open(pdf_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return process_text_with_llm(text, pdf_path)
    except ImportError:
        return f"[PDF file] {pdf_path.name} - the PyPDF2 library must be installed to process PDF files"
    except Exception as e:
        return f"[PDF file] {pdf_path.name} - processing error: {e}"


def synthesize_expert_knowledge(
    text_summaries: List[str],
    image_summaries: List[str],
    pdf_summaries: List[str]
) -> str:
    """Aggregate all expert knowledge information"""
    llm = make_chat_llm(
        temperature=0,
        max_tokens=settings.llm_max_tokens,
    )
    
    system_prompt = """You are an expert knowledge integration assistant. Your task is to integrate expert knowledge information extracted from multiple sources into a unified, structured knowledge summary.

This summary will be used to guide image feature extraction, segmentation, and feature generation, and should include:

1. **Important image features and morphological indicators**
   - Which features should be focused on and which can be ignored
   - The priority of feature importance

2. **Appearance descriptions of important features** (This is critical!)
   - Describe in detail the visual manifestation of these important features in images
   - Including: shape, size, color, spatial distribution, texture, intensity patterns, etc.
   - These descriptions will be used to:
     * **Segmentation**: how to identify and segment out these structures (e.g., "Tau protein appears in axons as slender linear structures, approximately 200-500nm wide, continuously distributed along the axon direction")
     * **Coding**: how to detect and quantify these features with code (e.g., "detect linear structures, compute length, width, continuity, etc.")
     * **VLM**: how to enable visual models to recognize and score these features (e.g., "assess the distribution density and continuity of Tau protein in axons")

3. **Biological background knowledge**
   - Relevant biological mechanisms and principles

4. **Feature extraction recommendations and considerations**
   - Specific recommendations for feature extraction
   - Common pitfalls and considerations

Output should be clear, structured, and particularly detailed in describing the appearance of important features, to facilitate subsequent feature generation, segmentation, and VLM scoring. Use English to answer. All output must be in English."""
    
    user_prompt_parts = []
    
    if text_summaries:
        user_prompt_parts.append("=== Information Extracted from Text Files ===\n")
        for i, summary in enumerate(text_summaries, 1):
            user_prompt_parts.append(f"Document {i}:\n{summary}\n")
    
    if image_summaries:
        user_prompt_parts.append("\n=== Information Extracted from Image Files ===\n")
        for i, summary in enumerate(image_summaries, 1):
            user_prompt_parts.append(f"Image {i}:\n{summary}\n")
    
    if pdf_summaries:
        user_prompt_parts.append("\n=== Information Extracted from PDF Files ===\n")
        for i, summary in enumerate(pdf_summaries, 1):
            user_prompt_parts.append(f"PDF {i}:\n{summary}\n")
    
    user_prompt = "\n".join(user_prompt_parts)
    user_prompt += "\n\nPlease integrate all the above information to generate a unified expert knowledge summary for guiding image feature extraction, segmentation, and feature generation.\n\n**Very Important**: Please describe in detail the appearance of important features, including their visual appearance in images, morphological features, spatial distribution, etc. These descriptions will be used for:\n1. Guiding segmentation: How to identify and segment these structures\n2. Guiding coding: How to detect and quantify these features with code\n3. Guiding VLM: How to enable visual models to recognize and score these features"
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    return response.content if hasattr(response, 'content') else str(response)


def extract_expert_knowledge(
    project_root: Path,
    enable_expert_knowledge: bool = True
) -> Optional[str]:
    """Extract expert knowledge from the expert_knowledge folder under the project root

    Args:
        project_root: Path to the project root directory (parent directory containing dataset and expert_knowledge)
        enable_expert_knowledge: Whether to enable expert knowledge extraction

    Returns:
        Expert knowledge summary text, or None if not enabled or not found
    """
    if not enable_expert_knowledge:
        return None
    
    # Try to find the expert_knowledge folder (supports different naming)
    expert_knowledge_dir = None
    possible_names = ["expert_knowledge", "expert-knowledge", "Expert_Knowledge", "Expert-Knowledge"]
    
    for name in possible_names:
        candidate_dir = project_root / name
        if candidate_dir.exists() and candidate_dir.is_dir():
            expert_knowledge_dir = candidate_dir
            break
    
    if expert_knowledge_dir is None:
        print(f"  [Expert Knowledge] expert_knowledge folder does not exist (tried: {', '.join(possible_names)})")
        print(f"  Project root directory: {project_root}")
        return None
    
    print(f"\n[Expert Knowledge] Starting to process the expert knowledge folder: {expert_knowledge_dir}")
    
    # Collect all files
    text_files = []
    image_files = []
    pdf_files = []
    
    for file_path in expert_knowledge_dir.iterdir():
        if file_path.is_file() and not file_path.name.startswith('.'):
            file_type = get_file_type(file_path)
            if file_type == 'text':
                text_files.append(file_path)
            elif file_type == 'image':
                image_files.append(file_path)
            elif file_type == 'pdf':
                pdf_files.append(file_path)
    
    print(f"  Found {len(text_files)} text files, {len(image_files)} image files, {len(pdf_files)} PDF files")
    
    if not (text_files or image_files or pdf_files):
        print("[Expert Knowledge] No supported files found")
        return None
    
    # Process text files
    text_summaries = []
    if text_files:
        print(f"  Processing {len(text_files)} text files...")
        for text_file in text_files:
            print(f"    Processing: {text_file.name}")
            try:
                text_content = load_text_file(text_file)
                summary = process_text_with_llm(text_content, text_file)
                text_summaries.append(summary)
            except Exception as e:
                print(f"    Error: error while processing {text_file.name}: {e}")
                text_summaries.append(f"[text file] {text_file.name} - processing error: {e}")
    
    # Process image files
    image_summaries = []
    if image_files:
        print(f"  Processing {len(image_files)} image files...")
        for image_file in image_files:
            print(f"    Processing: {image_file.name}")
            try:
                summary = process_image_with_vlm(image_file)
                image_summaries.append(summary)
            except Exception as e:
                print(f"    Error: error while processing {image_file.name}: {e}")
                image_summaries.append(f"[image file] {image_file.name} - processing error: {e}")
    
    # Process PDF files
    pdf_summaries = []
    if pdf_files:
        print(f"  Processing {len(pdf_files)} PDF files...")
        for pdf_file in pdf_files:
            print(f"    Processing: {pdf_file.name}")
            try:
                summary = process_pdf_file(pdf_file)
                pdf_summaries.append(summary)
            except Exception as e:
                print(f"    Error: error while processing {pdf_file.name}: {e}")
                pdf_summaries.append(f"[PDF file] {pdf_file.name} - processing error: {e}")
    
    # Aggregate all information
    print(f"  Aggregating all expert knowledge information...")
    expert_knowledge_summary = synthesize_expert_knowledge(
        text_summaries,
        image_summaries,
        pdf_summaries
    )
    
    print(f"  [OK] Expert knowledge extraction complete")
    
    # Print the expert knowledge summary
    print(f"\n[Expert Knowledge] Expert knowledge summary:")
    print("=" * 80)
    if expert_knowledge_summary:
        # Print the first 500 characters
        preview = expert_knowledge_summary[:500]
        print(preview)
        if len(expert_knowledge_summary) > 500:
            print(f"\n... ({len(expert_knowledge_summary)} characters total, full content saved)")
        else:
            print(expert_knowledge_summary)
    else:
        print("No expert knowledge content")
    print("=" * 80)
    
    return expert_knowledge_summary
