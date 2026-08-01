"""RAG processing module - extract PDF and XML information from the RAG folder"""
from pathlib import Path
from typing import Optional, List, Dict, Any
import os
import hashlib
import json
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage

from config import settings, make_chat_llm

from .pdf_text import extract_text_from_pdf

# Try to import BeautifulSoup for XML parsing
try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    BeautifulSoup = None


def extract_text_from_xml(xml_path: Path) -> str:
    """Extract text content from an XML file (mainly for the PMC XML format)
    
    Args:
        xml_path: XML file path
        
    Returns:
        The extracted text content
    """
    if not BEAUTIFULSOUP_AVAILABLE:
        return f"[ERROR] BeautifulSoup is not available. Cannot extract text from {xml_path.name}. Please install: pip install beautifulsoup4 lxml"
    
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()
        
        soup = BeautifulSoup(xml_content, 'lxml-xml')
        
        # Extract the title
        title = ''
        title_elem = soup.find('article-title')
        if title_elem:
            title = title_elem.get_text(strip=True)
        
        # Extract the abstract
        abstract = ''
        abstract_elem = soup.find('abstract')
        if abstract_elem:
            abstract = abstract_elem.get_text(strip=True)
        
        # Extract the body (all paragraphs)
        body_text = []
        body_elem = soup.find('body')
        if body_elem:
            # Extract all paragraphs
            paragraphs = body_elem.find_all('p')
            for p in paragraphs:
                text = p.get_text(strip=True)
                if text:
                    body_text.append(text)
        
        # Extract keywords
        keywords = []
        kwd_groups = soup.find_all('kwd-group')
        for kwd_group in kwd_groups:
            kwds = kwd_group.find_all('kwd')
            for kwd in kwds:
                keywords.append(kwd.get_text(strip=True))
        
        # Combine the full text
        full_text_parts = []
        if title:
            full_text_parts.append(f"Title: {title}")
        if abstract:
            full_text_parts.append(f"Abstract: {abstract}")
        if body_text:
            full_text_parts.append("Body:")
            full_text_parts.extend(body_text)
        if keywords:
            full_text_parts.append(f"Keywords: {', '.join(keywords)}")
        
        full_text = '\n\n'.join(full_text_parts)
        
        return full_text if full_text.strip() else f"[WARNING] No text content extracted from {xml_path.name}"
        
    except Exception as e:
        return f"[ERROR] Failed to extract text from {xml_path.name}: {e}"


def _estimate_tokens(text: str) -> int:
    """Roughly estimate the number of tokens in the text (about 4 characters = 1 token)"""
    return len(text) // 4


def _split_documents_into_batches(
    document_texts: List[str],
    document_names: List[str],
    document_types: List[str],
    max_tokens_per_batch: int = 200000  # Conservative estimate, leaving enough headroom
) -> List[tuple]:
    """Split the document list into multiple batches, ensuring each batch's token count does not exceed the limit
    
    Args:
        document_texts: list of document text contents
        document_names: list of document file names
        document_types: list of document types
        max_tokens_per_batch: maximum number of tokens per batch
        
    Returns:
        List of batches, where each batch is a (texts, names, types) tuple
    """
    batches = []
    current_batch_texts = []
    current_batch_names = []
    current_batch_types = []
    current_batch_tokens = 0
    
    for doc_text, doc_name, doc_type in zip(document_texts, document_names, document_types):
        # Truncate an individual document (to avoid a single document being too long)
        if len(doc_text) > 50000:
            doc_text = doc_text[:50000] + "\n\n[Text truncated; showing only the first 50000 characters]"
        
        # Estimate the token count after adding this document
        doc_tokens = _estimate_tokens(doc_text)
        # Add the tokens for the document markers (about 100 tokens)
        total_doc_tokens = doc_tokens + 100
        
        # If adding this document to the current batch would exceed the limit, start a new batch
        if current_batch_tokens + total_doc_tokens > max_tokens_per_batch and current_batch_texts:
            batches.append((current_batch_texts, current_batch_names, current_batch_types))
            current_batch_texts = []
            current_batch_names = []
            current_batch_types = []
            current_batch_tokens = 0
        
        # Add the document to the current batch
        current_batch_texts.append(doc_text)
        current_batch_names.append(doc_name)
        current_batch_types.append(doc_type)
        current_batch_tokens += total_doc_tokens
    
    # Add the last batch
    if current_batch_texts:
        batches.append((current_batch_texts, current_batch_names, current_batch_types))
    
    return batches


def summarize_rag_knowledge(
    document_texts: List[str],
    document_names: List[str],
    document_types: List[str] = None,
    batch_size: int = None  # If None, automatically batch based on the token limit
) -> str:
    """Summarize the RAG knowledge base content using an LLM (supports batched processing)
    
    Args:
        document_texts: list of document text contents (PDF or XML)
        document_names: list of document file names
        document_types: list of document types ('pdf' or 'xml'); if None, inferred automatically
        batch_size: number of documents to process per batch (if None, batched automatically based on the token limit)
        
    Returns:
        The summarized RAG knowledge information
    """
    llm = make_chat_llm(
        temperature=0,
        max_tokens=settings.llm_max_tokens,
    )
    
    # If no document types are provided, infer them from the file names
    if document_types is None:
        document_types = []
        for name in document_names:
            if name.lower().endswith('.pdf'):
                document_types.append('pdf')
            elif name.lower().endswith('.xml'):
                document_types.append('xml')
            else:
                document_types.append('unknown')
    
    # If there are many documents, use batched processing
    total_docs = len(document_texts)
    if batch_size is None:
        # Automatic batching: based on the token limit, about 50-100 documents per batch (depending on document length)
        # Conservative estimate: at most 200k tokens per batch (including prompt and response)
        batches = _split_documents_into_batches(document_texts, document_names, document_types)
    else:
        # Batch by a fixed count
        batches = []
        for i in range(0, total_docs, batch_size):
            batch_texts = document_texts[i:i+batch_size]
            batch_names = document_names[i:i+batch_size]
            batch_types = document_types[i:i+batch_size]
            batches.append((batch_texts, batch_names, batch_types))
    
    # If there is only one batch, process it directly
    if len(batches) == 1:
        return _summarize_single_batch(
            batches[0][0], batches[0][1], batches[0][2], llm, is_final=True
        )
    
    # Multiple batches: process each batch first, then summarize the results of all batches
    print(f"  [RAG] Many documents ({total_docs}); processing in {len(batches)} batches")
    batch_summaries = []
    
    for i, (batch_texts, batch_names, batch_types) in enumerate(batches, 1):
        print(f"  [RAG] Processing batch {i}/{len(batches)} ({len(batch_texts)} documents)...")
        batch_summary = _summarize_single_batch(
            batch_texts, batch_names, batch_types, llm, is_final=False
        )
        batch_summaries.append(batch_summary)
    
    # Summarize the results of all batches
    print(f"  [RAG] Summarizing the results of {len(batches)} batches...")
    final_summary = _summarize_batch_results(batch_summaries, llm)
    
    return final_summary


def _summarize_single_batch(
    document_texts: List[str],
    document_names: List[str],
    document_types: List[str],
    llm: Any,
    is_final: bool = True
) -> str:
    """Process a single batch of documents
    
    Args:
        document_texts: list of document text contents
        document_names: list of document file names
        document_types: list of document types
        llm: LLM instance
        is_final: whether this is the final batch (affects the prompt)
        
    Returns:
        The summary result for this batch
    """
    system_prompt = """You are a scientific-literature analyst supporting an image
feature-engineering agent for microscopy. Your job is to read the provided
documents and extract information that helps DESIGN and INTERPRET quantitative,
scalar image features for the target dataset. Stay grounded in what the
documents actually say.

Extract, when present:
- **Morphological / visual phenotypes**: describe cell/organelle/structure
  appearance in detail — shape, size, texture, intensity, spatial distribution,
  sub-cellular localization — the kind of visual cues a model could measure.
- **Quantitative image features / analysis methods**: named features, feature
  families, or measurement approaches (e.g. morphology, intensity, texture,
  spatial statistics) and what biological property they reflect.
- **Biological meaning & relationships**: which structures/markers/conditions
  produce which visual changes, and why (mechanisms, pathways, processes) — only
  to the extent it explains an observable morphological difference.
- **Segmentation / imaging considerations**: channels, stains/markers, imaging
  modality, and any caveats that affect how features should be computed.

Requirements:
1. Only extract content relevant to morphological/image feature design for this
   kind of data. Skip unrelated general biology.
2. Label each extracted point with an evidence strength:
   - **Strong Evidence**: clear experimental data, repeated/validated, strong source.
   - **Moderate Evidence**: limited experimental support, single study.
   - **Weak Evidence**: speculative or preliminary.
3. For morphological features, describe their visual appearance concretely
   (color, shape, size, distribution, texture, etc.).
4. Keep the output structured and readable.

Use English for all output."""
    
    user_prompt_parts = []
    
    for doc_name, doc_text, doc_type in zip(document_names, document_texts, document_types):
        # The text was already truncated during batching, so no need to truncate again here
        doc_type_label = f"({doc_type.upper()})" if doc_type != 'unknown' else ""
        user_prompt_parts.append(f"=== RAG Knowledge Base Document: {doc_name} {doc_type_label} ===\n{doc_text}\n")
    
    user_prompt = "\n".join(user_prompt_parts)
    
    if is_final:
        user_prompt += "\n\nPlease integrate the information from all the above documents into a unified knowledge summary focused on designing and interpreting quantitative image features for this dataset:\n1. Morphological / visual phenotypes (with concrete visual appearance)\n2. Quantitative image features and analysis methods, and the biological property each reflects\n3. Biological relationships that explain observable morphological differences\n\n**Very Important**: \n- Only keep information relevant to morphological / image feature design\n- Label each point with evidence strength (Strong/Moderate/Weak)\n- Describe morphological features in detail (visual appearance, spatial distribution, etc.)\n- These descriptions will be used to guide image feature extraction, segmentation, and VLM scoring."
    else:
        user_prompt += "\n\nPlease extract key information from the above documents, focused on designing and interpreting quantitative image features for this dataset:\n1. Morphological / visual phenotypes (with concrete visual appearance)\n2. Quantitative image features and analysis methods, and the biological property each reflects\n3. Biological relationships that explain observable morphological differences\n\n**Important**: \n- Only keep information relevant to morphological / image feature design\n- Label each point with evidence strength (Strong/Moderate/Weak)\n- This is a batch summary that will be combined with other batches later, so focus on key findings."
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        raw_summary = response.content if hasattr(response, 'content') else str(response)
        return raw_summary
    except Exception as e:
        print(f"  [RAG] [WARN]  Error while processing batch: {e}")
        return f"[ERROR] Failed to process batch: {e}"


def _summarize_batch_results(
    batch_summaries: List[str],
    llm: Any
) -> str:
    """Summarize the results of multiple batches
    
    Args:
        batch_summaries: list of summary results for each batch
        llm: LLM instance
        
    Returns:
        The final summary result
    """
    system_prompt = """You are a scientific-literature analyst. Merge the per-batch
summaries below into a single, structured knowledge summary that supports image
feature engineering for a microscopy dataset, focused on:
1. Morphological / visual phenotypes (with concrete visual appearance)
2. Quantitative image features and analysis methods, and the biological property each reflects
3. Biological relationships that explain observable morphological differences

Requirements:
- Merge all batches and remove duplicates.
- For the same point, keep the version with the strongest evidence.
- Keep the output structured and readable.
- Label each point with evidence strength (Strong/Moderate/Weak).
- Preserve every key finding.

Use English for all output."""
    
    user_prompt_parts = []
    for i, batch_summary in enumerate(batch_summaries, 1):
        user_prompt_parts.append(f"=== Batch {i} Summary ===\n{batch_summary}\n")
    
    user_prompt = "\n".join(user_prompt_parts)
    user_prompt += "\n\nPlease integrate all the above batch summaries into a unified RAG knowledge summary. Remove duplicates, consolidate similar information, and ensure all key findings are preserved."
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        final_summary = response.content if hasattr(response, 'content') else str(response)
        return final_summary
    except Exception as e:
        print(f"  [RAG] [WARN]  Error while summarizing batches: {e}")
        # If summarization fails, return a simple concatenation of all batches
        return "\n\n".join([f"=== Batch {i+1} ===\n{summary}" for i, summary in enumerate(batch_summaries)])


def refine_rag_knowledge(
    raw_rag_summary: str
) -> str:
    """Perform fine-grained filtering and evidence grading on the RAG knowledge
    
    Remove content unrelated to image feature design, and grade the strength of evidence, ensuring the content is precise and non-redundant.
    
    Args:
        raw_rag_summary: the raw RAG knowledge summary
        
    Returns:
        The filtered and graded RAG knowledge summary
    """
    if not raw_rag_summary or len(raw_rag_summary.strip()) < 100:
        return raw_rag_summary
    
    llm = make_chat_llm(
        temperature=0,
        max_tokens=settings.llm_max_tokens,
    )
    
    system_prompt = """You are a scientific-literature curator. Refine the knowledge
summary below by filtering for relevance and grading the evidence, so it can
directly guide image feature extraction, segmentation, and VLM scoring.

Keep (must be relevant to image feature design for this kind of microscopy data):
- Morphological / visual phenotypes and their concrete visual appearance.
- Quantitative image features / analysis methods and the biological property each reflects.
- Biological relationships that explain observable morphological differences.
- Segmentation / imaging considerations (channels, markers, modality, caveats).

Remove:
- General biology not tied to any observable morphological/image feature.
- Overly broad findings with no link to image analysis.
- Duplicated or redundant information.

Evidence grading:
- **Strong Evidence**: clear experimental data, repeated/validated, strong source.
- **Moderate Evidence**: limited experimental support, single study.
- **Weak Evidence**: speculative or preliminary.

Output format (use these sections; drop any that are empty):

# Knowledge Summary (Refined)

## 1. Morphological / Visual Phenotypes
- [point] [Evidence: Strong/Moderate/Weak]
- [describe visual appearance: color, shape, size, distribution, texture, localization]

## 2. Quantitative Image Features & Methods
- [feature / method — what biological property it reflects] [Evidence: Strong/Moderate/Weak]

## 3. Biological Relationships Explaining Morphology
- [structure/marker/condition -> observable morphological change, with mechanism if given] [Evidence: ...]

## 4. Segmentation / Imaging Considerations
- [channels, markers, modality, caveats affecting feature computation] [Evidence: ...]

Requirements:
- Keep only content relevant to image feature design.
- Label every point with evidence strength.
- For morphological features, always describe the visual appearance concretely.
- Remove redundancy but preserve every key finding.

Use English for all output."""
    
    user_prompt = f"""Please refine and evidence-grade the following knowledge summary:

{raw_rag_summary}

Follow the requirements above: drop irrelevant content, grade the evidence, and use the specified format."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    refined_summary = response.content if hasattr(response, 'content') else str(response)
    
    return refined_summary


def _compute_rag_folder_hash(rag_dir: Path, pdf_files: List[Path], xml_files: List[Path]) -> str:
    """Compute a hash of the RAG folder content, used for cache validation
    
    Args:
        rag_dir: RAG folder path
        pdf_files: list of PDF files
        xml_files: list of XML files
        
    Returns:
        The hash string
    """
    hash_data = []
    
    # Collect information about all files (file name, size, modification time)
    all_files = sorted(pdf_files + xml_files)
    for file_path in all_files:
        try:
            stat = file_path.stat()
            file_info = f"{file_path.name}:{stat.st_size}:{stat.st_mtime}"
            hash_data.append(file_info)
        except Exception as e:
            # If the file cannot be accessed, record the error information as well
            hash_data.append(f"{file_path.name}:error:{str(e)}")
    
    # Generate the hash
    hash_string = "\n".join(hash_data)
    return hashlib.md5(hash_string.encode('utf-8')).hexdigest()


def _load_rag_cache(cache_file: Path) -> Optional[tuple]:
    """Load the RAG cache file
    
    Args:
        cache_file: cache file path
        
    Returns:
        (rag_content, rag_hash) tuple, or None if the cache does not exist or is invalid
    """
    if not cache_file.exists():
        return None
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Look for the metadata end marker
        metadata_end_idx = -1
        for i, line in enumerate(lines):
            if line.strip() == "# End Metadata":
                metadata_end_idx = i
                break
        
        if metadata_end_idx == -1:
            # Old format without metadata; read the entire content directly
            content = ''.join(lines).strip()
            if content and not content.startswith('[ERROR]'):
                return (content, None)
            return None
        
        # Extract the hash from the metadata
        rag_hash = None
        for i in range(metadata_end_idx + 1):
            if 'hash' in lines[i]:
                import re
                match = re.search(r'"hash":\s*"([^"]+)"', lines[i])
                if match:
                    rag_hash = match.group(1)
                    break
        
        # Extract the content part (after the metadata)
        content_lines = lines[metadata_end_idx + 1:]
        content = ''.join(content_lines).strip()
        
        if content and not content.startswith('[ERROR]'):
            return (content, rag_hash)
    except Exception as e:
        print(f"  [RAG] Failed to read the cache file: {e}")
    
    return None


def _save_rag_cache(cache_file: Path, rag_content: str, rag_hash: str):
    """Save the RAG result to the cache file
    
    Args:
        cache_file: cache file path
        rag_content: RAG knowledge content
        rag_hash: hash of the RAG folder
    """
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the content, and add the hash and metadata at the beginning of the file
        metadata = {
            'hash': rag_hash,
            'created_at': datetime.now().isoformat(),
            'content_length': len(rag_content)
        }
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            # Write the metadata (JSON format, for easy later validation)
            f.write(f"# RAG Cache Metadata\n")
            f.write(f"# {json.dumps(metadata, ensure_ascii=False)}\n")
            f.write(f"# End Metadata\n\n")
            f.write(rag_content)
        
        print(f"  [RAG] Cache saved: {cache_file}")
    except Exception as e:
        print(f"  [RAG] Failed to save cache: {e}")


def extract_rag_knowledge(
    project_root: Path,
    enable_rag: bool = True,
    device: str = "gpu:0",
    use_cache: bool = True,
    cache_dir: Optional[Path] = None
) -> Optional[str]:
    """Extract RAG knowledge base information from the RAG folder under the project root (supports caching)
    
    Args:
        project_root: project root directory path (the parent directory containing dataset and RAG)
        enable_rag: whether to enable RAG knowledge extraction
        device: unused by the default lite PDF extractor; kept for CLI compatibility
            (PaddleX device when ``RAG_PDF_BACKEND=paddlex``)
        use_cache: whether to use caching (default True)
        cache_dir: cache directory path; if None, use project_root / ".rag_cache"
        
    Returns:
        The RAG knowledge summary text, or None if not enabled or not found
    """
    if not enable_rag:
        return None

    # Lite: only inject prepared txt; never PDF/XML parse or PubMed fetch.
    from knowledge.precomputed_lite import load_precomputed_summary

    precomputed = load_precomputed_summary(project_root, "rag")
    if precomputed:
        return precomputed
    print("  [RAG] No precomputed summary; skipping (Lite)")
    return None
