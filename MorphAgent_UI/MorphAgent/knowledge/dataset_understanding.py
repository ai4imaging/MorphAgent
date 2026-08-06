"""Dataset understanding module - generic implementation using an LLM to understand description files in any format"""
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage

from config import settings, make_chat_llm


def load_dataset_description(description_path: Path) -> str:
    """Load the dataset description file (supports any format)

    Supported formats: txt, json, md
    No parsing is performed; the raw content is returned directly for the LLM to understand.
    """
    if not description_path.exists():
        raise FileNotFoundError(f"Dataset description file does not exist: {description_path}")
    
    suffix = description_path.suffix.lower()
    
    if suffix == ".json":
        with open(description_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                # Try to extract text content
                for key in ["description", "content", "text", "summary"]:
                    if key in data:
                        return str(data[key])
                return json.dumps(data, indent=2, ensure_ascii=False)
            return str(data)
    else:
        # txt, md, or other text files; read directly
        with open(description_path, 'r', encoding='utf-8') as f:
            return f.read()


def analyze_dataset_structure(data_root: Path) -> Dict[str, Any]:
    """Analyze the dataset directory structure (generic implementation)

    Does not assume a specific format; only analyzes the directory structure.
    """
    info = {
        "data_root": str(data_root),
        "samples": [],
        "total_samples": 0,
        "sample_structure": {},
    }
    
    if not data_root.exists():
        return info
    
    # Find all sample directories
    sample_dirs = [d for d in data_root.iterdir() if d.is_dir() and not d.name.startswith('.')]
    sample_dirs = sorted(sample_dirs, key=lambda x: x.name)
    
    info["total_samples"] = len(sample_dirs)
    
    # Analyze the structure of the first few samples (as examples)
    for sample_dir in sample_dirs[:5]:
        sample_info = {
            "name": sample_dir.name,
            "files": [],
        }
        
        # List the files in the directory (without assuming specific file names)
        for item in sample_dir.iterdir():
            if item.is_file():
                sample_info["files"].append(item.name)
            elif item.is_dir():
                sample_info["files"].append(f"{item.name}/")
        
        info["samples"].append(sample_info)
    
    # Count common file types
    if info["samples"]:
        all_files = []
        for sample in info["samples"]:
            all_files.extend(sample["files"])
        
        # Count file extensions
        extensions = {}
        for f in all_files:
            if '.' in f:
                ext = f.split('.')[-1].lower()
                extensions[ext] = extensions.get(ext, 0) + 1
        
        info["file_types"] = extensions
    
    return info


def understand_dataset(
    data_root: Path,
    description_path: Optional[Path] = None,
    user_query: str = ""
) -> List[BaseMessage]:
    """Use an LLM to understand the dataset and generate a standardized description

    Fully generic implementation that assumes no specific format.
    The LLM is responsible for understanding description files in any format.
    """
    # 1. Load the dataset description (if provided)
    description_text = ""
    if description_path:
        try:
            description_text = load_dataset_description(description_path)
        except Exception as e:
            print(f"[WARN]  Warning: Unable to load description file: {e}")
    
    # 2. Analyze the dataset structure
    structure_info = analyze_dataset_structure(data_root)
    
    # 3. Build the LLM prompt (generic, does not assume a specific format)
    system_prompt = """You are a data science expert skilled at understanding and describing dataset structures.

Your task is to generate a standardized, detailed dataset description based on the provided dataset description file and directory structure analysis.

This description will be used to:
1. Help the AI Agent understand the dataset format
2. Guide code generation (feature extraction code)
3. Guide VLM analysis (image understanding)

Please generate a clear, detailed, and structured dataset description, including:

**1. Data dimension information (CRITICAL - must be stated explicitly)**
- Clearly state whether the data is a 2D, 3D, or multi-channel 2D dataset collection
- If it is 2D data: state that this is a single 2D image, and clearly state what the image captures (e.g., cell morphology, tissue sections, fluorescent labeling, etc.)
- If it is 3D data: state that this is a 3D z-stack or 3D volumetric data, and clearly state what the 3D data captures (e.g., the 3D structure of cells, the 3D morphology of tissue, etc.)
- If it is a multi-channel 2D data collection: state that this is a collection of multiple 2D channels, and clearly state what each channel captures (e.g., channel 1 captures the nucleus, channel 2 captures the cytoplasm, etc.)

**2. Channel information (if it is multi-channel data, this must be described in detail)**
- If it is multi-channel 2D data: you must clearly list the number, name, and specific content of each channel (e.g., channel 1 (w1, Hoechst) labels the nucleus; channel 2 (w2, ConA) labels the endoplasmic reticulum, etc.)
- The biological meaning of each channel and the subcellular structure it labels

**3. Data format and file organization**
- Image format: TIFF, PNG, JPG, etc.
- File organization structure (primary files: raw data; secondary files: derived data such as the slices directory)

**4. Data meaning and purpose**
- Data source and experimental purpose
- Key metadata information
- Usage recommendations and considerations

Especially important: please clearly explain the data organization structure:
- Primary files (raw data): Direct files in the sample directory, such as zstack.tif, MIP.tif
- Secondary files (derived data): Files in subdirectories under the sample directory, such as slices/*.png, segmentations/*.png

Use English to answer. All output must be in English."""
    
    user_prompt_parts = []
    
    if user_query:
        user_prompt_parts.append(f"User query: {user_query}\n")
    
    # Add the dataset description (if any)
    if description_text:
        user_prompt_parts.append(f"Dataset description file content:\n{description_text}\n")
    
    # Add the structure analysis results
    structure_summary = f"""Dataset directory structure analysis:

Data root directory: {structure_info['data_root']}
Total number of samples: {structure_info['total_samples']}

Example sample structures (first {len(structure_info['samples'])}):
"""
    for sample in structure_info['samples']:
        structure_summary += f"  - {sample['name']}: {', '.join(sample['files'][:10])}\n"
        if len(sample['files']) > 10:
            structure_summary += f"    ... and {len(sample['files']) - 10} more files\n"
    
    if structure_info.get('file_types'):
        structure_summary += f"\nCommon file types: {', '.join(structure_info['file_types'].keys())}\n"
    
    user_prompt_parts.append(structure_summary)
    
    user_prompt = "\n".join(user_prompt_parts)
    
    # 4. Call the LLM to generate the description
    print(f"\n[Dataset Understanding] Calling LLM to understand the dataset...")
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    llm = make_chat_llm(
        temperature=0,
        max_tokens=settings.llm_max_tokens,
    )
    
    # Print the first 5 and last 5 lines of the input
    print(f"[Dataset Understanding] Input content (first 5 lines):")
    input_lines = user_prompt.split('\n')
    for i, line in enumerate(input_lines[:5], 1):
        print(f"  {i}: {line[:100]}..." if len(line) > 100 else f"  {i}: {line}")
    
    if len(input_lines) > 10:
        print(f"  ... (omitting {len(input_lines) - 10} lines) ...")
        print(f"[Dataset Understanding] Input content (last 5 lines):")
        for i, line in enumerate(input_lines[-5:], len(input_lines) - 4):
            print(f"  {i}: {line[:100]}..." if len(line) > 100 else f"  {i}: {line}")
    else:
        print(f"[Dataset Understanding] Input content (last 5 lines):")
        for i, line in enumerate(input_lines[-5:], max(1, len(input_lines) - 4)):
            print(f"  {i}: {line[:100]}..." if len(line) > 100 else f"  {i}: {line}")
    
    response = llm.invoke(messages)
    response_content = response.content if hasattr(response, 'content') else str(response)
    
    # Print the first 5 and last 5 lines of the output
    print(f"\n[Dataset Understanding] Output content (first 5 lines):")
    output_lines = response_content.split('\n')
    for i, line in enumerate(output_lines[:5], 1):
        print(f"  {i}: {line[:100]}..." if len(line) > 100 else f"  {i}: {line}")
    
    if len(output_lines) > 10:
        print(f"  ... (omitting {len(output_lines) - 10} lines) ...")
        print(f"[Dataset Understanding] Output content (last 5 lines):")
        for i, line in enumerate(output_lines[-5:], len(output_lines) - 4):
            print(f"  {i}: {line[:100]}..." if len(line) > 100 else f"  {i}: {line}")
    else:
        print(f"[Dataset Understanding] Output content (last 5 lines):")
        for i, line in enumerate(output_lines[-5:], max(1, len(output_lines) - 4)):
            print(f"  {i}: {line[:100]}..." if len(line) > 100 else f"  {i}: {line}")
    
    messages.append(response)
    
    return messages


def get_dataset_description_text(messages: List[BaseMessage]) -> str:
    """Extract the dataset description text from the message list"""
    for msg in reversed(messages):
        if hasattr(msg, 'content') and msg.content:
            return str(msg.content)
    return "Dataset description not generated"
