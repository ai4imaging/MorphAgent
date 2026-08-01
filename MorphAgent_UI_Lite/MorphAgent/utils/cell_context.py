"""Cell context extraction tool - identify single-cell/multi-cell scenarios from the user query"""
import re
from typing import Dict, Optional, Literal


def extract_cell_context(user_query: str) -> Dict[str, any]:
    """Extract cell context information from the user query

    Args:
        user_query: The user query string

    Returns:
        A dict containing cell context information:
        {
            "cell_type": "single" | "multiple" | "unknown",
            "context_description": str,  # Descriptive text
            "detection_keywords": list,   # Detected keywords
        }
    """
    query_lower = user_query.lower()
    
    # Single-cell keyword patterns
    single_cell_patterns = [
        r'\bsingle\s+cell\b',
        r'\bone\s+cell\b',
        r'\bindividual\s+cell\b',
        r'\bsingle-cell\b',
        r'\bsinglecell\b',
        r'\bper\s+cell\b',
        r'\bcell-level\b',
        r'\bcell\s+level\b',
    ]
    
    # Multi-cell keyword patterns
    multiple_cell_patterns = [
        r'\bmultiple\s+cells\b',
        r'\bmany\s+cells\b',
        r'\bseveral\s+cells\b',
        r'\bmultiple-cell\b',
        r'\bmultiplecell\b',
        r'\bcells\s+in\s+the\s+image\b',
        r'\bcell\s+population\b',
        r'\bpopulation\s+of\s+cells\b',
        r'\bcell\s+culture\b',
        r'\bcell\s+field\b',
        r'\bfield\s+of\s+cells\b',
    ]
    
    detected_keywords = []
    cell_type: Literal["single", "multiple", "unknown"] = "unknown"
    
    # Check single-cell patterns
    for pattern in single_cell_patterns:
        matches = re.findall(pattern, query_lower)
        if matches:
            detected_keywords.extend(matches)
            cell_type = "single"
            break
    
    # Check multi-cell patterns (if single-cell has not been detected yet)
    if cell_type == "unknown":
        for pattern in multiple_cell_patterns:
            matches = re.findall(pattern, query_lower)
            if matches:
                detected_keywords.extend(matches)
                cell_type = "multiple"
                break
    
    # Generate descriptive text
    if cell_type == "single":
        context_description = (
            "**CRITICAL CONTEXT**: This dataset contains images with **SINGLE CELL** per image. "
            "Each image represents one individual cell. This is important because:\n"
            "- Features should focus on single-cell properties (cell morphology, intracellular structures, cell-level measurements)\n"
            "- Segmentation (if needed) should identify the single cell in the image\n"
            "- Aggregation across multiple cells is NOT applicable (there's only one cell per image)\n"
            "- Features should capture cell-level characteristics rather than population-level statistics"
        )
    elif cell_type == "multiple":
        context_description = (
            "**CRITICAL CONTEXT**: This dataset contains images with **MULTIPLE CELLS** per image. "
            "Each image represents a field of view containing multiple cells. This is important because:\n"
            "- Features can focus on both individual cell properties and population-level statistics\n"
            "- Segmentation (if needed) should identify and separate individual cells\n"
            "- Aggregation across multiple cells (mean, std, distribution) is applicable\n"
            "- Features can capture both cell-level characteristics and inter-cell relationships\n"
            "- Population-level features (cell density, spatial distribution, heterogeneity) are relevant"
        )
    else:
        context_description = (
            "**NOTE**: The cell context (single vs multiple cells per image) was not explicitly specified in the user query. "
            "You should design features that are flexible and can work with both scenarios, or make reasonable assumptions "
            "based on the dataset description and typical practices for this type of imaging data."
        )
    
    return {
        "cell_type": cell_type,
        "context_description": context_description,
        "detection_keywords": detected_keywords,
    }

