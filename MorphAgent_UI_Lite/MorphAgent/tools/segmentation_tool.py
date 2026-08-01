"""Segmentation Tool for LangChain

Wrap the segmentation functionality as a LangChain tool so it can be used inside an agent.
Respects SEGMENTATION_BACKEND (allen by default for UI / demo).
"""
from typing import Optional, List, Dict, Any
from pathlib import Path
from langchain_core.tools import tool
from tools.segmentation import (
    ensure_sample_segmentation,
    check_segmentation_exists,
)


@tool
def segment_image_tool(
    sample_dir: str,
    image_path: str,
    channels: Optional[List[int]] = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    conda_env: Optional[str] = None
) -> Dict[str, Any]:
    """Segment an image with the configured backend (Allen or Cellpose-SAM).

    This tool checks if segmentation already exists for the sample. If it exists,
    returns the existing segmentation path. Otherwise, performs segmentation
    and saves the result.

    Args:
        sample_dir: Path to the sample directory (e.g., data_root/sample_id)
        image_path: Path to the input image file
        channels: Optional list of channel indices to use for segmentation (e.g., [0, 1])
        flow_threshold: Flow threshold for Cellpose (default: 0.4)
        cellprob_threshold: Cell probability threshold for Cellpose (default: 0.0)
        conda_env: Conda environment name override (Allen default: morphagent_allen)

    Returns:
        Dictionary with:
        - success: bool, whether segmentation was successful
        - mask_path: str, path to the segmentation mask file (if successful)
        - message: str, status message
    """
    sample_dir_path = Path(sample_dir)
    image_path_obj = Path(image_path)

    existing_seg = check_segmentation_exists(sample_dir_path)
    if existing_seg:
        return {
            "success": True,
            "mask_path": str(existing_seg),
            "message": f"Segmentation already exists: {existing_seg}",
            "from_cache": True
        }

    mask_path = ensure_sample_segmentation(
        sample_dir=sample_dir_path,
        image_path=str(image_path_obj),
        channels=channels,
        conda_env=conda_env,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )

    if mask_path is not None:
        return {
            "success": True,
            "mask_path": str(mask_path),
            "message": f"Segmentation completed: {mask_path}",
            "from_cache": False
        }
    return {
        "success": False,
        "mask_path": None,
        "message": "Segmentation failed",
        "from_cache": False
    }


def get_segmentation_tool() -> Any:
    """Get the segmentation tool for use in LangChain agents"""
    return segment_image_tool
