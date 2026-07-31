"""Utility functions module"""
from pathlib import Path
from typing import List, Optional, Dict, Any
from tools.data_path_selector import get_data_path_selector


def select_appropriate_data_source(
    sample_dir: Path,
    feature: Dict[str, Any],
    dataset_description: Optional[str] = None
) -> List[str]:
    """Intelligently select an appropriate data source based on the feature requirements and dataset description

    Uses DataPathSelector for intelligent selection, ensuring code features and VLM features use the correct data source.

    Data organization structure notes:
    - Primary files (raw data): direct files under sample_dir (image files)
    - Secondary files (derived data): files in subdirectories under sample_dir (processed image files)
    - Code features: use only primary files (raw data)
    - VLM features: prefer secondary files (such as the slices directory); if not available, use primary files

    Args:
        sample_dir: Path to the sample directory
        feature: Feature definition (containing name, description, method, needs_segmentation, etc.)
        dataset_description: Dataset description (optional, used to understand the data format)

    Returns:
        List of selected image file paths
    """
    selector = get_data_path_selector()
    method = feature.get("method", "code")
    
    return selector.select_data_paths(
        sample_dir,
        feature,
        dataset_description,
        method=method
    )


def _find_code_data_sources(sample_dir: Path, dataset_description: Optional[str] = None) -> List[str]:
    """Find the data source used by code features (uses only primary files, i.e. raw data)

    Primary files: direct files under sample_dir
    Excludes files in subdirectories (secondary files)

    Args:
        sample_dir: Path to the sample directory
        dataset_description: Dataset description (optional, used for intelligent file selection)

    Returns:
        List of primary file paths
    """
    from config import settings
    image_extensions = settings.image_extensions
    image_paths = []
    
    # Only look for primary files (direct files under sample_dir, excluding subdirectories)
    for item in sample_dir.iterdir():
        if item.is_file() and item.suffix.lower() in image_extensions:
            image_paths.append(str(item))
    
    # If none found and a dataset description is provided, try to extract file patterns from the description
    # But do not hardcode file names here; rely on the information in the dataset description instead
    # If the dataset description mentions specific file names, they can be extracted via the LLM
    
    if image_paths:
        # Sort to ensure consistent ordering
        image_paths = sorted(image_paths)
        print(f"  [Data Selection] Code features use primary files (raw data): {len(image_paths)} files")
        print(f"    Files: {', '.join([Path(p).name for p in image_paths[:3]])}{'...' if len(image_paths) > 3 else ''}")
    
    return image_paths


def _find_vlm_data_sources(sample_dir: Path) -> List[str]:
    """Find the data source used by VLM features (can use secondary files)

    Secondary files: files in subdirectories under sample_dir
    These are data derived from primary files, used for VLM visual understanding

    Args:
        sample_dir: Path to the sample directory

    Returns:
        List of secondary file paths (secondary directory preferred); falls back to primary files if none exist
    """
    from config import settings
    image_extensions = settings.image_extensions
    
    # Priority 1: find all secondary directories (do not hardcode directory names)
    # Select the secondary directory containing the most image files
    secondary_dirs = {}
    for item in sample_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            dir_files = [
                str(p) for p in item.iterdir()
                if p.is_file() and p.suffix.lower() in image_extensions
            ]
            if dir_files:
                secondary_dirs[item.name] = sorted(dir_files)
    
    if secondary_dirs:
        # Select the directory with the most files
        best_dir = max(secondary_dirs.items(), key=lambda x: len(x[1]))
        dir_name, dir_files = best_dir
        print(f"  [Data Selection] VLM features use the {dir_name} directory (secondary files): {len(dir_files)} files")
        return dir_files
    
    # Fallback: if there are no secondary files, use primary files
    print(f"  [Data Selection] [WARN]  No secondary directory found; VLM features fall back to primary files")
    return _find_code_data_sources(sample_dir, None)


def find_image_paths(sample_dir: Path, dataset_description: Optional[str] = None) -> List[str]:
    """Find image files in the sample directory (generic implementation, used for visualization and similar scenarios)

    Prefer returning primary files (raw data); if none exist, look for secondary files

    Args:
        sample_dir: Path to the sample directory
        dataset_description: The LLM-generated dataset description (optional, used to guide file lookup)

    Returns:
        List of image file paths
    """
    # First try to find primary files
    code_sources = _find_code_data_sources(sample_dir, dataset_description)
    if code_sources:
        return code_sources
    
    # If there are no primary files, look for secondary files
    return _find_vlm_data_sources(sample_dir)


def read_dataset_index(data_root: Path) -> List[str]:
    """Read the dataset index and get all sample IDs (generic implementation)

    Directly scans the directory to get the actual sample directory names, without assuming any specific file structure.
    Applicable to any dataset.
    """
    sample_ids = []
    
    if data_root.exists():
        for item in sorted(data_root.iterdir()):
            if item.is_dir() and not item.name.startswith('.'):
                # Do not check for specific files; any non-hidden directory is considered a sample directory
                # This can later be validated using the LLM-understood dataset description
                sample_ids.append(item.name)
    
    return sorted(sample_ids)


def find_description_file(data_root: Path, custom_path: Optional[Path] = None) -> Optional[Path]:
    """Find the dataset description file

    Args:
        data_root: Dataset root directory
        custom_path: Custom description file path (optional)

    Returns:
        Path to the description file, or None if not found
    """
    from config import settings
    
    if custom_path and custom_path.exists():
        return custom_path
    
    # Use the list of description file names from the configuration
    for desc_file in settings.dataset_description_files:
        desc_path = data_root / desc_file
        if desc_path.exists():
            return desc_path
    
    return None
