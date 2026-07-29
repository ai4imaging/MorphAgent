"""Data statistics collection module - collect image and segmentation statistics for the prompt

Supports parsing segmentation semantics from the dataset description or segmentation/README.txt;
supports recognizing and describing instance label maps (multi-object labels).
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np


def _parse_segmentation_semantics(
    dataset_description: Optional[str] = None,
    sample_dir: Optional[Path] = None
) -> Dict[str, str]:
    """Parse segmentation file semantics from the dataset description or sample_dir/segmentation/README.txt.

    Convention: in the description, lines in the "Segmentation:" or "## Segmentation" section follow the format
    "filename: description" or "stem description"; each line of README.txt follows the same format.

    Args:
        dataset_description: Full text of the dataset description
        sample_dir: Sample directory (used for segmentation/README.txt)

    Returns:
        Keys are filename or stem, values are semantic descriptions
    """
    out: Dict[str, str] = {}
    # Parse from the description
    if dataset_description:
        desc_lower = dataset_description.lower()
        # Find the Segmentation section
        for marker in ["segmentation:", "## segmentation", "\nsegmentation:\n"]:
            idx = desc_lower.find(marker)
            if idx >= 0:
                segment = dataset_description[idx:idx + 2000]
                for line in segment.split("\n"):
                    line = line.strip()
                    if not line or line.lower().startswith("#"):
                        continue
                    # "filename.tif: description" or "stem: description"
                    if ":" in line:
                        k, _, v = line.partition(":")
                        k, v = k.strip(), v.strip()
                        if k and v:
                            out[k] = v
                            # Also use the stem as a key
                            if "." in k:
                                out[Path(k).stem] = v
                break
    # Parse from segmentation/README.txt
    if sample_dir:
        readme = sample_dir / "segmentation" / "README.txt"
        if readme.exists():
            try:
                text = readme.read_text(encoding="utf-8", errors="ignore")
                for line in text.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    if ":" in line:
                        k, _, v = line.partition(":")
                        k, v = k.strip(), v.strip()
                        if k and v:
                            out[k] = v
                            if "." in k:
                                out[Path(k).stem] = v
            except Exception:
                pass
    return out


def collect_data_statistics(
    image_path: Path,
    segmentation_paths: Optional[List[Path]] = None,
    dataset_description: Optional[str] = None,
    sample_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Collect image and segmentation statistics

    Args:
        image_path: Path to the image file
        segmentation_paths: List of segmentation file paths
        dataset_description: Dataset description (used to extract channel information and an optional Segmentation section)
        sample_dir: Sample directory (optional, used for parsing semantics from segmentation/README.txt)

    Returns:
        A dict containing statistics; seg_files_info is extended with unique_labels, used for instance-label descriptions
    """
    stats = {
        "image_shape": "Unknown",
        "image_dtype": "Unknown",
        "image_min": "Unknown",
        "image_max": "Unknown",
        "channel_information": "",
        "segmentation_files_info": "",
        "segmentation_statistics": "",
        "seg_keys_description": "No segmentation masks in this dataset. Use def extract(img): only.",
    }
    
    # Load the image and collect statistics
    try:
        from tools.image_io import load_image_array

        img = load_image_array(image_path)
        if img is not None:
            img_array = np.asarray(img)
            stats["image_shape"] = str(img_array.shape)
            stats["image_dtype"] = str(img_array.dtype)
            stats["image_min"] = float(np.min(img_array))
            stats["image_max"] = float(np.max(img_array))
            
            # Extract channel information
            stats["channel_information"] = _extract_channel_info(img_array, dataset_description)
    except Exception as e:
        print(f"  ⚠️  Failed to collect image statistics: {e}")
    
    # Collect segmentation statistics (including unique_labels for instance-label determination)
    if segmentation_paths:
        seg_stats_list = []
        seg_files_info: List[Dict[str, Any]] = []
        semantics = _parse_segmentation_semantics(dataset_description, sample_dir)

        for i, seg_path in enumerate(segmentation_paths, 1):
            seg_path_obj = Path(seg_path) if not isinstance(seg_path, Path) else seg_path
            if not seg_path_obj.exists():
                continue

            try:
                from tools.image_io import load_image_array

                seg_mask = load_image_array(seg_path_obj)
                seg_array = np.asarray(seg_mask)
                n_unique = len(np.unique(seg_array))

                seg_files_info.append({
                    "index": i,
                    "name": seg_path_obj.name,
                    "stem": seg_path_obj.stem,
                    "unique_labels": n_unique,
                })

                seg_stats_list.append(
                    f"  {i}. `{seg_path_obj.name}` (stem: `{seg_path_obj.stem}`):\n"
                    f"     - Shape: {seg_array.shape}\n"
                    f"     - Dtype: {seg_array.dtype}\n"
                    f"     - Value range: min={int(np.min(seg_array))}, max={int(np.max(seg_array))}\n"
                    f"     - Unique labels: {n_unique} (background + {n_unique - 1} objects)\n"
                )
            except Exception as e:
                print(f"  ⚠️  Failed to collect segmentation statistics ({seg_path_obj.name}): {e}")

        if seg_files_info:
            stats["segmentation_files_info"] = "\n**Available Segmentation Files** (in `sample_dir/segmentation/`):\n" + \
                "\n".join([f"  {f['index']}. `{f['name']}` (stem: `{f['stem']}`)" for f in seg_files_info]) + \
                f"\n\nTotal: {len(seg_files_info)} segmentation file(s) available.\n"
            # Keys for seg dict: use filename stem; coding agent must use these keys, not position
            keys_list = ", ".join(repr(f["stem"]) for f in seg_files_info)
            stats["seg_keys_description"] = (
                f"**Segmentation keys for this dataset** (use these exact keys in `seg`): {keys_list}.\n"
                "Access by key only, e.g. seg[\"mask_cell\"], seg.get(\"mask_nucleus\"). Do NOT guess by position or index."
            )
        else:
            stats["seg_keys_description"] = "No segmentation masks in this dataset. Use def extract(img): only."

        if seg_stats_list:
            stats["segmentation_statistics"] = "\n".join(seg_stats_list)

        # Generate the mask order description (used for the prompt; includes semantic override and instance-label notes)
        if seg_files_info:
            mask_order_description = _generate_mask_order_description(
                seg_files_info,
                semantics_override=semantics if semantics else None,
            )
            stats["segmentation_mask_order"] = mask_order_description
        else:
            stats["segmentation_mask_order"] = ""

    return stats


def _generate_mask_order_description(
    seg_files_info: List[Dict[str, Any]],
    semantics_override: Optional[Dict[str, str]] = None,
) -> str:
    """Generate the mask order description string.

    Args:
        seg_files_info: List of mask file information, each containing index, name, stem, and optionally unique_labels
        semantics_override: Optional; keys are filename or stem, values are semantic descriptions (from the description or README)

    Returns:
        A formatted mask order description string; if a mask is an instance label map (unique_labels > 2), per-object measurement and aggregation notes are appended
    """
    if not seg_files_info:
        return ""

    semantics_override = semantics_override or {}
    # Known mask type mapping (including common user scenarios)
    mask_type_map = {
        "cyto": "cell body (whole cell)",
        "nuclei": "nucleus",
        "nucleus": "nucleus",
        "cytoplasm": "cytoplasm",
        "cell": "cell body (whole cell)",
        "cell_body": "cell body (whole cell)",
        "mitochondria": "mitochondria (instance labels)",
        "labels": "instance label map (integer per object)",
        "instance": "instance label map (integer per object)",
    }

    description_parts = [
        "\n**CRITICAL: Segmentation is passed as a dict `seg` (key = filename stem)**",
        "Access masks by key only. Available keys and semantics:\n"
    ]

    for f in seg_files_info:
        name = f["name"]
        stem = f["stem"]
        unique_labels = f.get("unique_labels")

        mask_type = semantics_override.get(name) or semantics_override.get(f["stem"]) or semantics_override.get(stem.lower())
        if not mask_type:
            for key, value in mask_type_map.items():
                if key in stem.lower():
                    mask_type = value
                    break
        if not mask_type:
            mask_type = f"segmentation mask from `{name}`"

        line = f"  - **seg[\"{stem}\"]**: `{name}` — **{mask_type}**"
        description_parts.append(line)
        if unique_labels is not None and unique_labels > 2:
            description_parts.append(
                f"    **Instance label map**: multiple objects (unique labels: {unique_labels}). "
                "Iterate per object (e.g. skimage.measure.regionprops), then aggregate."
            )

    description_parts.append("\n**Important:** Use `seg.get(\"key\")` or `seg[\"key\"]`; do NOT use position or index.")
    description_parts.append("")

    return "\n".join(description_parts)


def _extract_channel_info(img_array: np.ndarray, dataset_description: Optional[str] = None) -> str:
    """Extract channel information from the image array and dataset description

    Args:
        img_array: Image array
        dataset_description: Dataset description

    Returns:
        Channel information string
    """
    info_parts = []
    
    # Analyze the array shape
    if len(img_array.shape) == 2:
        info_parts.append("- Single channel 2D image")
        info_parts.append(f"- Shape: {img_array.shape} (Height, Width)")
    elif len(img_array.shape) == 3:
        h, w, c = img_array.shape
        if c <= 10:  # Possibly multi-channel
            info_parts.append(f"- Multi-channel 2D image")
            info_parts.append(f"- Shape: {img_array.shape} (Height, Width, Channels)")
            info_parts.append(f"- Number of channels: {c}")
            
            # Try to extract channel names from the dataset description
            if dataset_description:
                desc_lower = dataset_description.lower()
                if "channel 0" in desc_lower or "red" in desc_lower or "actin" in desc_lower:
                    # Try to extract the channel mapping
                    channel_names = []
                    for i in range(c):
                        if i == 0 and ("actin" in desc_lower or "red" in desc_lower):
                            channel_names.append(f"Channel {i}: Actin (Red)")
                        elif i == 1 and ("tubulin" in desc_lower or "green" in desc_lower):
                            channel_names.append(f"Channel {i}: Tubulin (Green)")
                        elif i == 2 and ("dapi" in desc_lower or "blue" in desc_lower or "nuclei" in desc_lower):
                            channel_names.append(f"Channel {i}: DAPI (Blue/Nuclei)")
                        else:
                            channel_names.append(f"Channel {i}: Unknown")
                    
                    if channel_names:
                        info_parts.append("- Channel mapping:")
                        for name in channel_names:
                            info_parts.append(f"  {name}")
        else:
            # Possibly a 3D z-stack
            info_parts.append(f"- 3D z-stack image")
            info_parts.append(f"- Shape: {img_array.shape} (Depth/Channels, Height, Width)")
    elif len(img_array.shape) == 4:
        info_parts.append(f"- 4D image (possibly multi-channel 3D)")
        info_parts.append(f"- Shape: {img_array.shape}")
    else:
        info_parts.append(f"- Shape: {img_array.shape}")
        info_parts.append(f"- Unusual dimensionality, handle carefully")
    
    return "\n".join(info_parts)
