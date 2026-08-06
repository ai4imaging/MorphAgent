"""Data preprocessing module - automatically generate the slices directory"""
import json
import re
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Literal
import numpy as np
import tifffile
from PIL import Image
from tqdm import tqdm

from config import settings

# Try to import scipy (used for illumination correction)
try:
    from scipy.ndimage import median_filter, zoom
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    median_filter = None
    zoom = None


def detect_data_dimension_from_description(description_text: str) -> Literal["3d", "2d_multichannel", "2d_single", "unknown"]:
    """Detect the data dimension type from the dataset description
    
    Args:
        description_text: dataset description text
        
    Returns:
        Data dimension type: "3d", "2d_multichannel", "2d_single", "unknown"
    """
    if not description_text:
        return "unknown"
    
    desc_lower = description_text.lower()
    
    # Keywords for detecting 3D data
    d3_keywords = ["3d", "z-stack", "z stack", "zstack", "volume", "volumetric", 
                   "z-axis", "z axis", "depth", "layers", "slices along z"]
    # Keywords for detecting 2D multi-channel data
    d2_multi_keywords = ["multi-channel", "multichannel", "multi channel", 
                        "rgb", "(512, 512, 3)", "(h, w, c)", "channel collection",
                        "multiple channels", "separate channels"]
    
    # Check for 3D keywords
    has_3d = any(kw in desc_lower for kw in d3_keywords)
    # Check for 2D multi-channel keywords
    has_2d_multi = any(kw in desc_lower for kw in d2_multi_keywords)
    
    # Check shape descriptions, e.g. (512, 512, 3) is usually 2D multi-channel
    shape_pattern = r'\([^)]*\d+[^)]*\d+[^)]*\d+[^)]*\)'
    shape_matches = re.findall(shape_pattern, desc_lower)
    for match in shape_matches:
        # If the shape is in (H, W, C) format and C is small (usually the channel count)
        nums = re.findall(r'\d+', match)
        if len(nums) >= 3:
            try:
                h, w, c = int(nums[0]), int(nums[1]), int(nums[2])
                if c <= 10 and h > 100 and w > 100:  # Channel count is usually small, spatial dimensions are larger
                    has_2d_multi = True
            except:
                pass
    
    # Priority-based decision
    if has_3d and not has_2d_multi:
        return "3d"
    elif has_2d_multi:
        return "2d_multichannel"
    elif "2d" in desc_lower or "single image" in desc_lower:
        return "2d_single"
    else:
        return "unknown"


def extract_channel_mapping_from_description(description_text: str) -> Dict[int, Dict[str, str]]:
    """Extract channel mapping information from the dataset description
    
    Args:
        description_text: dataset description text
        
    Returns:
        Channel mapping dict: {channel_index: {"name": "channel name", "marker": "marker", "color": "color"}}
    """
    channel_mapping = {}
    
    if not description_text:
        return channel_mapping
    
    lines = description_text.split('\n')
    in_channel_table = False
    
    # Look for the channel mapping table
    channel_patterns = [
        r"Channel Number\s*\|\s*Filename Suffix\s*\|\s*Stain Name",
        r"Channel\s*\|\s*Channel Name",
        r"Channel\s*\|\s*Name",
        r"w\s*(\d+)\s*\|\s*w(\d+)",
    ]
    
    for i, line in enumerate(lines):
        # Detect whether we are entering the channel table
        if any(re.search(pattern, line, re.IGNORECASE) for pattern in channel_patterns):
            in_channel_table = True
            continue
        
        # If inside the channel table, extract channel information
        if in_channel_table:
            # Match a channel row, e.g.: "w 1      | w1        | Hoechst                | nucleus"
            # or: "Channel 0 | Red | Actin"
            match = re.search(r'(?:w\s*|channel\s*)(\d+)\s*[|:]\s*([^|:]+)', line, re.IGNORECASE)
            if match:
                channel_num = int(match.group(1))
                channel_info_str = match.group(2).strip()
                
                # Try to extract color and marker
                color = None
                marker = None
                name = channel_info_str
                
                # Check color keywords
                color_keywords = {
                    "red": ["red", "r"],
                    "green": ["green", "g"],
                    "blue": ["blue", "b"],
                    "yellow": ["yellow", "y"],
                }
                for color_name, keywords in color_keywords.items():
                    if any(kw in channel_info_str.lower() for kw in keywords):
                        color = color_name
                        break
                
                # Extract marker (common biological markers)
                marker_keywords = ["actin", "tubulin", "dapi", "hoechst", "phalloidin", 
                                 "tau"]
                for marker_kw in marker_keywords:
                    if marker_kw.lower() in channel_info_str.lower():
                        marker = marker_kw
                        break
                
                channel_mapping[channel_num] = {
                    "name": name,
                    "marker": marker or name,
                    "color": color or ("red" if channel_num == 0 else "green" if channel_num == 1 else "blue")
                }
            elif line.strip() and not line.strip().startswith('-'):
                # If we hit a non-separator, non-channel line, the table may have ended
                if len(channel_mapping) > 0:
                    break
    
    # If no table was found, try to extract channel info from the description
    if not channel_mapping:
        desc_lower = description_text.lower()
        # Look for patterns like "Channel 0 = Actin (Red)"
        channel_desc_patterns = [
            r'channel\s*(\d+)\s*[=:]\s*([^(]+)\s*\(([^)]+)\)',
        ]
        for pattern in channel_desc_patterns:
            matches = re.finditer(pattern, desc_lower, re.IGNORECASE)
            for match in matches:
                channel_num = int(match.group(1))
                marker = match.group(2).strip()
                color = match.group(3).strip()
                channel_mapping[channel_num] = {
                    "name": marker,
                    "marker": marker,
                    "color": color
                }
    
    return channel_mapping


def extract_naming_info_from_description(description_text: str) -> Dict[str, Any]:
    """Extract naming information from the dataset description file
    
    Args:
        description_text: dataset description text
        
    Returns:
        Dict containing channel information and naming rules
    """
    info = {
        "channels": [],
        "channel_names": [],
        "channel_mapping": {},  # Added: channel mapping information
        "naming_pattern": "slice_{:04d}.png",  # Default naming pattern
        "z_axis_name": "z",  # Default z-axis name
        "data_dimension": "unknown",  # Added: data dimension type
    }
    
    # Detect data dimension
    info["data_dimension"] = detect_data_dimension_from_description(description_text)
    
    # Extract channel mapping
    info["channel_mapping"] = extract_channel_mapping_from_description(description_text)
    
    # Build the channel list (for backward compatibility with old code)
    if info["channel_mapping"]:
        for channel_num in sorted(info["channel_mapping"].keys()):
            channel_info = info["channel_mapping"][channel_num]
            info["channels"].append({
                "number": channel_num,
                "suffix": f"w{channel_num + 1}",
                "name": channel_info.get("marker", f"Channel {channel_num}")
            })
            info["channel_names"].append(channel_info.get("marker", f"Channel {channel_num}"))
    
    # Try to extract z-axis naming information
    if description_text:
        lines = description_text.split('\n')
        z_patterns = [
            r"z-axis|z\s*dimension",
            r"slice|plane",
        ]
        for line in lines:
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in z_patterns):
                z_match = re.search(r'(z|slice|plane)\s*[:\-]?\s*(\d+)', line, re.IGNORECASE)
                if z_match:
                    info["z_axis_name"] = z_match.group(1).lower()
    
    return info


def compute_illumination_correction_factor(
    image_paths: List[Path],
    median_window: int,
    downsample_factor: int = 4,
    eps: float = 1e-8
) -> np.ndarray:
    """Compute the Illumination Correction Factor (ICF)
    
    Based on the method of Singh et al. (J. Microscopy 2014):
    ICF = median_filter(mean_over_batch(images), window_size)
    
    Args:
        image_paths: list of image file paths (all images of the same channel)
        median_window: median filter window size (pixels)
        downsample_factor: downsampling factor (used to speed up computation, default 4)
        eps: small value to prevent division by zero
        
    Returns:
        ICF array with the same shape as the input images
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required for illumination correction. Install with: pip install scipy")
    
    if len(image_paths) == 0:
        raise ValueError("Empty image group; cannot compute ICF.")
    
    # Compute the mean in a streaming fashion to avoid loading all images into memory at once
    mean_img = None
    count = 0
    
    for p in image_paths:
        img = tifffile.imread(str(p)).astype(np.float32)
        
        # If it is a multi-channel image, take only the first channel (assuming (C, H, W) format)
        if img.ndim == 3:
            if img.shape[0] <= 10:  # (C, H, W) format
                img = img[0]  # Take the first channel
            elif img.shape[2] <= 10:  # (H, W, C) format
                img = img[:, :, 0]  # Take the first channel
            else:
                # Possibly (Z, H, W); take the first Z layer
                img = img[0]
        
        if img.ndim != 2:
            raise ValueError(f"Expected 2D image after processing, got shape {img.shape} from {p}")
        
        if mean_img is None:
            mean_img = np.zeros_like(img, dtype=np.float32)
        
        mean_img += img
        count += 1
    
    mean_img /= float(count)
    
    # Downsample to speed up the median filter computation
    def downsample_mean_image(img: np.ndarray, factor: int) -> np.ndarray:
        """Downsample using block averaging"""
        if factor <= 1:
            return img
        h, w = img.shape
        h2 = (h // factor) * factor
        w2 = (w // factor) * factor
        trimmed = img[:h2, :w2]
        reshaped = trimmed.reshape(h2 // factor, factor, w2 // factor, factor)
        return reshaped.mean(axis=(1, 3))
    
    def upsample_to_shape(img_small: np.ndarray, target_shape: Tuple[int, int], factor: int) -> np.ndarray:
        """Upsample using bilinear interpolation"""
        if factor <= 1:
            return img_small
        z = zoom(img_small, zoom=(factor, factor), order=1)
        th, tw = target_shape
        return z[:th, :tw]
    
    mean_small = downsample_mean_image(mean_img, downsample_factor)
    
    # Adjust the window size (keep it at least 3)
    win_small = max(3, int(round(median_window / max(1, downsample_factor))))
    if win_small % 2 == 0:
        win_small += 1  # Median filter prefers an odd window size
    
    # Apply the median filter
    icf_small = median_filter(mean_small, size=win_small, mode="reflect")
    icf = upsample_to_shape(icf_small, target_shape=mean_img.shape, factor=downsample_factor)
    
    # Normalize the ICF so its mean is 1.0 (keeps the overall intensity scale stable)
    icf_mean = float(np.mean(icf))
    if icf_mean < eps:
        raise ValueError("ICF mean is near zero; check inputs.")
    icf = icf / icf_mean
    
    # Prevent division by zero
    icf = np.maximum(icf, eps).astype(np.float32)
    
    return icf


def apply_illumination_correction(img: np.ndarray, icf: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Apply illumination correction
    
    I_corr = I / ICF
    
    Args:
        img: input image
        icf: illumination correction factor
        eps: small value to prevent division by zero
        
    Returns:
        Corrected image (preserving the original data type)
    """
    # Ensure the ICF shape matches
    if img.ndim == 3:
        # Multi-channel image: apply the same ICF to each channel
        if img.shape[0] <= 10:  # (C, H, W) format
            corrected = np.zeros_like(img, dtype=np.float32)
            for c in range(img.shape[0]):
                if icf.shape != img[c].shape:
                    raise ValueError(f"Shape mismatch: ICF {icf.shape} vs image channel {img[c].shape}")
                corrected[c] = img[c].astype(np.float32) / np.maximum(icf, eps)
            # Convert back to the original data type
            if img.dtype == np.uint8:
                corrected = np.clip(corrected, 0, 255).astype(np.uint8)
            elif img.dtype == np.uint16:
                corrected = np.clip(corrected, 0, 65535).astype(np.uint16)
            return corrected
        elif img.shape[2] <= 10:  # (H, W, C) format
            corrected = np.zeros_like(img, dtype=np.float32)
            for c in range(img.shape[2]):
                if icf.shape != img[:, :, c].shape:
                    raise ValueError(f"Shape mismatch: ICF {icf.shape} vs image channel {img[:, :, c].shape}")
                corrected[:, :, c] = img[:, :, c].astype(np.float32) / np.maximum(icf, eps)
            if img.dtype == np.uint8:
                corrected = np.clip(corrected, 0, 255).astype(np.uint8)
            elif img.dtype == np.uint16:
                corrected = np.clip(corrected, 0, 65535).astype(np.uint16)
            return corrected
        else:
            # (Z, H, W) format: apply to each Z layer
            corrected = np.zeros_like(img, dtype=np.float32)
            for z in range(img.shape[0]):
                if icf.shape != img[z].shape:
                    raise ValueError(f"Shape mismatch: ICF {icf.shape} vs image slice {img[z].shape}")
                corrected[z] = img[z].astype(np.float32) / np.maximum(icf, eps)
            if img.dtype == np.uint8:
                corrected = np.clip(corrected, 0, 255).astype(np.uint8)
            elif img.dtype == np.uint16:
                corrected = np.clip(corrected, 0, 65535).astype(np.uint16)
            return corrected
    else:
        # 2D image
        if icf.shape != img.shape:
            raise ValueError(f"Shape mismatch: ICF {icf.shape} vs image {img.shape}")
        corrected = (img.astype(np.float32) / np.maximum(icf, eps))
        if img.dtype == np.uint8:
            corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        elif img.dtype == np.uint16:
            corrected = np.clip(corrected, 0, 65535).astype(np.uint16)
        else:
            corrected = corrected.astype(img.dtype)
        return corrected


def normalize_image_percentile(img: np.ndarray, lower_percentile: float = 1.0, upper_percentile: float = 99.0) -> np.ndarray:
    """Normalize an image using percentiles to avoid it being too dark
    
    Args:
        img: input image array
        lower_percentile: lower percentile (default 1.0)
        upper_percentile: upper percentile (default 99.0)
        
    Returns:
        Normalized uint8 image array
    """
    if img.dtype == np.uint8:
        return img
    
    # Compute the percentiles
    p_lower = np.percentile(img, lower_percentile)
    p_upper = np.percentile(img, upper_percentile)
    
    # If the lower and upper percentiles are the same or close, use min-max normalization
    if abs(p_upper - p_lower) < 1e-10:
        p_lower = img.min()
        p_upper = img.max()
        if abs(p_upper - p_lower) < 1e-10:
            # If the image is constant, return a zero image
            return np.zeros_like(img, dtype=np.uint8)
    
    # Clip to the percentile range
    img_clipped = np.clip(img, p_lower, p_upper)
    
    # Normalize to 0-255
    img_normalized = ((img_clipped - p_lower) / (p_upper - p_lower) * 255).astype(np.uint8)
    
    return img_normalized


def generate_slices_from_2d_image(
    image_path: Path,
    output_dir: Path,
    naming_info: Optional[Dict[str, Any]] = None,
) -> List[Path]:
    """Split a PNG/JPEG image into one grayscale slice per image channel."""

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(image_path) as source:
            if source.mode == "P":
                source = source.convert("RGBA" if "transparency" in source.info else "RGB")
            bands = list(source.getbands())
            image = np.asarray(source)
    except Exception as exc:
        print(f"  [WARN]  Warning: failed to process 2D image {image_path}: {exc}")
        return []

    if image.ndim == 3 and bands and bands[-1] == "A":
        alpha = image[..., -1]
        opaque_value = (
            np.iinfo(alpha.dtype).max
            if np.issubdtype(alpha.dtype, np.integer)
            else 1.0
        )
        if np.all(alpha == opaque_value):
            image = image[..., :-1]
            bands = bands[:-1]

    if image.ndim == 2:
        channels = [image]
        bands = bands[:1] or ["L"]
    elif image.ndim == 3 and image.shape[-1] >= 1:
        channels = [image[..., index] for index in range(image.shape[-1])]
        if len(bands) != len(channels):
            bands = [f"Channel {index}" for index in range(len(channels))]
    else:
        print(f"  [WARN]  Warning: unsupported 2D image shape {image.shape}: {image_path}")
        return []

    channel_mapping = (
        naming_info.get("channel_mapping", {})
        if isinstance(naming_info, dict)
        else {}
    )
    generated_paths: List[Path] = []
    mapping: Dict[int, Dict[str, str]] = {}
    for index, channel in enumerate(channels):
        configured = channel_mapping.get(index, {}) if isinstance(channel_mapping, dict) else {}
        band_name = str(bands[index]) if index < len(bands) else f"Channel {index}"
        marker = str(configured.get("marker") or band_name or f"Channel {index}")
        marker_clean = re.sub(r"[^A-Za-z0-9_]+", "_", marker).strip("_") or f"channel_{index}"
        filename = (
            "slice_0000.png"
            if len(channels) == 1
            else f"slice_{index:04d}_{marker_clean}.png"
        )
        output_path = output_dir / filename
        normalized = normalize_image_percentile(np.asarray(channel))
        Image.fromarray(normalized, mode="L").save(output_path)
        generated_paths.append(output_path)
        mapping[index] = {
            "marker": marker,
            "color": str(configured.get("color") or band_name).lower(),
        }

    if len(channels) > 1:
        mapping_data = {
            "data_dimension": "2d_multichannel",
            "num_channels": len(channels),
            "channel_mapping": mapping,
            "slice_files": {
                index: path.name
                for index, path in enumerate(generated_paths)
            },
        }
        with (output_dir / "channel_mapping.json").open("w", encoding="utf-8") as handle:
            json.dump(mapping_data, handle, indent=2, ensure_ascii=False)
    return generated_paths


def _infer_array_axes(
    shape: Tuple[int, ...],
    data_dimension: Optional[str] = None,
) -> str:
    """Infer conservative axis labels when a file format has no axis metadata."""

    if len(shape) == 2:
        return "YX"
    if len(shape) == 3:
        if shape[-1] <= 4 and shape[-1] < min(shape[0], shape[1]):
            return "YXC"
        if data_dimension == "2d_multichannel":
            return "CYX"
        if data_dimension == "3d":
            return "ZYX"
        if shape[0] <= 4 and shape[0] < min(shape[1], shape[2]):
            return "CYX"
        return "ZYX"

    leading = len(shape) - 2
    unknown_axes = iter("ABDEFGHIJKLMNOPQRSUVW")
    labels = [next(unknown_axes, "Q") for _index in range(leading)]
    small = [
        index
        for index, size in enumerate(shape[:-2])
        if size <= 4
    ]
    if small:
        labels[small[-1]] = "C"
    remaining = [index for index, label in enumerate(labels) if label != "C"]
    if remaining:
        labels[remaining[-1]] = "Z"
    if len(remaining) > 1:
        labels[remaining[-2]] = "T"
    return "".join(labels) + "YX"


def generate_slices_from_nd_array(
    image: np.ndarray,
    output_dir: Path,
    *,
    axes: Optional[str] = None,
    naming_info: Optional[Dict[str, Any]] = None,
    source_name: str = "",
) -> List[Path]:
    """Expand every non-spatial dimension of an image array into 2D PNG slices."""

    array = np.asarray(image)
    if array.ndim < 2:
        print(f"  [WARN]  Warning: image has fewer than 2 dimensions: {source_name}")
        return []
    data_dimension = (
        str(naming_info.get("data_dimension") or "unknown")
        if isinstance(naming_info, dict)
        else "unknown"
    )
    axis_labels = str(axes or "").upper()
    if (
        len(axis_labels) != array.ndim
        or any(label not in set("TCZYXS") for label in axis_labels)
    ):
        axis_labels = _infer_array_axes(tuple(array.shape), data_dimension)
    axis_labels = axis_labels.replace("S", "C")

    keep_indices = [
        index
        for index, size in enumerate(array.shape)
        if size != 1 or axis_labels[index] in {"Y", "X"}
    ]
    if len(keep_indices) != array.ndim:
        array = np.squeeze(array, axis=tuple(
            index for index in range(array.ndim) if index not in keep_indices
        ))
        axis_labels = "".join(axis_labels[index] for index in keep_indices)
    if array.ndim < 2:
        return []

    if "Y" in axis_labels and "X" in axis_labels:
        y_axis = axis_labels.index("Y")
        x_axis = axis_labels.index("X")
    else:
        y_axis, x_axis = array.ndim - 2, array.ndim - 1
        labels = list(axis_labels)
        labels[y_axis], labels[x_axis] = "Y", "X"
        axis_labels = "".join(labels)
    leading_axes = [
        index for index in range(array.ndim) if index not in {y_axis, x_axis}
    ]
    permutation = leading_axes + [y_axis, x_axis]
    array = np.transpose(array, permutation)
    ordered_axes = "".join(axis_labels[index] for index in permutation)
    non_spatial_axes = ordered_axes[:-2]
    non_spatial_shape = array.shape[:-2]

    output_dir.mkdir(parents=True, exist_ok=True)
    channel_mapping = (
        naming_info.get("channel_mapping", {})
        if isinstance(naming_info, dict)
        else {}
    )
    generated: List[Path] = []
    frames: List[Dict[str, Any]] = []
    indices = list(np.ndindex(non_spatial_shape)) if non_spatial_shape else [()]
    for serial, index_tuple in enumerate(indices):
        plane = array[index_tuple] if index_tuple else array
        if plane.ndim != 2:
            print(
                f"  [WARN]  Warning: could not reduce frame to 2D "
                f"({plane.shape}): {source_name}"
            )
            continue
        coordinates = {
            axis: int(value)
            for axis, value in zip(non_spatial_axes, index_tuple)
        }
        channel_index = coordinates.get("C")
        configured = (
            channel_mapping.get(channel_index, {})
            if channel_index is not None and isinstance(channel_mapping, dict)
            else {}
        )
        marker = str(
            configured.get("marker")
            or (f"Channel {channel_index}" if channel_index is not None else "")
        )
        marker_clean = re.sub(r"[^A-Za-z0-9_]+", "_", marker).strip("_")
        if not coordinates:
            filename = "slice_0000.png"
        elif set(coordinates) == {"C"}:
            filename = f"slice_{channel_index:04d}_{marker_clean or f'channel_{channel_index}'}.png"
        else:
            coordinate_text = "_".join(
                f"{axis.lower()}{value:04d}"
                for axis, value in coordinates.items()
            )
            if marker_clean:
                coordinate_text += f"_{marker_clean}"
            filename = f"slice_{coordinate_text}.png"
        output_path = output_dir / filename
        Image.fromarray(normalize_image_percentile(plane), mode="L").save(output_path)
        generated.append(output_path)
        frames.append(
            {
                "file": output_path.name,
                "indices": coordinates,
                "serial": serial,
            }
        )

    manifest = {
        "source_file": source_name,
        "source_shape": list(np.asarray(image).shape),
        "axes": ordered_axes,
        "frame_count": len(generated),
        "frames": frames,
    }
    with (output_dir / "slice_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    if "C" in non_spatial_axes:
        channel_count = non_spatial_shape[non_spatial_axes.index("C")]
        mapping = {
            index: channel_mapping.get(
                index,
                {"marker": f"Channel {index}", "color": "unknown"},
            )
            for index in range(channel_count)
        }
        with (output_dir / "channel_mapping.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "data_dimension": (
                        "2d_multichannel"
                        if set(non_spatial_axes) == {"C"}
                        else "multidimensional"
                    ),
                    "num_channels": channel_count,
                    "channel_mapping": mapping,
                    "slice_files": (
                        {
                            index: next(
                                frame["file"]
                                for frame in frames
                                if frame["indices"].get("C") == index
                            )
                            for index in range(channel_count)
                        }
                        if set(non_spatial_axes) == {"C"}
                        else {
                            index: [
                                frame["file"]
                                for frame in frames
                                if frame["indices"].get("C") == index
                            ]
                            for index in range(channel_count)
                        }
                    ),
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
    return generated


def generate_slices_from_image_file(
    image_path: Path,
    output_dir: Path,
    naming_info: Optional[Dict[str, Any]] = None,
) -> List[Path]:
    """Read a supported raster, TIFF/OME-TIFF, or MRC volume and create slices."""

    suffix = image_path.suffix.lower()
    data_dimension = (
        str(naming_info.get("data_dimension") or "unknown")
        if isinstance(naming_info, dict)
        else "unknown"
    )
    try:
        if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}:
            with Image.open(image_path) as source:
                frame_count = int(getattr(source, "n_frames", 1) or 1)
                if frame_count > 1:
                    frames = []
                    use_alpha = "A" in source.getbands() or "transparency" in source.info
                    for frame_index in range(frame_count):
                        source.seek(frame_index)
                        frame = source.convert("RGBA" if use_alpha else "RGB")
                        frames.append(np.asarray(frame))
                    stack = np.stack(frames, axis=0)
                    if use_alpha and np.all(stack[..., -1] == 255):
                        stack = stack[..., :-1]
                    return generate_slices_from_nd_array(
                        stack,
                        output_dir,
                        axes="TYXC",
                        naming_info=naming_info,
                        source_name=image_path.name,
                    )
            return generate_slices_from_2d_image(image_path, output_dir, naming_info)
        if suffix in {".tif", ".tiff"}:
            with tifffile.TiffFile(str(image_path)) as tif:
                series = tif.series[0]
                image = series.asarray()
                axes = getattr(series, "axes", None)
                page_count = len(tif.pages)
            if (
                image.ndim == 3
                and page_count == image.shape[0]
                and page_count > 1
                and str(axes or "").upper() in {"QYX", "SYX", "IYX"}
            ):
                axes = "ZYX"
            return generate_slices_from_nd_array(
                image,
                output_dir,
                axes=axes,
                naming_info=naming_info,
                source_name=image_path.name,
            )
        if suffix in {".mrc", ".map", ".rec"}:
            try:
                import mrcfile
            except ImportError as exc:
                raise ImportError(
                    "MRC input requires the 'mrcfile' package. "
                    "Install project requirements and retry."
                ) from exc
            with mrcfile.open(str(image_path), permissive=True) as handle:
                image = np.asarray(handle.data).copy()
            axes = "YX" if image.ndim == 2 else "ZYX" if image.ndim == 3 else None
            return generate_slices_from_nd_array(
                image,
                output_dir,
                axes=axes,
                naming_info=naming_info,
                source_name=image_path.name,
            )
    except Exception as exc:
        print(f"  [WARN]  Warning: failed to read {image_path}: {exc}")
        return []
    print(f"  [WARN]  Warning: unsupported image format: {image_path}")
    return []


def generate_slices_from_3d_tiff(
    tiff_path: Path,
    output_dir: Path,
    naming_info: Optional[Dict[str, Any]] = None,
    channel_index: Optional[int] = None,
    data_dimension: Optional[Literal["3d", "2d_multichannel", "2d_single", "unknown"]] = None,
    icf_cache: Optional[Dict[int, np.ndarray]] = None
) -> List[Path]:
    """Generate slices (PNG format) from a TIFF file, handled intelligently based on the data dimension type
    
    Args:
        tiff_path: input TIFF file path
        output_dir: output directory (slices directory)
        naming_info: naming information (extracted from description), includes data_dimension and channel_mapping
        channel_index: if the TIFF is multi-channel, the channel index to use (None means process all channels)
        data_dimension: data dimension type (if provided, it takes priority; otherwise obtained from naming_info)
        
    Returns:
        List of generated PNG file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Read the TIFF file
        img = tifffile.imread(str(tiff_path))
        
        # If an ICF cache is provided and illumination correction is enabled, apply the ICF
        if icf_cache is not None and settings.enable_illumination_correction and SCIPY_AVAILABLE:
            try:
                if img.ndim == 3 and img.shape[0] <= 10:  # (C, H, W) format
                    # Apply the corresponding ICF to each channel
                    for c in range(min(img.shape[0], len(icf_cache))):
                        if c in icf_cache:
                            # Extract a single channel, apply the ICF, then put it back
                            channel_img = img[c:c+1].copy()
                            corrected_channel = apply_illumination_correction(channel_img, icf_cache[c])
                            img[c] = corrected_channel[0] if corrected_channel.ndim == 3 else corrected_channel
                elif len(icf_cache) > 0:
                    # Use the first ICF (if there is only one)
                    icf = list(icf_cache.values())[0]
                    img = apply_illumination_correction(img, icf)
            except Exception as e:
                print(f"  \u26a0\ufe0f  Warning: failed to apply ICF to {tiff_path.name}: {e}; continuing with the original image")
        
        # Determine the data dimension type
        if data_dimension is None:
            if naming_info and naming_info.get("data_dimension"):
                data_dimension = naming_info["data_dimension"]
            else:
                # Fall back to auto-detection
                data_dimension = "unknown"
        
        # Get channel mapping information
        channel_mapping = {}
        if naming_info and naming_info.get("channel_mapping"):
            channel_mapping = naming_info["channel_mapping"]
        
        # Handle based on the data dimension type
        if img.ndim == 2:
            # 2D single-channel image; save directly
            img_normalized = normalize_image_percentile(img)
            output_path = output_dir / f"slice_0000.png"
            Image.fromarray(img_normalized, mode='L').save(output_path)
            return [output_path]
        
        elif img.ndim == 3:
            # 3D array, possibly (Z, H, W), (C, H, W), or (H, W, C)
            
            # Determine the data format:
            # - If the first dimension is small (<=10) and smaller than the others, it is likely (C, H, W) format (multi-channel 2D)
            # - If the last dimension is small (<=10) and smaller than the others, it is likely (H, W, C) format (needs conversion)
            # - Otherwise, it may be (Z, H, W) format (3D Z-stack)
            
            # Check whether it is (H, W, C) format (needs conversion to (C, H, W))
            if img.shape[2] <= 10 and img.shape[2] < img.shape[0] and img.shape[2] < img.shape[1]:
                # (H, W, C) format; convert to (C, H, W)
                h, w, c = img.shape
                img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
            
            # Now determine whether it is (C, H, W) or (Z, H, W)
            if img.shape[0] <= 10 and img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
                # (C, H, W) format: multi-channel 2D data; generate one slice per channel
                num_channels = img.shape[0]
                generated_paths = []
                
                # Generate one slice per channel
                for c in range(num_channels):
                    channel_img = img[c]
                    channel_img_normalized = normalize_image_percentile(channel_img)
                    
                    # Generate the filename (if channel information is available)
                    if channel_mapping and c in channel_mapping:
                        marker = channel_mapping[c].get("marker", f"Channel_{c}")
                        marker_clean = marker.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                        filename = f"slice_{c:04d}_{marker_clean}.png"
                    else:
                        # No channel mapping provided -> generic channel names.
                        filename = f"slice_{c:04d}_channel_{c}.png"
                    
                    output_path = output_dir / filename
                    Image.fromarray(channel_img_normalized, mode='L').save(output_path)
                    generated_paths.append(output_path)
                
                # Save channel mapping information to a JSON file
                mapping_file = output_dir / "channel_mapping.json"
                if channel_mapping:
                    mapping_data = {
                        "data_dimension": "2d_multichannel",
                        "num_channels": num_channels,
                        "channel_mapping": channel_mapping,
                        "slice_files": {c: Path(p).name for c, p in enumerate(generated_paths)}
                    }
                else:
                    # No channel mapping provided -> generic channel names/colors.
                    mapping_data = {
                        "data_dimension": "2d_multichannel",
                        "num_channels": num_channels,
                        "channel_mapping": {
                            c: {"marker": f"Channel {c}",
                                "color": ["red", "green", "blue"][c] if c < 3 else "unknown"}
                            for c in range(num_channels)
                        },
                        "slice_files": {c: Path(p).name for c, p in enumerate(generated_paths)}
                    }
                
                with open(mapping_file, 'w', encoding='utf-8') as f:
                    json.dump(mapping_data, f, indent=2, ensure_ascii=False)
                
                return generated_paths
            
            # Decide the handling based on the data_dimension flag (3D data)
            if data_dimension == "3d":
                # 3D data: slice along the Z axis
                # Assume the format is (Z, H, W)
                num_slices = img.shape[0]
                generated_paths = []
                
                for z in range(num_slices):
                    slice_img = img[z]
                    slice_img_normalized = normalize_image_percentile(slice_img)
                    
                    # Generate the filename
                    if naming_info and naming_info.get("naming_pattern"):
                        pattern = naming_info["naming_pattern"]
                        if "{:04d}" in pattern:
                            filename = pattern.format(z)
                        else:
                            filename = f"slice_{z:04d}.png"
                    else:
                        filename = f"slice_{z:04d}.png"
                    
                    output_path = output_dir / filename
                    Image.fromarray(slice_img_normalized, mode='L').save(output_path)
                    generated_paths.append(output_path)
                
                return generated_paths
            
            else:
                # unknown or other cases: try to determine automatically
                # If the first dimension is small and <=10, it may be the channel dimension; handle as multi-channel
                if img.shape[0] <= 10 and img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
                    # (C, H, W) format: multi-channel 2D data; generate one slice per channel
                    num_channels = img.shape[0]
                    generated_paths = []
                    
                    for c in range(num_channels):
                        channel_img = img[c]
                        channel_img_normalized = normalize_image_percentile(channel_img)
                        filename = f"slice_{c:04d}_channel_{c}.png"
                        output_path = output_dir / filename
                        Image.fromarray(channel_img_normalized, mode='L').save(output_path)
                        generated_paths.append(output_path)
                    
                    return generated_paths
                else:
                    # Handle along the Z axis (3D data)
                    num_slices = img.shape[0]
                    generated_paths = []
                    for z in range(num_slices):
                        slice_img = img[z]
                        slice_img_normalized = normalize_image_percentile(slice_img)
                        filename = f"slice_{z:04d}.png"
                        output_path = output_dir / filename
                        Image.fromarray(slice_img_normalized, mode='L').save(output_path)
                        generated_paths.append(output_path)
                    return generated_paths
        
        elif img.ndim == 4:
            # 4D array, possibly (Z, C, H, W) or (C, Z, H, W)
            if data_dimension == "3d":
                # 3D data: slice along the Z axis; each Z layer may have multiple channels
                num_z = img.shape[0]
                num_channels = img.shape[1] if img.shape[1] <= 10 else 1
                generated_paths = []
                
                for z in range(num_z):
                    if num_channels > 1:
                        # Multi-channel: combine into RGB
                        r_channel = normalize_image_percentile(img[z, 0])
                        g_channel = normalize_image_percentile(img[z, 1]) if num_channels > 1 else r_channel
                        b_channel = normalize_image_percentile(img[z, 2]) if num_channels > 2 else r_channel
                        rgb_img = np.stack([r_channel, g_channel, b_channel], axis=2)
                        filename = f"slice_z{z:04d}.png"
                        output_path = output_dir / filename
                        Image.fromarray(rgb_img, mode='RGB').save(output_path)
                    else:
                        # Single channel
                        slice_img = normalize_image_percentile(img[z, 0])
                        filename = f"slice_z{z:04d}.png"
                        output_path = output_dir / filename
                        Image.fromarray(slice_img, mode='L').save(output_path)
                    generated_paths.append(output_path)
                
                return generated_paths
            else:
                # Other cases: simplified handling
                num_z = img.shape[0]
                generated_paths = []
                for z in range(num_z):
                    slice_img = normalize_image_percentile(img[z, 0] if img.shape[1] > 0 else img[z])
                    filename = f"slice_z{z:04d}.png"
                    output_path = output_dir / filename
                    Image.fromarray(slice_img, mode='L').save(output_path)
                    generated_paths.append(output_path)
                return generated_paths
        
        else:
            print(f"  [WARN]  Warning: unsupported image dimension {img.ndim}D: {tiff_path}")
            return []
    
    except Exception as e:
        print(f"  [WARN]  Warning: failed to process TIFF file {tiff_path}: {e}")
        import traceback
        traceback.print_exc()
        return []


def ensure_slices_directory(
    sample_dir: Path,
    description_text: Optional[str] = None,
    secondary_dir_name: str = "slices",
    icf_cache: Optional[Dict[int, np.ndarray]] = None
) -> Tuple[bool, List[Path]]:
    """Ensure the slices directory exists; generate it automatically if it does not
    
    Args:
        sample_dir: sample directory
        description_text: dataset description text (used to extract naming rules)
        secondary_dir_name: name of the secondary directory (default "slices")
        
    Returns:
        (whether it already existed, list of generated slices file paths)
    """
    slices_dir = sample_dir / secondary_dir_name
    
    # If the slices directory already exists and is not empty, check whether the file format is correct
    if slices_dir.exists() and any(slices_dir.iterdir()):
        existing_files = list(slices_dir.glob("*.png"))
        if existing_files:
            # Check whether the first file is a valid RGB image
            try:
                from PIL import Image
                test_img = Image.open(existing_files[0])
                width, height = test_img.size
                mode = test_img.mode
                test_img.close()
                
                # Check the image format: should be grayscale (L mode) or RGB, with reasonable dimensions
                # If the image dimensions are abnormal, regenerate
                if (width < 10 or height < 10 or 
                    (width == 3 and height == 512) or 
                    (width == 512 and height == 3)):
                    # Old slice files have the wrong format; delete and regenerate
                    shutil.rmtree(slices_dir)
                    slices_dir.mkdir(parents=True, exist_ok=True)
                else:
                    # File format is correct; check whether channel_mapping.json exists
                    mapping_file = slices_dir / "channel_mapping.json"
                    if not mapping_file.exists() and len(existing_files) > 0:
                        # If there is no channel mapping file, try to infer the channel count from the filenames
                        # Use the number of slice files as the channel count
                        num_slices = len(existing_files)
                        if num_slices <= 10:  # A reasonable channel count
                            try:
                                with open(mapping_file, 'w', encoding='utf-8') as f:
                                    # No dataset channel info -> generic per-channel names.
                                    mapping_data = {
                                        "data_dimension": "2d_multichannel" if num_slices <= 10 else "3d",
                                        "num_channels": num_slices if num_slices <= 10 else None,
                                        "channel_mapping": {c: {"marker": f"Channel {c}"} for c in range(num_slices)},
                                        "slice_files": {c: Path(p).name for c, p in enumerate(sorted(existing_files))}
                                    }
                                    json.dump(mapping_data, f, indent=2, ensure_ascii=False)
                            except Exception as e:
                                # If generation fails, it does not affect the main flow
                                pass
                    return True, existing_files
            except Exception as e:
                # If the check fails, delete and regenerate
                if slices_dir.exists():
                    shutil.rmtree(slices_dir)
                slices_dir.mkdir(parents=True, exist_ok=True)
    
    # Need to generate slices (do not print, to avoid interrupting the progress bar)
    
    # Extract naming information and data dimension
    naming_info = None
    data_dimension = None
    if description_text:
        naming_info = extract_naming_info_from_description(description_text)
        data_dimension = naming_info.get("data_dimension", "unknown")
        # Do not print channel information, to avoid interrupting the progress bar
    
    # Look for primary files (raw data)
    from config import settings
    image_extensions = settings.image_extensions
    primary_files = []
    
    for ext in image_extensions:
        primary_files.extend(list(sample_dir.glob(f"*{ext}")))
        primary_files.extend(list(sample_dir.glob(f"*{ext.upper()}")))
    
    if not primary_files:
        print(f"    [WARN]  Warning: no primary files (raw data) found")
        return False, []
    
    # Process each primary file
    all_generated_paths = []
    
    for primary_file in tqdm(primary_files, desc=f"    Processing files", leave=False, ncols=60):
        suffix = primary_file.suffix.lower()
        if (
            suffix in {".tif", ".tiff"}
            and settings.enable_illumination_correction
            and icf_cache
        ):
            # Keep the legacy TIFF path when illumination correction is explicitly enabled.
            generated = generate_slices_from_3d_tiff(
                primary_file,
                slices_dir,
                naming_info,
                channel_index=None,
                data_dimension=data_dimension,
                icf_cache=icf_cache
            )
            all_generated_paths.extend(generated)
        else:
            generated = generate_slices_from_image_file(
                primary_file,
                slices_dir,
                naming_info,
            )
            all_generated_paths.extend(generated)
    
    if not all_generated_paths:
        print(f"    [WARN]  Warning: failed to generate any slice files")
    
    return False, all_generated_paths


def preprocess_all_samples(
    data_root: Path,
    sample_ids: List[str],
    description_text: Optional[str] = None,
    secondary_dir_name: str = "slices"
) -> Dict[str, Tuple[bool, List[Path]]]:
    """Preprocess all samples, ensuring the slices directory exists
    
    If illumination correction is enabled, the ICF is computed first and applied to the images.
    
    Args:
        data_root: dataset root directory
        sample_ids: list of sample IDs
        description_text: dataset description text
        secondary_dir_name: name of the secondary directory
        
    Returns:
        Dict where the key is sample_id and the value is (whether it already existed, list of slices file paths)
    """
    results = {}
    
    # If illumination correction is enabled, compute the ICF first
    icf_cache = {}  # Cache of ICFs, keyed by channel index
    if settings.enable_illumination_correction:
        if not SCIPY_AVAILABLE:
            print("[WARN]  Warning: illumination correction is enabled, but scipy is not installed. Skipping illumination correction.")
            print("Install command: pip install scipy")
        else:
            print(f"\n[Data Preprocessing] Computing Illumination Correction Factor (ICF)...")
            
            # Collect all TIFF files
            all_tiff_files = []
            for sample_id in sample_ids:
                sample_dir = data_root / sample_id
                if not sample_dir.exists():
                    continue
                tiff_files = list(sample_dir.glob("*.tif")) + list(sample_dir.glob("*.tiff"))
                all_tiff_files.extend(tiff_files)
            
            if len(all_tiff_files) == 0:
                print("[WARN]  Warning: no TIFF files found; skipping illumination correction")
            else:
                # Group by channel (assuming all images are in (C, H, W) format)
                if settings.illumination_correction_group_by_channel:
                    # Read the first file to determine the number of channels
                    try:
                        first_img = tifffile.imread(str(all_tiff_files[0]))
                        if first_img.ndim == 3 and first_img.shape[0] <= 10:
                            num_channels = first_img.shape[0]
                        elif first_img.ndim == 3 and first_img.shape[2] <= 10:
                            num_channels = first_img.shape[2]
                        else:
                            num_channels = 1
                        
                        # Compute the ICF grouped by channel
                        for channel_idx in range(num_channels):
                            try:
                                icf = compute_illumination_correction_factor(
                                    image_paths=all_tiff_files,
                                    median_window=settings.illumination_correction_median_window,
                                    downsample_factor=settings.illumination_correction_downsample_factor
                                )
                                icf_cache[channel_idx] = icf
                                print(f"  [OK] Channel {channel_idx} ICF computed (shape: {icf.shape})")
                            except Exception as e:
                                print(f"  [WARN]  Warning: ICF computation failed for channel {channel_idx}: {e}")
                    except Exception as e:
                        print(f"  [WARN]  Warning: unable to determine the number of channels; skipping grouping by channel: {e}")
                else:
                    # Compute over all images together (not grouped by channel)
                    try:
                        icf = compute_illumination_correction_factor(
                            image_paths=all_tiff_files,
                            median_window=settings.illumination_correction_median_window,
                            downsample_factor=settings.illumination_correction_downsample_factor
                        )
                        icf_cache[0] = icf  # Use channel 0 as the key
                        print(f"  [OK] ICF computed (shape: {icf.shape})")
                    except Exception as e:
                        print(f"  [WARN]  Warning: ICF computation failed: {e}")
    
    # Store the ICF cache in a module-level variable for use by generate_slices_from_3d_tiff
    _icf_cache = icf_cache
    
    print(f"\n[Data Preprocessing] Checking and generating the {secondary_dir_name} directory...")
    
    for sample_id in tqdm(sample_ids, desc="  Preprocessing samples", ncols=80):
        sample_dir = data_root / sample_id
        if not sample_dir.exists():
            results[sample_id] = (False, [])
            continue
        
        existed, paths = ensure_slices_directory(
            sample_dir,
            description_text,
            secondary_dir_name,
            icf_cache=_icf_cache if settings.enable_illumination_correction and SCIPY_AVAILABLE else None
        )
        results[sample_id] = (existed, paths)
    
    # Statistics
    existed_count = sum(1 for existed, _ in results.values() if existed)
    generated_count = len(sample_ids) - existed_count
    
    print(f"  [OK] Preprocessing complete: {existed_count} samples already existed, {generated_count} samples generated")
    
    return results

