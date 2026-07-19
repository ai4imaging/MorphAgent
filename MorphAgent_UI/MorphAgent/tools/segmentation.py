"""Segmentation tool - Cellpose-SAM wrapper

Use cellpose-SAM to segment images, executed in a dedicated cellpose conda environment.
Supports user-uploaded segmentation files (same scan rules as data_path_selector).
"""
from typing import Optional, Any, List, Dict, Tuple
from pathlib import Path
import subprocess
import json
import os
import sys


def _seg_env() -> str:
    """Resolve the conda env that has Cellpose-SAM installed (configurable)."""
    try:
        from config import settings
        return settings.segmentation_conda_env
    except Exception:
        return os.getenv("SEGMENTATION_CONDA_ENV", os.getenv("CONDA_ENV", "morphagent"))


def list_segmentation_files(sample_dir: Path) -> List[Tuple[str, Path]]:
    """Scan all image files under sample_dir/segmentation/ and return (name, path) sorted by filename.
    Uses the same image_extensions as data_path_selector to ensure a single consistent source.

    Args:
        sample_dir: Path to the sample directory

    Returns:
        [(filename, Path), ...] sorted by filename, excluding directories
    """
    try:
        from config import settings
        image_extensions = settings.image_extensions
    except Exception:
        image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}
    seg_dir = sample_dir / "segmentation"
    out = []
    if not seg_dir.exists() or not seg_dir.is_dir():
        return out
    for seg_file in seg_dir.iterdir():
        if seg_file.is_file() and seg_file.suffix.lower() in image_extensions:
            out.append((seg_file.name, seg_file))
    out.sort(key=lambda x: x[0])
    return out


def segment_image_with_cellpose(
    input_image_path: str,
    output_mask_path: str,
    channels: Optional[List[int]] = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    tile_norm_blocksize: int = 0,
    batch_size: int = 32,
    conda_env: Optional[str] = None
) -> bool:
    """Segment an image using cellpose-SAM.

    Runs the segmentation in the specified conda environment and saves the result to the given path.

    Args:
        input_image_path: Input image path (TIFF format)
        output_mask_path: Output mask path (TIFF format)
        channels: List of channel indices to use (0-based), e.g. [0, 1] means use channels 0 and 1
        flow_threshold: Flow threshold, default 0.4. Increasing this reduces the number of returned masks
        cellprob_threshold: Cell probability threshold, default 0.0. Lowering this increases the number of returned masks
        tile_norm_blocksize: Normalization block size, default 0 (whole-image normalization)
        batch_size: Batch size, default 32
        conda_env: conda environment name, default "cellpose"

    Returns:
        True if segmentation succeeds, otherwise False
    """
    conda_env = conda_env or _seg_env()
    input_path = Path(input_image_path)
    output_path = Path(output_mask_path)
    
    if not input_path.exists():
        print(f"  ⚠️  Input image does not exist: {input_path}")
        return False
    
    # Make sure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Find the segmentation script path (inside MorphAgent)
    # The script is located at tools/segment_tif_with_cpsam.py
    current_file = Path(__file__).resolve()
    script_path = current_file.parent / "segment_tif_with_cpsam.py"
    if not script_path.exists():
        print(f"  ⚠️  Segmentation script does not exist: {script_path}")
        return False
    
    # Determine the output directory (the script automatically saves three files to the segmentation directory)
    output_dir = output_path.parent
    
    # Build the command
    cmd = [
        "conda", "run", "-n", conda_env,
        "python", str(script_path),
        str(input_path),
        "-o", str(output_dir),  # pass the directory path
        "--flow_threshold", str(flow_threshold),
        "--cellprob_threshold", str(cellprob_threshold),
        "--tile_norm_blocksize", str(tile_norm_blocksize),
        "--batch_size", str(batch_size)
    ]
    
    # If channels are specified, add the channel arguments
    if channels is not None and len(channels) > 0:
        cmd.extend(["-c"] + [str(c) for c in channels])
    
    # Execute the command
    try:
        print(f"  [Segmentation] Running segmentation: {input_path.name}")
        print(f"  [Segmentation] Using environment: {conda_env}")
        if channels:
            print(f"  [Segmentation] Using channels: {channels}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Check whether all three files exist
        cyto_file = output_dir / "cyto.tif"
        nuclei_file = output_dir / "nuclei.tif"
        cytoplasm_file = output_dir / "cytoplasm.tif"
        
        if cyto_file.exists() and nuclei_file.exists() and cytoplasm_file.exists():
            print(f"  ✅ Segmentation complete; three mask files saved to: {output_dir}")
            print(f"     - cyto.tif")
            print(f"     - nuclei.tif")
            print(f"     - cytoplasm.tif")
            return True
        else:
            missing = []
            if not cyto_file.exists():
                missing.append("cyto.tif")
            if not nuclei_file.exists():
                missing.append("nuclei.tif")
            if not cytoplasm_file.exists():
                missing.append("cytoplasm.tif")
            print(f"  ⚠️  Segmentation command succeeded, but files are missing: {', '.join(missing)}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Segmentation failed: {e}")
        if e.stdout:
            print(f"  stdout: {e.stdout}")
        if e.stderr:
            print(f"  stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"  ❌ Error during segmentation: {e}")
        return False


def check_segmentation_exists(sample_dir: Path) -> Optional[Path]:
    """Check whether the sample directory already has segmentation results (including any user-provided segmentation file).
    If any image file exists under segmentation/, treat it as already segmented and do not run automatic segmentation (cellpose),
    to ensure "when the user provides seg, use only the user's seg, and keep every iteration consistent".

    Args:
        sample_dir: Path to the sample directory

    Returns:
        If any segmentation file exists, return the path of the first file (sorted by filename); otherwise return None.
    """
    files = list_segmentation_files(sample_dir)
    if files:
        return files[0][1]  # (name, path) -> path
    return None


def check_all_segmentation_masks_exist(sample_dir: Path) -> bool:
    """Check whether the sample directory already has complete segmentation results (three mask files)

    Args:
        sample_dir: Path to the sample directory

    Returns:
        True if all three mask files exist; otherwise False
    """
    seg_dir = sample_dir / "segmentation"
    cyto_file = seg_dir / "cyto.tif"
    nuclei_file = seg_dir / "nuclei.tif"
    cytoplasm_file = seg_dir / "cytoplasm.tif"
    
    return cyto_file.exists() and nuclei_file.exists() and cytoplasm_file.exists()


def get_segmentation_mask_paths(sample_dir: Path) -> Dict[str, Path]:
    """Get the save paths of all segmentation masks (only the cellpose trio, for backward compatibility).

    It is recommended to use list_segmentation_files(sample_dir) to get all segmentation files in the directory (including user uploads).

    Args:
        sample_dir: Path to the sample directory

    Returns:
        A dictionary with the three mask paths: {'cyto': Path, 'nuclei': Path, 'cytoplasm': Path}
    """
    seg_dir = sample_dir / "segmentation"
    return {
        "cyto": seg_dir / "cyto.tif",
        "nuclei": seg_dir / "nuclei.tif",
        "cytoplasm": seg_dir / "cytoplasm.tif"
    }


def get_segmentation_mask_path(sample_dir: Path) -> Path:
    """Get the save path of the segmentation mask (backward compatible, returns the cyto path)

    Args:
        sample_dir: Path to the sample directory

    Returns:
        Segmentation mask file path (cyto.tif)
    """
    seg_dir = sample_dir / "segmentation"
    return seg_dir / "cyto.tif"


def load_segmentation_mask(mask_path: Path) -> Optional[Any]:
    """Load a segmentation mask (for use in code features)

    Args:
        mask_path: Mask file path

    Returns:
        The mask array, or None if loading fails
    """
    try:
        import tifffile
        mask = tifffile.imread(str(mask_path))
        return mask
    except Exception as e:
        print(f"  ⚠️  Failed to load segmentation mask: {e}")
        return None


def segment_all_samples(
    sample_ids: List[str],
    data_root: Path,
    dataset_description: Optional[str] = None,
    channels: Optional[List[int]] = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    tile_norm_blocksize: int = 0,
    batch_size: int = 32,
    conda_env: Optional[str] = None,
    skip_if_any_segmentation_exists: bool = True
) -> Dict[str, Any]:
    """Segment all samples.

    Args:
        sample_ids: List of sample IDs
        data_root: Dataset root directory
        dataset_description: Dataset description (optional, used to determine the image path)
        channels: List of channel indices to use
        flow_threshold: Flow threshold
        cellprob_threshold: Cell probability threshold
        tile_norm_blocksize: Normalization block size
        batch_size: Batch size
        conda_env: conda environment name
        skip_if_any_segmentation_exists: When True, skip a sample if any image file already exists under sample_dir/segmentation/ (the user has provided segmentation); when False, rerun Cellpose for every sample and overwrite its generated cyto/nuclei/cytoplasm trio.

    Returns:
        A dictionary where the key is sample_id and the value is "success" | "failed" | "skipped_user_seg"
    """
    from utils_helpers import find_image_paths

    conda_env = conda_env or _seg_env()
    results: Dict[str, Any] = {}

    print(f"\n[Batch Segmentation] Starting segmentation of all {len(sample_ids)} samples...")
    if skip_if_any_segmentation_exists:
        print(f"  [Policy] Skip a sample if it already has any segmentation file (user upload takes priority)")
    else:
        print("  [Policy] Regenerate Cellpose masks for every sample (generated trio will be overwritten)")

    for i, sample_id in enumerate(sample_ids, 1):
        sample_dir = data_root / sample_id
        if not sample_dir.exists():
            print(f"  [{i}/{len(sample_ids)}] ⚠️  Sample directory does not exist: {sample_dir}")
            results[sample_id] = "failed"
            continue

        # Policy A: skip if any segmentation file already exists (user provided)
        if skip_if_any_segmentation_exists:
            existing = list_segmentation_files(sample_dir)
            if existing:
                print(f"  [{i}/{len(sample_ids)}] ✅ {sample_id}: already has segmentation files ({len(existing)}), skipping")
                results[sample_id] = "skipped_user_seg"
                continue

        # Find the image file
        image_paths = find_image_paths(sample_dir, dataset_description)
        if not image_paths:
            print(f"  [{i}/{len(sample_ids)}] ⚠️  {sample_id}: no image file found")
            results[sample_id] = "failed"
            continue

        # Get the output directory
        output_dir = sample_dir / "segmentation"

        # Run segmentation
        print(f"  [{i}/{len(sample_ids)}] 🔄 {sample_id}: running segmentation...")
        success = segment_image_with_cellpose(
            input_image_path=str(image_paths[0]),
            output_mask_path=str(output_dir / "cyto.tif"),  # this argument is now ignored, since the script automatically saves three files
            channels=channels,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            tile_norm_blocksize=tile_norm_blocksize,
            batch_size=batch_size,
            conda_env=conda_env
        )

        if success:
            # Verify that all three files exist
            if check_all_segmentation_masks_exist(sample_dir):
                print(f"  [{i}/{len(sample_ids)}] ✅ {sample_id}: segmentation complete")
                results[sample_id] = "success"
            else:
                print(f"  [{i}/{len(sample_ids)}] ⚠️  {sample_id}: segmentation complete but files are incomplete")
                results[sample_id] = "failed"
        else:
            print(f"  [{i}/{len(sample_ids)}] ❌ {sample_id}: segmentation failed")
            results[sample_id] = "failed"

    success_count = sum(1 for v in results.values() if v == "success")
    skipped_count = sum(1 for v in results.values() if v == "skipped_user_seg")
    if skipped_count:
        print(f"\n[Batch Segmentation] Done: {success_count} succeeded, {skipped_count} skipped (already segmented), {len(sample_ids) - success_count - skipped_count} failed")
    else:
        print(f"\n[Batch Segmentation] Done: {success_count}/{len(sample_ids)} samples succeeded")

    return results
