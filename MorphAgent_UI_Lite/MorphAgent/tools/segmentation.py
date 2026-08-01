"""Segmentation tool - Allen / Cellpose-SAM wrappers.

Default UI backend is Allen (SEGMENTATION_BACKEND=allen). Cellpose remains available.
Supports user-uploaded segmentation files (same scan rules as data_path_selector).
"""
from typing import Optional, Any, List, Dict, Tuple
from pathlib import Path
import shutil
import subprocess
import os


def _segmentation_backend() -> str:
    """Return 'none', 'allen', or 'cellpose'."""
    try:
        from config import settings
        backend = str(getattr(settings, "segmentation_backend", "") or "").strip().lower()
        if backend:
            return backend
    except Exception:
        pass
    return os.getenv("SEGMENTATION_BACKEND", "none").strip().lower() or "none"


def _backend_is_disabled(backend: str) -> bool:
    return backend in {"none", "off", "skip", "disabled", "false", "0"}


def _seg_env() -> str:
    """Resolve the conda env used for segmentation (configurable).

    Allen must use ``morphagent_allen``; never silently reuse the agent env
    (``CONDA_ENV`` / ``morphagent``), which lacks aicsimageio / aicssegmentation.
    """
    explicit = (os.getenv("SEGMENTATION_CONDA_ENV") or "").strip()
    if explicit:
        return explicit
    try:
        from config import settings
        configured = str(getattr(settings, "segmentation_conda_env", "") or "").strip()
        if configured:
            return configured
    except Exception:
        pass
    if _segmentation_backend() == "allen":
        return "morphagent_allen"
    return os.getenv("CONDA_ENV", "morphagent")


def list_segmentation_files(sample_dir: Path) -> List[Tuple[str, Path]]:
    """Scan mask files under sample_dir/segmentation/.

    Includes TIFF/PNG/JPEG/etc. Excludes preview overlays such as
    ``segmentation_visualization.png`` that Allen/Cellpose write beside masks.
    """
    try:
        from config import settings
        image_extensions = settings.image_extensions
    except Exception:
        image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}
    try:
        from tools.image_io import is_segmentation_mask_filename
    except Exception:
        from .image_io import is_segmentation_mask_filename  # type: ignore

    seg_dir = sample_dir / "segmentation"
    out = []
    if not seg_dir.exists() or not seg_dir.is_dir():
        return out
    for seg_file in seg_dir.iterdir():
        if (
            seg_file.is_file()
            and seg_file.suffix.lower() in image_extensions
            and is_segmentation_mask_filename(seg_file.name)
        ):
            out.append((seg_file.name, seg_file))
    out.sort(key=lambda x: x[0])
    return out


def segment_image_with_allen(
    input_image_path: str,
    output_dir: str,
    channels: Optional[List[int]] = None,
    conda_env: Optional[str] = None,
) -> bool:
    """Segment an image with the Allen aicssegmentation CLI (morphagent_allen env).

    Failures return False so callers can skip the sample without aborting the run.
    """
    conda_env = conda_env or _seg_env()
    input_path = Path(input_image_path)
    out_dir = Path(output_dir)

    if not input_path.exists():
        print(f"  [WARN]  Input image does not exist: {input_path}")
        return False

    if shutil.which("conda") is None:
        print("[WARN]  conda not found; skipping Allen segmentation")
        return False

    current_file = Path(__file__).resolve()
    script_path = current_file.parent.parent / "segmentation_allen" / "run_segment_image_tif.py"
    if not script_path.exists():
        print(f"  [WARN]  Allen segmentation script does not exist: {script_path}")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "conda", "run", "-n", conda_env,
        "python", str(script_path),
        str(input_path),
        "-o", str(out_dir),
    ]
    if channels is not None and len(channels) > 0:
        cmd.extend(["-c"] + [str(c) for c in channels])

    try:
        print(f"  [Allen] Running segmentation: {input_path.name}")
        print(f"  [Allen] Using environment: {conda_env}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            print(f"  [WARN]  Allen segmentation failed (exit {result.returncode}); skipping sample")
            if result.stdout:
                print(f"  stdout: {result.stdout[-2000:]}")
            if result.stderr:
                print(f"  stderr: {result.stderr[-2000:]}")
            return False

        # Allen writes nucleus_segmentation.tiff / cytoplasm_segmentation.tiff (no preview by default).
        image_extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".gif"}
        try:
            from tools.image_io import is_segmentation_mask_filename
        except Exception:
            from .image_io import is_segmentation_mask_filename  # type: ignore
        out_files = [
            p for p in out_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in image_extensions
            and is_segmentation_mask_filename(p.name)
        ] if out_dir.is_dir() else []
        if out_files:
            print(f"  [OK] Allen segmentation complete; {len(out_files)} file(s) in {out_dir}")
            return True
        print("[WARN]  Allen command finished but no mask files were found; skipping")
        return False
    except Exception as e:
        print(f"  [WARN]  Allen segmentation error (skipping): {e}")
        return False


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
        print(f"  [WARN]  Input image does not exist: {input_path}")
        return False
    
    # Make sure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Find the segmentation script path (inside MorphAgent)
    # The script is located at tools/segment_tif_with_cpsam.py
    current_file = Path(__file__).resolve()
    script_path = current_file.parent / "segment_tif_with_cpsam.py"
    if not script_path.exists():
        print(f"  [WARN]  Segmentation script does not exist: {script_path}")
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
            print(f"  [OK] Segmentation complete; three mask files saved to: {output_dir}")
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
            print(f"  [WARN]  Segmentation command succeeded, but files are missing: {', '.join(missing)}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"  [WARN]  Segmentation failed (skipping): {e}")
        if e.stdout:
            print(f"  stdout: {e.stdout[-2000:]}")
        if e.stderr:
            print(f"  stderr: {e.stderr[-2000:]}")
        return False
    except Exception as e:
        print(f"  [WARN]  Error during segmentation (skipping): {e}")
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


def ensure_sample_segmentation(
    sample_dir: Path,
    image_path: str,
    channels: Optional[List[int]] = None,
    conda_env: Optional[str] = None,
    *,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
) -> Optional[Path]:
    """Reuse existing masks or run the configured backend (Allen / Cellpose).

    Returns a path to an existing or newly written mask file, or None on failure.
    """
    sample_dir = Path(sample_dir)
    existing = check_segmentation_exists(sample_dir)
    if existing is not None:
        return existing

    backend = _segmentation_backend()
    if _backend_is_disabled(backend):
        return None

    env = conda_env or _seg_env()
    out_dir = sample_dir / "segmentation"

    if backend == "allen":
        ok = segment_image_with_allen(
            input_image_path=str(image_path),
            output_dir=str(out_dir),
            channels=channels,
            conda_env=env,
        )
        return check_segmentation_exists(sample_dir) if ok else None

    mask_path = get_segmentation_mask_path(sample_dir)
    ok = segment_image_with_cellpose(
        input_image_path=str(image_path),
        output_mask_path=str(mask_path),
        channels=channels,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        conda_env=env,
    )
    if ok and mask_path.exists():
        return mask_path
    return None


def load_segmentation_mask(mask_path: Path) -> Optional[Any]:
    """Load a segmentation mask (TIFF/PNG/JPEG/…)."""
    try:
        from tools.image_io import load_image_array

        return load_image_array(mask_path)
    except Exception as e:
        print(f"  [WARN]  Failed to load segmentation mask: {e}")
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
    backend = _segmentation_backend()
    results: Dict[str, Any] = {}

    print(f"\n[Batch Segmentation] Starting segmentation of all {len(sample_ids)} samples...")
    print(f"  [Backend] {backend}")
    if skip_if_any_segmentation_exists:
        print(f"  [Policy] Skip a sample if it already has any segmentation file (user upload takes priority)")
    else:
        print(f"  [Policy] Regenerate masks for every sample via {backend}")

    for i, sample_id in enumerate(sample_ids, 1):
        sample_dir = data_root / sample_id
        if not sample_dir.exists():
            print(f"  [{i}/{len(sample_ids)}] [WARN]  Sample directory does not exist: {sample_dir}")
            results[sample_id] = "failed"
            continue

        # Policy A: skip if any segmentation file already exists (user provided)
        if skip_if_any_segmentation_exists:
            existing = list_segmentation_files(sample_dir)
            if existing:
                print(f"  [{i}/{len(sample_ids)}] [OK] {sample_id}: already has segmentation files ({len(existing)}), skipping")
                results[sample_id] = "skipped_user_seg"
                continue

        if _backend_is_disabled(backend):
            print(f"  [{i}/{len(sample_ids)}] [OK] {sample_id}: no masks; backend={backend} (skip auto-seg)")
            results[sample_id] = "skipped_no_backend"
            continue

        # Find the image file
        image_paths = find_image_paths(sample_dir, dataset_description)
        if not image_paths:
            print(f"  [{i}/{len(sample_ids)}] [WARN]  {sample_id}: no image file found")
            results[sample_id] = "failed"
            continue

        # Get the output directory
        output_dir = sample_dir / "segmentation"

        print(f"  [{i}/{len(sample_ids)}] [RETRY] {sample_id}: running {backend} segmentation...")
        if backend == "allen":
            success = segment_image_with_allen(
                input_image_path=str(image_paths[0]),
                output_dir=str(output_dir),
                channels=channels,
                conda_env=conda_env,
            )
            if success and list_segmentation_files(sample_dir):
                print(f"  [{i}/{len(sample_ids)}] [OK] {sample_id}: segmentation complete")
                results[sample_id] = "success"
            else:
                # Experience-first: Allen missing/failed -> skip, do not abort the run.
                print(f"  [{i}/{len(sample_ids)}] [WARN]  {sample_id}: Allen unavailable or failed; skipping sample")
                results[sample_id] = "skipped_allen_unavailable"
        else:
            success = segment_image_with_cellpose(
                input_image_path=str(image_paths[0]),
                output_mask_path=str(output_dir / "cyto.tif"),
                channels=channels,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                tile_norm_blocksize=tile_norm_blocksize,
                batch_size=batch_size,
                conda_env=conda_env,
            )
            if success:
                if check_all_segmentation_masks_exist(sample_dir):
                    print(f"  [{i}/{len(sample_ids)}] [OK] {sample_id}: segmentation complete")
                    results[sample_id] = "success"
                else:
                    print(f"  [{i}/{len(sample_ids)}] [WARN]  {sample_id}: segmentation incomplete; skipping sample")
                    results[sample_id] = "skipped_seg_unavailable"
            else:
                print(f"  [{i}/{len(sample_ids)}] [WARN]  {sample_id}: segmentation unavailable; skipping sample")
                results[sample_id] = "skipped_seg_unavailable"

    success_count = sum(1 for v in results.values() if v == "success")
    skipped_user = sum(1 for v in results.values() if v == "skipped_user_seg")
    skipped_allen = sum(1 for v in results.values() if v == "skipped_allen_unavailable")
    skipped_seg = sum(1 for v in results.values() if v == "skipped_seg_unavailable")
    failed_count = len(sample_ids) - success_count - skipped_user - skipped_allen - skipped_seg
    print(
        f"\n[Batch Segmentation] Done: {success_count} succeeded, "
        f"{skipped_user} skipped (already segmented), "
        f"{skipped_allen + skipped_seg} skipped (segmentation unavailable), "
        f"{failed_count} failed"
    )

    return results
