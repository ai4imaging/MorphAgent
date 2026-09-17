def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.filters import threshold_otsu, gaussian
    from skimage.morphology import binary_closing, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not the expected 3-channel image, return 0.0
        return 0.0

    # Extract relevant channels
    # Channel 0: Actin (Cytoskeleton)
    # Channel 1: Tubulin (Microtubules)
    # Channel 2: DAPI (Nucleus) - Not used for total area coverage in this specific feature definition
    actin = arr[..., 0]
    tubulin = arr[..., 1]

    # Check for empty image (all zeros)
    if np.max(actin) == 0 and np.max(tubulin) == 0:
        return 0.0

    # Normalize to [0, 1] for processing
    # We normalize each channel independently to handle varying exposure times
    def normalize_channel(ch):
        vmax = np.percentile(ch, 99.5) if ch.size > 0 else 1.0
        if vmax <= 0:
            return ch  # Avoid division by zero
        return np.clip(ch / vmax, 0.0, 1.0)

    actin_norm = normalize_channel(actin)
    tubulin_norm = normalize_channel(tubulin)

    # Preprocessing: Gaussian blur to reduce noise before thresholding
    # Sigma=2.0 is chosen to smooth out pixel noise while preserving cell boundaries
    actin_blur = gaussian(actin_norm, sigma=2.0)
    tubulin_blur = gaussian(tubulin_norm, sigma=2.0)

    # Thresholding
    # Use Otsu's method to find an optimal threshold for separating foreground (cells) from background
    # We wrap this in a try-except block because threshold_otsu raises an error on constant images
    try:
        thresh_actin = threshold_otsu(actin_blur)
        # Sanity check: if threshold is extremely low, it might be thresholding noise in an empty image
        # 0.05 is a heuristic lower bound for normalized fluorescence signal
        if thresh_actin < 0.02: 
            mask_actin = np.zeros_like(actin_blur, dtype=bool)
        else:
            mask_actin = actin_blur > thresh_actin
    except Exception:
        mask_actin = np.zeros_like(actin_blur, dtype=bool)

    try:
        thresh_tubulin = threshold_otsu(tubulin_blur)
        if thresh_tubulin < 0.02:
            mask_tubulin = np.zeros_like(tubulin_blur, dtype=bool)
        else:
            mask_tubulin = tubulin_blur > thresh_tubulin
    except Exception:
        mask_tubulin = np.zeros_like(tubulin_blur, dtype=bool)

    # Morphological Refinement
    # Close small holes inside the cell bodies (e.g. less stained nuclear regions in cytoskeletal channels)
    # Disk size 2 is small enough to not merge distinct cells too aggressively but fill gaps
    selem = disk(2)
    mask_actin = binary_closing(mask_actin, footprint=selem)
    mask_tubulin = binary_closing(mask_tubulin, footprint=selem)

    # Combine masks
    # The confluence is defined by the union of Actin and Tubulin areas
    total_mask = np.logical_or(mask_actin, mask_tubulin)

    # Compute Ratio
    foreground_pixels = np.sum(total_mask)
    total_pixels = total_mask.size

    if total_pixels == 0:
        return 0.0

    confluence_ratio = foreground_pixels / total_pixels

    return float(confluence_ratio)
