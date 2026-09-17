def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_erosion, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), Channel 1 is Tubulin (Green)
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Extract Channel 1 (Tubulin)
        tubulin_channel = arr[:, :, 1]
    elif arr.ndim == 2:
        # Fallback if single channel passed (unlikely based on desc, but safe)
        tubulin_channel = arr
    else:
        return 0.0

    # Intensity normalization
    # Normalize to [0, 1] for consistent Laplacian scale
    # Using robust min/max to avoid hot pixel issues
    p_min, p_max = np.percentile(tubulin_channel, (0.1, 99.9))
    if p_max > p_min:
        norm_img = (tubulin_channel - p_min) / (p_max - p_min)
    else:
        if p_max > 0:
            norm_img = tubulin_channel / p_max
        else:
            norm_img = tubulin_channel # All zeros
            
    norm_img = np.clip(norm_img, 0.0, 1.0)

    # Handle segmentation masks
    # We want to measure texture *inside* the cells, avoiding the background noise
    # and avoiding the high-contrast edge between cell and background.
    
    mask = None
    
    # 1. Try to use provided segmentation masks
    if len(segmentation_masks) > 0:
        # Combine all masks to get a general "cellular foreground" mask
        combined_mask = np.zeros(norm_img.shape, dtype=bool)
        for m in segmentation_masks:
            if m is not None:
                # Ensure mask matches image shape (handle potential squeezing issues)
                if m.shape == norm_img.shape:
                    combined_mask = np.logical_or(combined_mask, m > 0)
                elif m.ndim == 3 and m.shape[:2] == norm_img.shape:
                     # If mask is 3D (e.g. labeled volume projected), take max projection or slice
                     combined_mask = np.logical_or(combined_mask, np.max(m, axis=-1) > 0)
        
        if np.any(combined_mask):
            mask = combined_mask

    # 2. Fallback: Generate mask from Tubulin channel if no external mask provided or valid
    if mask is None:
        try:
            thresh = threshold_otsu(norm_img)
            mask = norm_img > thresh
        except Exception:
            # Fallback for extremely low signal images where otsu fails
            mask = norm_img > 0.1

    # 3. Erode the mask
    # The edge of the cell often has a very high gradient (black background to bright cell).
    # We want the texture of the *microtubules*, not the cell boundary.
    # Erode the mask to exclude the boundary pixels.
    if np.any(mask):
        # Use a small disk for erosion
        selem = disk(2)
        mask = binary_erosion(mask, footprint=selem)

    # If mask is empty after erosion (e.g. very small cells), return 0.0
    if not np.any(mask):
        return 0.0

    # Compute Laplacian
    # The Laplacian highlights regions of rapid intensity change.
    # High variance of the Laplacian indicates high spatial frequency (sharp texture).
    laplacian = ndimage.laplace(norm_img)

    # Extract values within the mask
    masked_laplacian = laplacian[mask]

    # Calculate Variance
    # Variance of the Laplacian is a standard measure for focus/texture frequency
    if masked_laplacian.size < 2:
        return 0.0
        
    spatial_frequency_metric = np.var(masked_laplacian)

    return float(spatial_frequency_metric)
