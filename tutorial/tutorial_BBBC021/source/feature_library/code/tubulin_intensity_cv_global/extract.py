def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu

    # 1. Input Validation and Channel Extraction
    # Ensure image is a numpy array
    img = np.asarray(img)
    
    # Check dimensionality. We expect (H, W, 3) for this dataset.
    # If 2D (H, W), assume it's a single channel image, but dataset spec says 3 channels.
    # If 3D (H, W, C), extract Tubulin (Channel 1).
    if img.ndim == 3 and img.shape[2] == 3:
        # Channel 1 is Tubulin (Green)
        tubulin_channel = img[:, :, 1]
    elif img.ndim == 2:
        # Fallback for single channel input (unlikely based on spec, but safe)
        tubulin_channel = img
    else:
        # Unexpected format
        return 0.0

    # Convert to float64 for precision in statistical calculations
    # We do NOT normalize to [0,1] here because CV is scale-invariant, 
    # but working with original intensity values is often more interpretable for debugging.
    # However, we must ensure we don't overflow.
    tubulin_float = tubulin_channel.astype(np.float64)

    # 2. Define Region of Interest (ROI) / Masking
    # We want to calculate CV only on cellular regions, not the black background.
    # Background pixels (0) would drag down the mean and inflate the std dev artificially.
    
    mask = None
    
    # Check if valid segmentation masks are provided
    if len(segmentation_masks) > 0:
        # Strategy: Use the mask that covers the most area (likely whole cell or cytoplasm)
        # to capture the microtubule network.
        # We combine all provided masks to be safe, treating any labeled region as foreground.
        combined_mask = np.zeros(tubulin_channel.shape, dtype=bool)
        for m in segmentation_masks:
            if m is not None and m.shape == tubulin_channel.shape:
                combined_mask = np.logical_or(combined_mask, m > 0)
        
        if np.any(combined_mask):
            mask = combined_mask

    # Fallback: If no masks provided or masks were empty, generate one using Otsu
    if mask is None:
        # Blur slightly to reduce noise before thresholding
        blurred = ndimage.gaussian_filter(tubulin_float, sigma=2.0)
        
        # Calculate threshold. Handle case where image is uniform (min==max)
        if np.min(blurred) == np.max(blurred):
            return 0.0
            
        try:
            thresh = threshold_otsu(blurred)
            mask = blurred > thresh
        except Exception:
            # Fallback for extremely low signal images
            mask = blurred > np.mean(blurred)

    # 3. Pixel Selection
    # Extract pixels belonging to the mask
    foreground_pixels = tubulin_float[mask]

    # 4. Compute Coefficient of Variation (CV)
    # CV = Standard Deviation / Mean
    
    # Edge case: No foreground pixels found
    if foreground_pixels.size == 0:
        return 0.0

    mean_val = np.mean(foreground_pixels)
    std_val = np.std(foreground_pixels)

    # Edge case: Mean is zero (avoid division by zero)
    if mean_val <= 1e-7:
        return 0.0

    cv = std_val / mean_val

    return float(cv)
