def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import stats

    # 1. Input Validation and Formatting
    # Convert to float32 to avoid overflow during statistical moment calculations
    # We do not strictly need [0,1] normalization for skewness (it's scale invariant),
    # but converting to float is essential.
    arr = np.asarray(img, dtype=np.float32)

    # Check dimensionality
    # Expected shape is (H, W, 3). If it's 2D (H, W), it might be a single channel image.
    # If it's (H, W, C), we need to select the Actin channel.
    if arr.ndim == 3:
        # Dataset description: Channel 0 = Red = Actin
        if arr.shape[2] >= 1:
            actin_channel = arr[:, :, 0]
        else:
            return 0.0
    elif arr.ndim == 2:
        # Fallback: assume the single channel provided is the relevant one
        actin_channel = arr
    else:
        return 0.0

    # 2. Region of Interest Selection (Masking)
    # If segmentation masks are provided, we compute skewness only on the biological regions.
    # This prevents the massive background peak (at 0) from dominating the skewness calculation,
    # which would make the feature less sensitive to actual cytoskeletal texture.
    
    pixels_to_process = None

    if len(segmentation_masks) > 0:
        # Combine all available masks to get a "foreground" mask
        # Masks are typically labeled integers (0=bg, 1..N=cells)
        combined_mask = np.zeros(actin_channel.shape, dtype=bool)
        for mask in segmentation_masks:
            if mask is not None and mask.shape == actin_channel.shape:
                combined_mask = np.logical_or(combined_mask, mask > 0)
        
        if np.any(combined_mask):
            pixels_to_process = actin_channel[combined_mask]
    
    # Fallback: If no valid mask found or mask is empty, use the whole image
    if pixels_to_process is None or pixels_to_process.size == 0:
        pixels_to_process = actin_channel.flatten()

    # 3. Compute Feature: Skewness
    # Skewness measures the asymmetry of the probability distribution.
    # High skewness: Tail on the right (sparse bright structures on dark background).
    # Low skewness: Symmetric or tail on left (uniform coverage).
    
    # Edge case: Not enough pixels or zero variance
    if pixels_to_process.size < 3:
        return 0.0
    
    # Check for zero variance (uniform image) to avoid division by zero in skewness calc
    if np.std(pixels_to_process) == 0:
        return 0.0

    # Calculate skewness using scipy.stats.skew (Fisher-Pearson coefficient of skewness)
    # bias=False calculates the sample skewness (unbiased estimator)
    skewness_val = stats.skew(pixels_to_process, bias=False)

    # Handle NaN result (can happen if variance is extremely close to zero)
    if np.isnan(skewness_val):
        return 0.0

    return float(skewness_val)
