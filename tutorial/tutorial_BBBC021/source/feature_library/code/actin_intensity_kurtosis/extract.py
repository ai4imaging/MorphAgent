def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import stats
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_opening, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Select Actin channel (Channel 0 based on dataset description)
    actin_channel = arr[:, :, 0]

    # Normalize intensity to [0, 1]
    # Using robust min/max to handle potential outliers or noise
    p_min, p_max = np.percentile(actin_channel, (1, 99))
    if p_max > p_min:
        actin_norm = (actin_channel - p_min) / (p_max - p_min)
    else:
        if p_max > 0:
            actin_norm = actin_channel / p_max
        else:
            actin_norm = actin_channel # All zeros or constant
    
    actin_norm = np.clip(actin_norm, 0.0, 1.0)

    # Determine Region of Interest (ROI)
    # If segmentation masks are provided, try to use them to isolate cells.
    # If not, generate a mask from the image itself to exclude background.
    
    mask = None
    
    # Check if valid segmentation masks are provided
    if len(segmentation_masks) > 0:
        # Usually, masks are passed. We need a general cell mask.
        # If multiple masks exist, we might combine them or pick the most relevant (e.g., cell body).
        # Assuming the first mask might be a general segmentation or we combine all non-zero labels.
        # Let's try to combine all provided masks to get a "cellular region"
        combined_mask = np.zeros(actin_channel.shape, dtype=bool)
        for seg in segmentation_masks:
            if seg is not None and seg.shape == actin_channel.shape:
                combined_mask = np.logical_or(combined_mask, seg > 0)
        
        if np.any(combined_mask):
            mask = combined_mask

    # Fallback: Generate mask from Actin channel if no external mask is usable
    if mask is None:
        # Critic feedback suggests Otsu might be too aggressive or include background.
        # We will use a combination of a low absolute threshold (to remove pure background)
        # and Otsu's method to find the foreground, but be conservative.
        
        # 1. Simple background exclusion (very dark pixels are definitely background)
        # Using a small constant threshold like 0.1 (10% of dynamic range) as suggested.
        bg_threshold = 0.1
        mask_bg = actin_norm > bg_threshold
        
        # 2. Refine with Otsu on the potential foreground to separate signal from noise
        # Only calculate Otsu on pixels that passed the initial low threshold
        if np.any(mask_bg):
            try:
                thresh_val = threshold_otsu(actin_norm[mask_bg])
                # Use a slightly lower threshold than Otsu to ensure we capture the whole cell body,
                # not just the brightest fibers, which is important for kurtosis context.
                # However, for kurtosis of *fibers*, we want the distribution of the cell.
                # Let's stick to the suggested conservative approach:
                # Use the max of (bg_threshold, 0.5 * thresh_val) to be safe.
                final_thresh = max(bg_threshold, 0.5 * thresh_val)
                mask = actin_norm > final_thresh
            except Exception:
                mask = mask_bg
        else:
            mask = mask_bg

    # Apply morphological opening to remove small noise specks
    if mask is not None and np.any(mask):
        mask = binary_opening(mask, footprint=disk(2))

    # Extract pixels
    if mask is not None and np.any(mask):
        pixels = actin_norm[mask]
    else:
        # Fallback to whole image if mask generation fails completely (unlikely)
        pixels = actin_norm.flatten()

    # Compute Kurtosis
    # Kurtosis (Fisher's definition, normal = 0.0) measures the "tailedness".
    # High kurtosis in actin often corresponds to bright stress fibers against a darker cytoplasmic background.
    
    if len(pixels) < 10:
        return 0.0

    # Scipy's kurtosis is Fisher's (excess kurtosis), so normal distribution is 0.0.
    # Pearson's kurtosis would be 3.0 for normal.
    # We use Fisher's as it's standard in scipy.stats.kurtosis (default fisher=True).
    k = stats.kurtosis(pixels, fisher=True)
    
    # Sanity check / Clipping as per feedback
    # Extreme outliers in kurtosis calculation can happen with very sparse signals.
    # We clip the result to a reasonable range [-3, 50] to avoid exploding gradients or analysis issues.
    # A very high kurtosis (e.g. > 10) indicates extremely sparse, bright signals.
    
    # Handle NaN or Inf
    if not np.isfinite(k):
        return 0.0
        
    return float(k)
