def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    
    # Convert to appropriate array type (float32 for precision)
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract specific channels based on dataset description
    # Channel 0: Actin (Red)
    # Channel 1: Tubulin (Green)
    actin_channel = arr[:, :, 0]
    tubulin_channel = arr[:, :, 1]

    # Define the Region of Interest (ROI) mask
    # The goal is to compute correlation only within cellular regions to avoid 
    # the large background (where both are 0) artificially inflating the correlation to 1.0.
    
    mask = None
    
    # 1. Try to use provided segmentation masks
    if len(segmentation_masks) > 0:
        # Combine all available masks (logical OR) to get a general "foreground" mask
        combined_mask = np.zeros(actin_channel.shape, dtype=bool)
        for seg in segmentation_masks:
            if seg is not None and seg.shape == actin_channel.shape:
                combined_mask = combined_mask | (seg > 0)
        
        if np.any(combined_mask):
            mask = combined_mask

    # 2. Fallback: Intensity-based thresholding if no valid masks provided
    if mask is None:
        # Simple background exclusion:
        # Consider pixels where either channel has significant signal.
        # Using a low threshold (e.g., > 10 on 0-255 scale) to exclude pure background noise.
        # Since we converted to float32 but didn't normalize to [0,1] yet, values are 0-255.
        threshold = 10.0
        mask = (actin_channel > threshold) | (tubulin_channel > threshold)

    # Flatten the arrays and select only pixels within the mask
    # If mask is empty (no signal), return 0.0
    if not np.any(mask):
        return 0.0

    actin_pixels = actin_channel[mask]
    tubulin_pixels = tubulin_channel[mask]

    # Check for sufficient data points (need at least 2 for correlation)
    if len(actin_pixels) < 2:
        return 0.0

    # Check for zero variance (cannot compute correlation if one signal is constant)
    if np.std(actin_pixels) == 0 or np.std(tubulin_pixels) == 0:
        return 0.0

    # Compute Pearson Correlation Coefficient
    # np.corrcoef returns a matrix [[1, r], [r, 1]]
    try:
        corr_matrix = np.corrcoef(actin_pixels, tubulin_pixels)
        pcc = corr_matrix[0, 1]
    except Exception:
        return 0.0

    # Handle NaN result (can happen if numerical instability occurs despite variance check)
    if np.isnan(pcc):
        return 0.0

    # Clip result to valid range [-1, 1] to handle potential floating point errors
    result = np.clip(pcc, -1.0, 1.0)

    return float(result)
