def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import stats

    # Convert to appropriate array type (float32 for statistical calculations)
    # We do not normalize to [0,1] because skewness is scale-invariant, 
    # and keeping original values avoids floating point precision issues near zero.
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) where Channel 0 is Actin
    if arr.ndim != 3 or arr.shape[2] < 1:
        return 0.0
    
    # Extract Actin channel (Channel 0)
    actin_channel = arr[:, :, 0]

    # Handle segmentation masks if available
    # If masks are provided, we compute skewness only on the cellular regions.
    # This prevents the large black background from dominating the distribution 
    # and artificially inflating skewness.
    pixels = None
    
    if len(segmentation_masks) > 0:
        # Combine all available masks to define the Region of Interest (ROI)
        # We assume any non-zero value in any mask indicates a cell/object
        combined_mask = np.zeros(actin_channel.shape, dtype=bool)
        for mask in segmentation_masks:
            if mask is not None and mask.shape == actin_channel.shape:
                combined_mask = np.logical_or(combined_mask, mask > 0)
        
        # If we have a valid mask with pixels
        if np.any(combined_mask):
            pixels = actin_channel[combined_mask]
    
    # Fallback: If no masks provided or mask is empty, use the whole image
    if pixels is None:
        pixels = actin_channel.flatten()

    # Edge Case: Empty data or insufficient data for skewness
    if pixels.size < 3:
        return 0.0

    # Edge Case: Zero variance (uniform image)
    # Skewness is undefined if variance is 0 (division by zero in formula)
    if np.std(pixels) == 0:
        return 0.0

    # Compute Skewness
    # Fisher-Pearson coefficient of skewness
    # bias=False calculates the sample skewness (unbiased estimator)
    skewness_val = stats.skew(pixels, bias=False)

    # Handle potential NaN returns from scipy (though std check covers most)
    if np.isnan(skewness_val):
        return 0.0

    return float(skewness_val)
