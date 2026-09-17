def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import stats

    # Convert to appropriate array type
    # We use float32 to prevent overflow during statistical moment calculations
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) where Channel 1 is Tubulin (Green)
    if arr.ndim != 3 or arr.shape[2] < 2:
        # If dimensions are unexpected, return 0.0
        return 0.0

    # Extract the Tubulin channel (Channel index 1)
    tubulin_channel = arr[:, :, 1]

    # Define the region of interest (ROI)
    # We want to calculate skewness on the actual cellular signal, excluding the large black background
    # which would artificially inflate the zero-peak and distort the skewness measure.
    
    mask = None
    
    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0:
        # If masks are available, we try to find a suitable one.
        # Ideally, we want a cell mask or cytoplasm mask.
        # Since we don't know the exact order/content without metadata, we can combine them 
        # or just use the first available one as a region of interest.
        # A common strategy is to take the union of all masks to capture all cellular material.
        combined_mask = np.zeros(tubulin_channel.shape, dtype=bool)
        for m in segmentation_masks:
            if m is not None and m.shape == tubulin_channel.shape:
                combined_mask = np.logical_or(combined_mask, m > 0)
        
        if np.any(combined_mask):
            mask = combined_mask

    # Fallback if no valid mask is found: Simple intensity thresholding
    # This removes the pure background (0 or near-0 values) to focus on the signal distribution
    if mask is None:
        # Use a low threshold to separate background from foreground
        # 10.0 is a conservative threshold for uint8 data (range 0-255)
        mask = tubulin_channel > 10.0

    # Extract pixels belonging to the ROI
    valid_pixels = tubulin_channel[mask]

    # Edge Case: No valid pixels found (empty image or mask)
    if valid_pixels.size < 3:
        return 0.0

    # Edge Case: Zero variance (all pixels have the same value)
    # Skewness is undefined if variance is 0
    if np.std(valid_pixels) == 0:
        return 0.0

    # Compute Skewness
    # Fisher-Pearson coefficient of skewness
    # A high positive skewness indicates a "long tail" of high-intensity values,
    # which corresponds to the bright microtubule bundles seen in taxane-treated cells.
    # A lower skewness indicates a more uniform/diffuse distribution.
    skewness_val = stats.skew(valid_pixels, bias=False)

    # Handle NaN result from stats.skew (can happen with insufficient data points)
    if np.isnan(skewness_val):
        return 0.0

    return float(skewness_val)
