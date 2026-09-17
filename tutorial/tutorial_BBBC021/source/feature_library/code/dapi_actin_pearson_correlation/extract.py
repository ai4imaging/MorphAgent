def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    
    # Convert to appropriate array type
    # The input is expected to be (512, 512, 3) uint8 based on dataset description
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality and validate input
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If the image doesn't match the expected (H, W, C) format with 3 channels, return 0.0
        return 0.0

    # Extract relevant channels based on dataset description:
    # Channel 0: Red (Actin)
    # Channel 2: Blue (DAPI)
    actin_channel = arr[:, :, 0]
    dapi_channel = arr[:, :, 2]

    # Determine Foreground Mask
    # Calculating correlation on the black background artificially inflates the result.
    # We need to isolate the cellular regions.
    
    mask = None
    
    # 1. Try to use provided segmentation masks
    if len(segmentation_masks) > 0:
        # Combine all available masks into a single binary foreground mask
        combined_mask = np.zeros(actin_channel.shape, dtype=bool)
        for seg in segmentation_masks:
            if seg is not None and seg.shape == actin_channel.shape:
                combined_mask = combined_mask | (seg > 0)
        
        if np.any(combined_mask):
            mask = combined_mask

    # 2. Fallback: If no valid segmentation provided, create a simple intensity-based mask
    if mask is None:
        # Simple thresholding to separate foreground from background
        # We use a low threshold because we want to include the whole cell, not just bright spots
        # A value of 10/255 (approx 0.04) is usually sufficient for uint8 fluorescence data
        # Since we converted to float32 but didn't normalize to [0,1] yet (values are 0-255), use 10.0
        threshold_val = 10.0
        mask = (actin_channel > threshold_val) | (dapi_channel > threshold_val)

    # Flatten the arrays using the mask to get 1D lists of pixel intensities
    # This selects only the pixels belonging to the cells
    if not np.any(mask):
        # If mask is empty (no cells detected), return 0.0
        return 0.0

    actin_pixels = actin_channel[mask]
    dapi_pixels = dapi_channel[mask]

    # Check for sufficient data points
    if len(actin_pixels) < 2:
        return 0.0

    # Calculate Pearson Correlation
    # Formula: cov(X, Y) / (std(X) * std(Y))
    
    # Check for zero variance (constant intensity) to avoid division by zero
    actin_std = np.std(actin_pixels)
    dapi_std = np.std(dapi_pixels)

    if actin_std == 0 or dapi_std == 0:
        return 0.0

    # Use numpy's correlation coefficient function
    # Returns a matrix [[1.0, r], [r, 1.0]]
    correlation_matrix = np.corrcoef(actin_pixels, dapi_pixels)
    
    # Extract the correlation coefficient
    pearson_r = correlation_matrix[0, 1]

    # Handle potential NaN results (though checks above should prevent most)
    if np.isnan(pearson_r):
        return 0.0

    return float(pearson_r)
