def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    
    # Convert to appropriate array type (float32 for precision in correlation calc)
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), Channels: 0=Actin, 1=Tubulin, 2=DAPI
    if arr.ndim != 3 or arr.shape[2] < 2:
        # If image doesn't have at least 2 channels or is not 3D, we can't compute correlation
        return 0.0

    # Extract relevant channels
    # Channel 0: Actin (Red)
    # Channel 1: Tubulin (Green)
    actin = arr[:, :, 0]
    tubulin = arr[:, :, 1]

    # Define Region of Interest (ROI)
    # Correlation should be calculated on cellular regions to avoid background inflation.
    # If we include the large black background (0,0), correlation artificially skews to 1.0.
    
    roi_mask = None
    
    # 1. Try to use provided segmentation masks
    if len(segmentation_masks) > 0:
        # Combine all available masks
        combined_mask = np.zeros(actin.shape, dtype=bool)
        for mask in segmentation_masks:
            if mask is not None:
                # Ensure mask is 2D and matches image spatial dims
                if mask.ndim == 2 and mask.shape == actin.shape:
                    combined_mask = combined_mask | (mask > 0)
                elif mask.ndim == 3 and mask.shape[:2] == actin.shape:
                    # Handle case where mask might be 3D (e.g. one-hot encoded or RGB mask)
                    # Flatten channel dim if present
                    combined_mask = combined_mask | (np.any(mask > 0, axis=-1))
        
        if np.any(combined_mask):
            roi_mask = combined_mask

    # 2. Fallback: Intensity-based thresholding if no valid masks provided
    if roi_mask is None:
        # Simple background exclusion
        # Calculate a low threshold to exclude pure background noise
        # For uint8 data (0-255), a value of ~10-15 is usually safe for background
        # Here we use a dynamic threshold based on the data
        threshold_actin = np.percentile(actin, 20) # Estimate background level
        threshold_tubulin = np.percentile(tubulin, 20)
        
        # Create mask where either channel has significant signal
        # We use a small epsilon or fixed low value to ensure we don't just select 0
        bg_thresh = 5.0 
        roi_mask = (actin > max(threshold_actin, bg_thresh)) | (tubulin > max(threshold_tubulin, bg_thresh))

    # Flatten arrays based on the mask
    # If mask is empty (no cells), return 0.0
    if not np.any(roi_mask):
        return 0.0

    # Select pixels within the ROI
    actin_pixels = actin[roi_mask]
    tubulin_pixels = tubulin[roi_mask]

    # Check for sufficient data points
    if len(actin_pixels) < 2:
        return 0.0

    # Check for zero variance (constant signal)
    # Pearson correlation is undefined if standard deviation is zero
    if np.std(actin_pixels) == 0 or np.std(tubulin_pixels) == 0:
        return 0.0

    # Calculate Pearson Correlation Coefficient
    # Returns matrix [[1.0, r], [r, 1.0]]
    corr_matrix = np.corrcoef(actin_pixels, tubulin_pixels)
    
    # Extract the correlation value
    r = corr_matrix[0, 1]

    # Handle NaN results (can happen if inputs are effectively constant due to precision)
    if np.isnan(r):
        return 0.0

    # Clip result to valid range [-1, 1] to handle floating point epsilon errors
    result = np.clip(r, -1.0, 1.0)

    return float(result)
