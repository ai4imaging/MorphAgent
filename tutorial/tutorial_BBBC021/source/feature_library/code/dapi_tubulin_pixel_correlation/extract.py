def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.filters import threshold_otsu

    # 1. Input Validation and Preparation
    # Convert to float32 to prevent overflow during statistical calculations
    arr = np.asarray(img, dtype=np.float32)

    # Check dimensionality: Expecting (H, W, 3) for BBBC021
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # 2. Channel Extraction
    # According to dataset description:
    # Channel 0: Actin (Red)
    # Channel 1: Tubulin (Green) -> Target 1
    # Channel 2: DAPI (Blue) -> Target 2
    tubulin_channel = arr[:, :, 1]
    dapi_channel = arr[:, :, 2]

    # 3. Define Region of Interest (ROI) / Foreground Mask
    # Calculating correlation on the entire image including black background (0,0)
    # artificially inflates correlation. We need to focus on cellular regions.
    
    foreground_mask = None

    # Strategy A: Use provided segmentation masks if available
    if len(segmentation_masks) > 0:
        # Combine all available masks to get a comprehensive foreground
        combined_mask = np.zeros(arr.shape[:2], dtype=bool)
        for mask in segmentation_masks:
            if mask is not None:
                # Handle potential dimension mismatch if mask is 3D
                if mask.ndim == 3:
                    mask = np.max(mask, axis=2) # Project or take first channel
                if mask.shape == arr.shape[:2]:
                    combined_mask = combined_mask | (mask > 0)
        
        if np.any(combined_mask):
            foreground_mask = combined_mask

    # Strategy B: Fallback if no masks provided or masks were empty
    if foreground_mask is None:
        # Create a simple intensity-based mask to exclude background
        # We use a low threshold or Otsu to separate signal from background noise
        try:
            # Calculate thresholds for both channels
            # Use a safe default if image is blank
            if np.max(dapi_channel) > 0:
                t_dapi = threshold_otsu(dapi_channel)
            else:
                t_dapi = 0
                
            if np.max(tubulin_channel) > 0:
                t_tub = threshold_otsu(tubulin_channel)
            else:
                t_tub = 0
            
            # Union of signals: if either channel has signal, it's foreground
            foreground_mask = (dapi_channel > t_dapi) | (tubulin_channel > t_tub)
        except Exception:
            # Fallback for extremely sparse/empty images where Otsu might fail
            foreground_mask = (dapi_channel > 5) | (tubulin_channel > 5)

    # 4. Pixel Selection
    # Flatten the arrays using the mask
    # If mask is empty (no cells), return 0.0
    if not np.any(foreground_mask):
        return 0.0

    pixels_dapi = dapi_channel[foreground_mask]
    pixels_tubulin = tubulin_channel[foreground_mask]

    # 5. Correlation Calculation
    # Check for constant input (variance = 0) which causes division by zero in correlation
    if len(pixels_dapi) < 2:
        return 0.0
        
    std_dapi = np.std(pixels_dapi)
    std_tubulin = np.std(pixels_tubulin)

    if std_dapi == 0 or std_tubulin == 0:
        return 0.0

    # Compute Pearson Correlation Coefficient
    # corr = cov(x, y) / (std(x) * std(y))
    # numpy.corrcoef returns the correlation matrix [[1, r], [r, 1]]
    try:
        correlation_matrix = np.corrcoef(pixels_dapi, pixels_tubulin)
        pearson_r = correlation_matrix[0, 1]
        
        # Handle NaN result if it occurs despite checks
        if np.isnan(pearson_r):
            return 0.0
            
        return float(pearson_r)
    except Exception:
        return 0.0
