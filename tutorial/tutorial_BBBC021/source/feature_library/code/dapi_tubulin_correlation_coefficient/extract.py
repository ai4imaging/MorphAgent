def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    
    # Convert to appropriate array type (float32 for precision)
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract specific channels based on dataset description
    # Channel 1: Tubulin (Green)
    # Channel 2: DAPI (Blue)
    tubulin_ch = arr[:, :, 1]
    dapi_ch = arr[:, :, 2]

    # Define Region of Interest (ROI) - The "Cell Area"
    # We need to compute correlation only on foreground pixels to avoid 
    # the massive background (0,0) correlation dominating the statistic.
    
    mask = None
    
    # Strategy 1: Use provided segmentation masks if available
    if len(segmentation_masks) > 0:
        # Combine all masks to get a general "biological material" mask
        combined_mask = np.zeros(arr.shape[:2], dtype=bool)
        for m in segmentation_masks:
            if m is not None and m.shape == arr.shape[:2]:
                combined_mask = np.logical_or(combined_mask, m > 0)
        
        if np.any(combined_mask):
            mask = combined_mask

    # Strategy 2: Fallback to intensity-based thresholding if no masks or empty masks
    if mask is None:
        # Create a composite intensity image for thresholding (average of DAPI and Tubulin)
        # This ensures we capture both nuclear and cytoplasmic areas
        composite_intensity = (dapi_ch + tubulin_ch) / 2.0
        
        # Check if the image is not empty/black
        if np.max(composite_intensity) > 0:
            try:
                thresh = threshold_otsu(composite_intensity)
                mask = composite_intensity > thresh
            except Exception:
                # Fallback for extremely low contrast or uniform images
                mask = composite_intensity > np.mean(composite_intensity)
        else:
            return 0.0

    # Ensure we have a valid mask with enough pixels
    if mask is None or np.sum(mask) < 2:
        return 0.0

    # Extract pixel values within the ROI
    dapi_pixels = dapi_ch[mask]
    tubulin_pixels = tubulin_ch[mask]

    # Check for zero variance (e.g., flat regions) which causes division by zero in correlation
    if np.std(dapi_pixels) == 0 or np.std(tubulin_pixels) == 0:
        return 0.0

    # Compute Pearson Correlation Coefficient
    # Formula: cov(X, Y) / (std(X) * std(Y))
    # numpy.corrcoef returns the correlation matrix [[1, r], [r, 1]]
    try:
        correlation_matrix = np.corrcoef(dapi_pixels, tubulin_pixels)
        if correlation_matrix.shape == (2, 2):
            result = correlation_matrix[0, 1]
            
            # Handle NaN result (can happen if variance is effectively zero due to float precision)
            if np.isnan(result):
                return 0.0
            return float(result)
        else:
            return 0.0
    except Exception:
        return 0.0
