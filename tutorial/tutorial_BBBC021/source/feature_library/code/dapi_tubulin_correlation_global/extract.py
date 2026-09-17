def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract relevant channels based on BBBC021 channel map
    # Channel 1: Tubulin (Green)
    # Channel 2: DAPI (Blue)
    tubulin_channel = arr[:, :, 1]
    dapi_channel = arr[:, :, 2]

    # Define Region of Interest (ROI) to exclude background
    # Background pixels (0,0) artificially inflate correlation
    foreground_mask = None

    if len(segmentation_masks) > 0:
        # If segmentation masks are provided, combine them to define cellular regions
        # Masks are typically labeled integers (0=bg, 1..N=cells)
        combined_mask = np.zeros(arr.shape[:2], dtype=bool)
        for mask in segmentation_masks:
            if mask is not None:
                # Ensure mask is 2D
                if mask.ndim == 3:
                    mask = np.max(mask, axis=2) # Project if needed, though unlikely for masks
                if mask.shape == arr.shape[:2]:
                    combined_mask = combined_mask | (mask > 0)
        
        if np.any(combined_mask):
            foreground_mask = combined_mask

    # Fallback if no masks provided or masks were empty
    if foreground_mask is None:
        # Create a mask based on DAPI signal, as it's usually the most distinct
        # Check if image has content
        if np.max(dapi_channel) > np.min(dapi_channel):
            try:
                thresh = threshold_otsu(dapi_channel)
                # Dilate slightly to include perinuclear tubulin if needed, 
                # but simple threshold is usually sufficient for global correlation
                foreground_mask = dapi_channel > thresh
                
                # Also include significant tubulin signal to capture mitotic spindles 
                # that might extend beyond the main nuclear body defined by Otsu
                if np.max(tubulin_channel) > np.min(tubulin_channel):
                    thresh_tub = threshold_otsu(tubulin_channel)
                    foreground_mask = foreground_mask | (tubulin_channel > thresh_tub)
            except Exception:
                # Fallback for extremely low contrast images
                foreground_mask = np.ones(arr.shape[:2], dtype=bool)
        else:
            return 0.0

    # Flatten arrays based on the mask
    dapi_pixels = dapi_channel[foreground_mask]
    tubulin_pixels = tubulin_channel[foreground_mask]

    # Check for sufficient data points
    if dapi_pixels.size < 2:
        return 0.0

    # Check for zero variance (cannot compute correlation)
    if np.std(dapi_pixels) == 0 or np.std(tubulin_pixels) == 0:
        return 0.0

    # Compute Pearson Correlation Coefficient
    # Returns matrix [[1.0, r], [r, 1.0]]
    correlation_matrix = np.corrcoef(dapi_pixels, tubulin_pixels)
    
    if np.isnan(correlation_matrix[0, 1]):
        return 0.0
        
    result = correlation_matrix[0, 1]

    return float(result)
