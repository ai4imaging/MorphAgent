def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) - (Height, Width, Channels)
    # Channel 0: Actin (Red)
    # Channel 1: Tubulin (Green)
    # Channel 2: DAPI (Blue)
    
    if arr.ndim != 3 or arr.shape[2] != 3:
        # Unexpected format, return 0.0
        return 0.0

    # Extract relevant channels
    actin_channel = arr[:, :, 0]
    tubulin_channel = arr[:, :, 1]

    # Define Region of Interest (ROI)
    # Correlation should be calculated on cellular regions, not background.
    # Background (0,0) pixels artificially inflate correlation.
    
    roi_mask = None

    # 1. Try to use provided segmentation masks
    if len(segmentation_masks) > 0:
        # Combine all available masks to get a general "cellular foreground" mask
        combined_mask = np.zeros(arr.shape[:2], dtype=bool)
        for mask in segmentation_masks:
            if mask is not None:
                # Ensure mask is 2D and matches image dimensions
                if mask.ndim == 2 and mask.shape == arr.shape[:2]:
                    combined_mask = np.logical_or(combined_mask, mask > 0)
                elif mask.ndim == 3 and mask.shape[:2] == arr.shape[:2]:
                     # Handle case where mask might be 3D (e.g. one-hot or labeled stack)
                     combined_mask = np.logical_or(combined_mask, np.any(mask > 0, axis=-1))
        
        if np.any(combined_mask):
            roi_mask = combined_mask

    # 2. Fallback: Create a simple intensity-based mask if no segmentation provided
    if roi_mask is None:
        # Simple thresholding to separate foreground from background
        # Assuming uint8 data range roughly 0-255, a low threshold like 10 is usually safe for fluorescence
        # We use the max of the two channels of interest to define the ROI
        intensity_sum = actin_channel + tubulin_channel
        threshold = np.percentile(intensity_sum, 20) # Dynamic threshold based on lower percentile
        # Ensure threshold is at least above absolute black noise
        threshold = max(threshold, 5.0)
        roi_mask = intensity_sum > threshold

    # Flatten arrays based on the mask
    # We only want pixels that are part of the cell(s)
    actin_pixels = actin_channel[roi_mask]
    tubulin_pixels = tubulin_channel[roi_mask]

    # Check validity
    if actin_pixels.size < 2:
        return 0.0

    # Calculate Pearson Correlation Coefficient
    # Formula: cov(x, y) / (std(x) * std(y))
    
    # Check for zero variance (constant intensity) which causes division by zero
    actin_std = np.std(actin_pixels)
    tubulin_std = np.std(tubulin_pixels)

    if actin_std == 0 or tubulin_std == 0:
        return 0.0

    # Use numpy's correlation coefficient function
    # Returns a matrix [[1.0, r], [r, 1.0]]
    correlation_matrix = np.corrcoef(actin_pixels, tubulin_pixels)
    
    if np.isnan(correlation_matrix).any():
        return 0.0
        
    result = correlation_matrix[0, 1]

    return float(result)
