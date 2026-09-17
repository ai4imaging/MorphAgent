def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.filters import threshold_otsu
    
    # Convert to appropriate array type
    # The input is (512, 512, 3), uint8. We convert to float64 for statistical precision.
    arr = np.asarray(img, dtype=np.float64)

    # Handle dimensionality according to dataset format
    # Dataset is (Height, Width, Channels) = (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If unexpected dimensions, return 0.0
        return 0.0

    # Extract the Tubulin channel (Channel 1: Green)
    # Channel 0: Actin (Red), Channel 1: Tubulin (Green), Channel 2: DAPI (Blue)
    tubulin_channel = arr[:, :, 1]

    # Define Region of Interest (ROI) - The Cell Body
    # We need to calculate statistics only on the cells, not the background.
    
    roi_mask = None

    # Strategy 1: Use provided segmentation masks
    if len(segmentation_masks) > 0:
        # If masks are provided, we combine them to form a global foreground mask.
        # We assume masks are labeled (0=bg, >0=cells).
        # We iterate through available masks to find a suitable one.
        # Often the last mask in a sequence might be the most comprehensive (e.g., cytoplasm),
        # but we will union all non-zero pixels from all masks to be safe.
        combined_mask = np.zeros(tubulin_channel.shape, dtype=bool)
        has_valid_mask = False
        
        for mask in segmentation_masks:
            if mask is not None and mask.shape == tubulin_channel.shape:
                combined_mask = np.logical_or(combined_mask, mask > 0)
                has_valid_mask = True
        
        if has_valid_mask:
            roi_mask = combined_mask

    # Strategy 2: Fallback to intensity-based thresholding if no valid mask found
    if roi_mask is None or np.sum(roi_mask) == 0:
        # Use Otsu's thresholding on the tubulin channel itself to separate foreground
        try:
            # Check if the image has any variation
            if np.min(tubulin_channel) == np.max(tubulin_channel):
                return 0.0
            
            thresh = threshold_otsu(tubulin_channel)
            roi_mask = tubulin_channel > thresh
        except Exception:
            # Fallback for extremely low signal images or errors
            # Use a simple mean threshold
            roi_mask = tubulin_channel > np.mean(tubulin_channel)

    # Extract pixels belonging to the ROI
    # Flatten the array and select only the masked pixels
    foreground_pixels = tubulin_channel[roi_mask]

    # Calculate Coefficient of Variation (CV)
    # CV = Standard Deviation / Mean
    
    if foreground_pixels.size == 0:
        return 0.0

    mean_val = np.mean(foreground_pixels)
    std_val = np.std(foreground_pixels)

    # Handle division by zero if mean is 0 (e.g., black image)
    if mean_val <= 1e-9:
        return 0.0

    cv = std_val / mean_val

    return float(cv)
