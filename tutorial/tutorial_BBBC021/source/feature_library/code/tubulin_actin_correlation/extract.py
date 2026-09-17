def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    
    # Convert to appropriate array type (float32 for calculation precision)
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract relevant channels
    # Channel 0: Actin (Red)
    # Channel 1: Tubulin (Green)
    actin_channel = arr[:, :, 0]
    tubulin_channel = arr[:, :, 1]

    # Define Region of Interest (ROI) / Foreground Mask
    # The correlation should only be calculated within cellular regions to avoid 
    # the massive background (0,0) correlation which artificially inflates the score.
    
    foreground_mask = None

    # Strategy 1: Use provided segmentation masks
    if len(segmentation_masks) > 0:
        # Combine all available masks into a single boolean mask
        # Masks are typically labeled integers (0=bg, 1..N=cells)
        combined_mask = np.zeros(arr.shape[:2], dtype=bool)
        for mask in segmentation_masks:
            if mask is not None:
                # Ensure mask is 2D matching image H,W
                if mask.shape == arr.shape[:2]:
                    combined_mask = np.logical_or(combined_mask, mask > 0)
        
        if np.any(combined_mask):
            foreground_mask = combined_mask

    # Strategy 2: Fallback if no masks provided or masks were empty
    if foreground_mask is None:
        # Create a mask based on signal intensity
        # We use the maximum of Actin and Tubulin to capture the cytoskeleton structure
        signal_intensity = np.maximum(actin_channel, tubulin_channel)
        
        # Check if image is not purely black
        if np.max(signal_intensity) > 0:
            try:
                thresh = threshold_otsu(signal_intensity)
                # Ensure threshold is above noise floor (e.g., 5/255 for uint8 data)
                # Since we converted to float but didn't normalize to [0,1] yet, values are 0-255
                thresh = max(thresh, 5.0) 
                foreground_mask = signal_intensity > thresh
            except Exception:
                # Fallback for extremely low contrast images
                foreground_mask = signal_intensity > 10.0
        else:
            return 0.0

    # Extract pixels within the ROI
    # If mask is still empty or None (e.g. black image), return 0
    if foreground_mask is None or np.sum(foreground_mask) == 0:
        return 0.0

    actin_pixels = actin_channel[foreground_mask]
    tubulin_pixels = tubulin_channel[foreground_mask]

    # Check for sufficient data points
    if len(actin_pixels) < 2:
        return 0.0

    # Check for variance
    # Pearson correlation is undefined if standard deviation is 0 (constant signal)
    if np.std(actin_pixels) == 0 or np.std(tubulin_pixels) == 0:
        return 0.0

    # Compute Pearson Correlation Coefficient
    # np.corrcoef returns a matrix [[1, r], [r, 1]]
    try:
        corr_matrix = np.corrcoef(actin_pixels, tubulin_pixels)
        correlation = corr_matrix[0, 1]
        
        # Handle NaN result (can happen if numerical instability occurs)
        if np.isnan(correlation):
            return 0.0
            
        return float(correlation)
    except Exception:
        return 0.0
