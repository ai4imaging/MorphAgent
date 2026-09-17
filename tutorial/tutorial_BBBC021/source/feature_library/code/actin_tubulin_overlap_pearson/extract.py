def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    
    # Convert to float32 for accurate statistical computation
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected shape: (512, 512, 3)
    # If shape is (3, 512, 512), transpose it
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[2] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    
    # Check for valid dimensions
    if arr.ndim != 3 or arr.shape[2] < 2:
        return 0.0

    # Extract relevant channels
    # Channel 0: Actin (Red)
    # Channel 1: Tubulin (Green)
    actin = arr[:, :, 0]
    tubulin = arr[:, :, 1]

    # Define Foreground Mask (ROI)
    # We must restrict calculation to cellular regions to avoid high correlation driven by background zeros.
    mask = None
    
    # 1. Try to use provided segmentation masks
    if len(segmentation_masks) > 0:
        # Combine all available masks (logical OR)
        # Masks are typically labeled integers, convert to boolean
        combined_mask = np.zeros(actin.shape, dtype=bool)
        for m in segmentation_masks:
            if m is not None:
                # Ensure mask shape matches image shape (handle potential 2D vs 3D mismatch)
                if m.shape == actin.shape:
                    combined_mask = combined_mask | (m > 0)
                elif m.ndim == 3 and m.shape[:2] == actin.shape:
                     # If mask is 3D (e.g. one-hot or RGB mask), flatten
                    combined_mask = combined_mask | (np.max(m, axis=2) > 0)
        
        if np.any(combined_mask):
            mask = combined_mask

    # 2. Fallback: If no masks provided or mask is empty, generate a simple intensity threshold mask
    if mask is None:
        # Simple background exclusion: keep pixels where either channel has signal
        # Using a low threshold (e.g., > 5% of max or absolute value > 10/255)
        # Since input is float32, we check range. If original was uint8 0-255:
        # A safe heuristic for background in fluorescence is > 0 (if background subtracted) or > small epsilon
        # Here we use a dynamic threshold based on the data to be robust
        
        # Calculate a low percentile threshold to exclude pure background
        # If the image is mostly background, mean might be low.
        thresh_actin = np.mean(actin) * 0.5
        thresh_tubulin = np.mean(tubulin) * 0.5
        
        mask = (actin > thresh_actin) | (tubulin > thresh_tubulin)

    # Flatten pixels within the mask
    # If mask is still empty (e.g. completely black image), return 0.0
    if not np.any(mask):
        return 0.0

    actin_pixels = actin[mask]
    tubulin_pixels = tubulin[mask]

    # Check for sufficient variance to compute correlation
    # If all pixels have the same value, std dev is 0 and correlation is undefined (NaN)
    if np.std(actin_pixels) == 0 or np.std(tubulin_pixels) == 0:
        return 0.0

    # Compute Pearson Correlation Coefficient
    # np.corrcoef returns a matrix [[1.0, r], [r, 1.0]]
    try:
        corr_matrix = np.corrcoef(actin_pixels, tubulin_pixels)
        pearson_r = corr_matrix[0, 1]
        
        # Handle potential NaN result from corrcoef
        if np.isnan(pearson_r):
            return 0.0
            
        return float(pearson_r)
        
    except Exception:
        return 0.0
