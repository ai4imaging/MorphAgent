def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import stats
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type
    # The dataset is uint8, but we need float64 for accurate moment calculations (kurtosis)
    arr = np.asarray(img, dtype=np.float64)

    # Handle dimensionality according to dataset format
    # Expected: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] < 2:
        return 0.0

    # Extract the Tubulin channel
    # Based on dataset info: Channel 0=Actin, Channel 1=Tubulin, Channel 2=DAPI
    tubulin_channel = arr[:, :, 1]

    # Define Region of Interest (ROI)
    # Calculating kurtosis on the entire image (including black background) is often misleading
    # because the massive peak at 0 (background) dominates the distribution.
    # We aim to calculate the texture statistics of the *cellular* regions.
    
    roi_mask = None
    
    # 1. Try to use provided segmentation masks
    if len(segmentation_masks) > 0:
        # Initialize an empty boolean mask
        combined_mask = np.zeros(tubulin_channel.shape, dtype=bool)
        valid_mask_found = False
        
        for mask in segmentation_masks:
            # Ensure mask is valid and matches image dimensions (ignoring channels if mask is 2D)
            mask_arr = np.asarray(mask)
            if mask_arr.shape == tubulin_channel.shape:
                # Assume mask labels are > 0 for foreground
                combined_mask = combined_mask | (mask_arr > 0)
                valid_mask_found = True
        
        if valid_mask_found and np.any(combined_mask):
            roi_mask = combined_mask

    # 2. Fallback: Generate mask from image content if no valid external masks provided
    if roi_mask is None:
        # Check for empty/constant image to avoid threshold errors
        if np.min(tubulin_channel) == np.max(tubulin_channel):
            return 0.0
            
        try:
            # Otsu's method to separate foreground (cells) from background
            thresh = threshold_otsu(tubulin_channel)
            roi_mask = tubulin_channel > thresh
        except Exception:
            # Fallback for extremely low contrast images where Otsu might fail
            return 0.0

    # Extract pixels belonging to the ROI
    # If mask is still empty (e.g. very dark image), return 0.0
    if not np.any(roi_mask):
        return 0.0
        
    roi_pixels = tubulin_channel[roi_mask]

    # Compute Kurtosis
    # Requirements:
    # 1. Enough pixels to be statistically significant
    # 2. Non-zero variance (otherwise division by zero in kurtosis formula)
    
    if roi_pixels.size < 10:
        return 0.0
        
    pixel_std = np.std(roi_pixels)
    if pixel_std == 0:
        # If variance is 0, the distribution is a delta function.
        # Excess kurtosis for a constant value is undefined or often treated as -3.0 (limit of platykurtic)
        # However, returning 0.0 is safer for feature stability.
        return 0.0

    # Calculate Fisher's Excess Kurtosis
    # Fisher=True subtracts 3.0, so a Normal distribution = 0.0
    # High positive values indicate heavy tails/peakedness (e.g., bright microtubule bundles)
    # Low negative values indicate flat distributions (e.g., diffuse signal)
    k_val = stats.kurtosis(roi_pixels, fisher=True, bias=False)

    # Handle NaN result (can happen with very small N or specific degenerate distributions)
    if np.isnan(k_val):
        return 0.0

    return float(k_val)
