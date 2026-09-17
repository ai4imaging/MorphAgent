def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import stats
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type (float64 for precision in statistical moments)
    arr = np.asarray(img, dtype=np.float64)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) where Channel 0 is Actin
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Extract Actin channel (Channel 0)
        actin_channel = arr[:, :, 0]
    elif arr.ndim == 2:
        # Fallback if passed a single channel image (unlikely based on spec but safe)
        actin_channel = arr
    else:
        # Unexpected format
        return 0.0

    # Define the Region of Interest (ROI) - The biological foreground
    # We want to calculate skewness only on cellular pixels, not the black background.
    # Including the massive background peak at 0 would artificially skew the distribution.
    
    mask = None
    
    # 1. Try to use provided segmentation masks
    if len(segmentation_masks) > 0:
        # Combine all available masks to get a total cellular footprint
        combined_mask = np.zeros(actin_channel.shape, dtype=bool)
        for m in segmentation_masks:
            if m is not None:
                # Ensure mask matches image dimensions (handle potential 2D vs 3D mismatch)
                if m.shape == actin_channel.shape:
                    combined_mask = np.logical_or(combined_mask, m > 0)
                elif m.ndim == 3 and m.shape[:2] == actin_channel.shape:
                     # If mask is 3D (e.g. labeled volume), project or slice? 
                     # Usually masks for 2D images are 2D. If 3D, take max projection or slice.
                     # Given dataset is 2D, masks should be 2D.
                     combined_mask = np.logical_or(combined_mask, np.max(m, axis=-1) > 0)
        
        if np.any(combined_mask):
            mask = combined_mask

    # 2. Fallback: Generate mask from image if no valid segmentation provided
    if mask is None:
        # Use simple Otsu thresholding on the actin channel to separate cells from background
        # Check if image has content (variance > 0)
        if np.var(actin_channel) > 1e-6:
            try:
                thresh = threshold_otsu(actin_channel)
                mask = actin_channel > thresh
            except Exception:
                # Fallback for extremely low contrast/empty images
                mask = np.ones(actin_channel.shape, dtype=bool)
        else:
            # Flat image, return 0 skewness
            return 0.0

    # Extract pixels belonging to the ROI
    roi_pixels = actin_channel[mask]

    # Edge Case: No pixels selected or too few pixels for statistics
    if roi_pixels.size < 3:
        return 0.0

    # Edge Case: Constant value in ROI (variance is 0, skewness undefined)
    if np.std(roi_pixels) < 1e-6:
        return 0.0

    # Compute Skewness
    # Skewness measures the asymmetry of the probability distribution.
    # Positive skew: tail on the right (high intensity structures like stress fibers).
    # Negative skew: tail on the left.
    # bias=False calculates the sample skewness (unbiased estimator).
    skew_val = stats.skew(roi_pixels, bias=False, nan_policy='omit')

    # Handle potential NaN result from stats.skew
    if np.isnan(skew_val):
        return 0.0

    return float(skew_val)
