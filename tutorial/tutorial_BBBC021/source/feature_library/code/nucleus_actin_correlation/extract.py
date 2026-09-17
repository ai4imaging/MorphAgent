def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type (float32 to prevent overflow/underflow during stats)
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), Channels: 0=Actin(R), 1=Tubulin(G), 2=DAPI(B)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract relevant channels
    # Channel 0: Actin (Cytoskeleton)
    # Channel 2: DAPI (Nucleus)
    actin_channel = arr[:, :, 0]
    dapi_channel = arr[:, :, 2]

    # Define the Region of Interest (ROI) - "within the cell area"
    # We need a binary mask to exclude the background, otherwise the large number of (0,0) pixels
    # will artificially inflate the correlation.
    
    mask = None
    
    # Priority 1: Use provided segmentation masks
    if len(segmentation_masks) > 0:
        # Combine all available masks to get the total cellular area
        combined_mask = np.zeros(arr.shape[:2], dtype=bool)
        for seg in segmentation_masks:
            if seg is not None and seg.shape == arr.shape[:2]:
                combined_mask = np.logical_or(combined_mask, seg > 0)
        
        if np.any(combined_mask):
            mask = combined_mask

    # Priority 2: Fallback to auto-segmentation if no valid masks provided
    if mask is None:
        # Use Actin channel for segmentation as it typically covers the largest cell area
        # Apply slight smoothing to reduce noise
        smooth_actin = ndimage.gaussian_filter(actin_channel, sigma=2.0)
        
        # Check if the image is not empty/black
        if np.max(smooth_actin) > np.min(smooth_actin):
            try:
                thresh = threshold_otsu(smooth_actin)
                mask = smooth_actin > thresh
            except Exception:
                # Fallback for extremely low contrast images
                mask = smooth_actin > np.mean(smooth_actin)
        else:
            # Image is flat/empty
            return 0.0

    # Flatten the arrays and select only pixels within the mask
    # This creates 1D arrays of intensities for the cellular regions
    valid_indices = np.where(mask)
    
    # If mask is empty or too small for correlation
    if len(valid_indices[0]) < 2:
        return 0.0

    actin_pixels = actin_channel[valid_indices]
    dapi_pixels = dapi_channel[valid_indices]

    # Check for variance
    # Pearson correlation is undefined if standard deviation is 0 (constant values)
    if np.std(actin_pixels) == 0 or np.std(dapi_pixels) == 0:
        return 0.0

    # Compute Pearson Correlation Coefficient
    # np.corrcoef returns the correlation matrix: [[1.0, r], [r, 1.0]]
    # We want the off-diagonal element
    correlation_matrix = np.corrcoef(actin_pixels, dapi_pixels)
    
    # Handle potential NaNs from numerical instability
    if np.isnan(correlation_matrix).any():
        return 0.0
        
    result = correlation_matrix[0, 1]

    return float(result)
