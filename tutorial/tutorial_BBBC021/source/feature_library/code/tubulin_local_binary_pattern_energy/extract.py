def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import local_binary_pattern
    from skimage.filters import threshold_otsu
    
    # Convert to appropriate array type
    # The input image is expected to be (512, 512, 3) based on dataset description
    arr = np.asarray(img)
    
    # Handle dimensionality and select Tubulin channel (Channel 1 - Green)
    # Dataset description: Channel 0=Actin, Channel 1=Tubulin, Channel 2=DAPI
    if arr.ndim == 3 and arr.shape[2] == 3:
        tubulin_channel = arr[:, :, 1]
    elif arr.ndim == 2:
        # Fallback if single channel passed (unlikely given description, but safe)
        tubulin_channel = arr
    else:
        return 0.0

    # Determine Region of Interest (ROI)
    # We want to compute texture features only within the cellular regions, not the background.
    mask = None
    
    # 1. Try using provided segmentation masks
    if len(segmentation_masks) > 0:
        # Combine all available masks to get a general foreground mask
        combined_mask = np.zeros(tubulin_channel.shape, dtype=bool)
        for m in segmentation_masks:
            if m is not None and m.shape == tubulin_channel.shape:
                combined_mask = np.logical_or(combined_mask, m > 0)
        
        if np.any(combined_mask):
            mask = combined_mask

    # 2. Fallback: Generate mask from image content if no valid masks provided
    if mask is None:
        # Simple background separation using Otsu's thresholding on the tubulin channel
        # Check if image has content (variance > 0)
        if np.std(tubulin_channel) < 1e-6:
            return 0.0 # Flat image, no texture
            
        try:
            thresh = threshold_otsu(tubulin_channel)
            mask = tubulin_channel > thresh
        except Exception:
            # Fallback for extremely sparse or uniform images where Otsu might fail
            mask = tubulin_channel > np.mean(tubulin_channel)

    # Ensure mask has pixels
    if not np.any(mask):
        return 0.0

    # LBP Parameters
    # P=8, R=1 is a standard configuration for capturing fine texture details
    # 'uniform' method reduces the feature space to 10 patterns (for P=8), making it rotation invariant
    # and robust to noise.
    P = 8
    R = 1
    method = 'uniform'
    
    # Compute LBP
    # Note: local_binary_pattern expects 2D array
    lbp = local_binary_pattern(tubulin_channel, P, R, method)
    
    # Extract LBP codes only from the ROI
    roi_lbp_codes = lbp[mask]
    
    if roi_lbp_codes.size == 0:
        return 0.0

    # Compute Histogram of LBP codes
    # For 'uniform' method with P=8, values range from 0 to P+1 (0 to 9)
    # n_bins = P + 2
    n_bins = int(roi_lbp_codes.max() + 1)
    hist, _ = np.histogram(roi_lbp_codes, bins=n_bins, density=True)
    
    # Compute Energy (Uniformity) of the histogram
    # Energy = sum(p_i^2)
    # High energy means the texture is consistent (dominated by few patterns)
    # Low energy means the texture is complex/random (flat distribution of patterns)
    energy = np.sum(hist ** 2)
    
    return float(energy)
