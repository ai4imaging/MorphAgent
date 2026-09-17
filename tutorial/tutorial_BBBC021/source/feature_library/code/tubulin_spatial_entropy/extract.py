def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import stats
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type and handle dimensionality
    # Dataset is (512, 512, 3), uint8. Channel 1 is Tubulin (Green).
    arr = np.asarray(img, dtype=np.float32)
    
    # Check for valid shape
    if arr.ndim != 3 or arr.shape[2] < 2:
        return 0.0

    # Extract Tubulin channel (Index 1)
    tubulin_channel = arr[:, :, 1]

    # Define Region of Interest (ROI)
    # We want to calculate entropy only within the biological structures (cells),
    # not the large black background, which would skew the distribution.
    
    roi_mask = None
    
    # 1. Try to use provided segmentation masks
    if len(segmentation_masks) > 0:
        # Combine all available masks to get a general "foreground" mask
        # Masks are typically labeled integers (0=bg, 1..N=cells)
        combined_mask = np.zeros(tubulin_channel.shape, dtype=bool)
        for mask in segmentation_masks:
            if mask is not None and mask.shape == tubulin_channel.shape:
                combined_mask = np.logical_or(combined_mask, mask > 0)
        
        if np.any(combined_mask):
            roi_mask = combined_mask

    # 2. Fallback: If no valid masks provided, generate one via Otsu thresholding on Tubulin
    if roi_mask is None:
        # Simple background subtraction/thresholding
        # Check if image is not empty
        if np.max(tubulin_channel) > 0:
            try:
                thresh = threshold_otsu(tubulin_channel)
                roi_mask = tubulin_channel > thresh
            except Exception:
                # Fallback for extremely low contrast images
                roi_mask = tubulin_channel > np.mean(tubulin_channel)
        else:
            return 0.0

    # Extract pixels within the ROI
    # If mask is empty (e.g., blank image), return 0.0
    if not np.any(roi_mask):
        return 0.0

    roi_pixels = tubulin_channel[roi_mask]

    # Calculate Spatial Entropy
    # We treat the intensity values as a probability mass function (PMF).
    # High entropy = Uniform distribution of intensity (Diffuse / Depolymerized)
    # Low entropy = Peaked distribution of intensity (Concentrated / Spindles / Bundles)

    # 1. Normalize intensities to sum to 1 (create PMF)
    total_intensity = np.sum(roi_pixels)
    
    if total_intensity == 0:
        return 0.0
        
    probabilities = roi_pixels / total_intensity

    # 2. Calculate Shannon Entropy
    # H = -sum(p * log2(p))
    # We use base 2 for bits
    ent = stats.entropy(probabilities, base=2)

    # 3. Normalize Entropy (Optional but recommended for robustness)
    # Raw entropy increases with the number of pixels (ROI size).
    # To measure "disorder" independent of cell size, we normalize by the maximum possible entropy
    # for that number of pixels (which occurs if the distribution is perfectly uniform).
    # Max Entropy = log2(N), where N is number of pixels in ROI.
    
    n_pixels = len(roi_pixels)
    if n_pixels <= 1:
        return 0.0
        
    max_entropy = np.log2(n_pixels)
    
    # Normalized Entropy (0.0 to 1.0)
    # 1.0 = Perfectly diffuse
    # Lower values = More structure/concentration
    normalized_entropy = ent / max_entropy

    return float(normalized_entropy)
