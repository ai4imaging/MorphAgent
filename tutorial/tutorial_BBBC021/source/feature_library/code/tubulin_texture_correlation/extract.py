def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    from skimage.filters import threshold_otsu
    
    # 1. Input Validation and Channel Extraction
    # Ensure input is numpy array
    img = np.asarray(img)
    
    # Handle dimensionality
    # Expected shape: (512, 512, 3) for RGB TIFF
    # Channel 1 is Tubulin (Green)
    if img.ndim == 3 and img.shape[2] >= 2:
        tubulin_channel = img[:, :, 1]
    elif img.ndim == 2:
        # Fallback if single channel image is passed (unlikely based on spec but safe)
        tubulin_channel = img
    else:
        return 0.0

    # Ensure uint8 for GLCM
    if tubulin_channel.dtype != np.uint8:
        # Normalize to 0-255 if not already
        if tubulin_channel.max() <= 1.0:
            tubulin_channel = (tubulin_channel * 255).astype(np.uint8)
        else:
            # Clip and cast
            tubulin_channel = np.clip(tubulin_channel, 0, 255).astype(np.uint8)

    # 2. Masking / Region of Interest (ROI) Definition
    # We want to calculate texture only on the biological structures, not the black background.
    # The background (0,0) co-occurrences dominate GLCM in fluorescence images if not handled.
    
    mask = None
    if len(segmentation_masks) > 0:
        # Combine all available masks to define the cellular region
        # Assuming masks are labeled arrays (0=bg, >0=cells)
        combined_mask = np.zeros(tubulin_channel.shape, dtype=bool)
        for m in segmentation_masks:
            if m is not None and m.shape == tubulin_channel.shape:
                combined_mask = np.logical_or(combined_mask, m > 0)
        
        if np.any(combined_mask):
            mask = combined_mask
    
    # Fallback: If no masks provided or masks are empty, use intensity thresholding
    if mask is None:
        # Simple background separation
        try:
            thresh = threshold_otsu(tubulin_channel)
            mask = tubulin_channel > thresh
        except Exception:
            # If image is uniform, otsu fails
            mask = np.ones(tubulin_channel.shape, dtype=bool)

    # Apply mask: Set background pixels to 0
    # We will handle the "0-0" background correlation in the GLCM step
    masked_img = tubulin_channel.copy()
    masked_img[~mask] = 0

    # 3. Compute GLCM (Gray Level Co-occurrence Matrix)
    # Parameters:
    # - distances=[1]: analyze immediate neighbors for fine texture
    # - angles=[0, 45, 90, 135]: rotational invariance
    # - levels=256: for uint8
    try:
        glcm = graycomatrix(masked_img, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                            levels=256, symmetric=True, normed=True)
    except ValueError:
        return 0.0

    # 4. Background Correction (CRITICAL)
    # In the masked image, the background is 0. The GLCM entry (0,0) represents 
    # background-background transitions, which are artificially high and perfect correlated.
    # We remove this specific entry to focus on foreground texture.
    
    # Set (0,0) to 0 for all distances and angles
    glcm[0, 0, :, :] = 0
    
    # Re-normalize the GLCM so probabilities sum to 1
    # Iterate over distances and angles
    for d_idx in range(glcm.shape[2]):
        for a_idx in range(glcm.shape[3]):
            s = np.sum(glcm[:, :, d_idx, a_idx])
            if s > 0:
                glcm[:, :, d_idx, a_idx] /= s
            else:
                # If sum is 0 (e.g. image was all background), this slice is empty
                pass

    # 5. Compute Feature: Correlation
    # Correlation measures linear dependency of gray levels.
    # High correlation -> structured, bundled microtubules
    # Low correlation -> diffuse, disordered signal
    try:
        correlation_values = graycoprops(glcm, 'correlation')
    except Exception:
        return 0.0

    # 6. Aggregation
    # Average over the 4 angles to get a rotation-invariant scalar
    result = np.mean(correlation_values)
    
    # Handle NaN (can happen if variance is 0, e.g., flat constant region)
    if np.isnan(result):
        return 0.0

    return float(result)
