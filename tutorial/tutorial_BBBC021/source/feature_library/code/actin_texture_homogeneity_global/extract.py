def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    from skimage.filters import threshold_otsu

    # 1. Data Loading and Validation
    # Ensure image is numpy array
    img = np.asarray(img)
    
    # Check dimensions. Expected: (512, 512, 3)
    if img.ndim != 3 or img.shape[2] != 3:
        # If not 3 channels, we can't reliably identify the Actin channel (Channel 0)
        return 0.0

    # 2. Channel Extraction
    # Dataset info: Channel 0 = Red = Actin
    actin_channel = img[:, :, 0]

    # 3. Preprocessing & Normalization
    # GLCM requires integer types. The dataset is uint8.
    # If the input is float (0-1), scale to 0-255. If uint8, use as is.
    if np.issubdtype(actin_channel.dtype, np.floating):
        # Normalize to 0-255
        min_val = np.min(actin_channel)
        max_val = np.max(actin_channel)
        if max_val > min_val:
            actin_channel = ((actin_channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            actin_channel = np.zeros_like(actin_channel, dtype=np.uint8)
    elif actin_channel.dtype != np.uint8:
        # Clip and cast if it's some other integer type
        actin_channel = np.clip(actin_channel, 0, 255).astype(np.uint8)

    # 4. Mask Generation (Region of Interest)
    # We want to measure texture *inside* the cells, not the texture of the black background.
    mask = None
    
    if len(segmentation_masks) > 0:
        # Combine all provided masks
        combined_mask = np.zeros(actin_channel.shape, dtype=bool)
        for m in segmentation_masks:
            # Ensure mask matches image spatial dimensions
            if m.shape == actin_channel.shape:
                combined_mask = np.logical_or(combined_mask, m > 0)
            elif m.ndim == 3 and m.shape[:2] == actin_channel.shape:
                 # Handle case where mask might be 3D (e.g. one-hot or RGB mask)
                 combined_mask = np.logical_or(combined_mask, np.any(m > 0, axis=2))
        
        if np.any(combined_mask):
            mask = combined_mask

    # Fallback: If no valid masks provided, generate one using Otsu thresholding on the Actin channel
    if mask is None:
        try:
            # Check if image has content
            if np.max(actin_channel) == np.min(actin_channel):
                return 0.0 # Flat image, no texture meaningful
            
            thresh = threshold_otsu(actin_channel)
            mask = actin_channel > thresh
        except Exception:
            # Fallback for extremely low contrast or empty images
            return 0.0

    # 5. Apply Mask
    # We set background pixels to 0. 
    # Note: This creates a strong texture boundary at the cell edge (cell value <-> 0).
    # However, standard practice for object-based texture often involves masking.
    # To avoid the massive "background-to-background" (0-0) co-occurrence dominating the matrix,
    # we will handle P[0,0] specifically.
    masked_actin = actin_channel.copy()
    masked_actin[~mask] = 0

    # 6. GLCM Computation
    # Parameters:
    # - distances: [1] (immediate neighbors)
    # - angles: [0, 45, 90, 135] degrees (0, pi/4, pi/2, 3pi/4) for rotational invariance
    # - levels: 256 (for uint8)
    try:
        glcm = graycomatrix(masked_actin, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                            levels=256, symmetric=True, normed=False)
    except ValueError:
        return 0.0

    # 7. Background Suppression
    # The (0,0) entry in the GLCM corresponds to background-neighboring-background.
    # Since the background is artificial (masked out), we remove its contribution 
    # so the metric reflects the texture of the cells and their boundaries.
    glcm[0, 0, :, :] = 0

    # Re-normalize the GLCM after removing the background peak
    # Sum over levels i, j to get normalization factor for each angle/distance
    glcm_sums = np.sum(glcm, axis=(0, 1))
    
    # Avoid division by zero if image was empty after masking
    # We iterate through the angles/distances to normalize
    normalized_glcm = np.zeros_like(glcm, dtype=np.float64)
    for d_idx in range(glcm.shape[2]):
        for a_idx in range(glcm.shape[3]):
            s = glcm_sums[d_idx, a_idx]
            if s > 0:
                normalized_glcm[:, :, d_idx, a_idx] = glcm[:, :, d_idx, a_idx] / s
            else:
                # If sum is 0, it means no texture (empty image), homogeneity is technically undefined or 1 (uniform)
                # We'll leave it as 0 for "no signal"
                pass

    # 8. Feature Calculation: Homogeneity
    # Formula: sum( P[i,j] / (1 + |i-j|) )
    homogeneity_values = graycoprops(normalized_glcm, 'homogeneity')
    
    # Average over all angles to get a rotation-invariant global scalar
    feature_value = np.mean(homogeneity_values)

    return float(feature_value)
