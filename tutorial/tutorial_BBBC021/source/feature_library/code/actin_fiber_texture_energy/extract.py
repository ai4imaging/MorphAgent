def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    from skimage.filters import threshold_otsu
    
    # 1. Input Validation and Channel Extraction
    # Ensure image is numpy array
    img = np.asarray(img)
    
    # Handle dimensionality
    # Expected shape: (H, W, 3) or (H, W) if single channel
    if img.ndim == 3:
        # Channel 0 is Actin (Red) based on dataset description
        if img.shape[2] == 3:
            actin_channel = img[:, :, 0]
        else:
            # Fallback for unexpected channel dim, take first
            actin_channel = img[:, :, 0]
    elif img.ndim == 2:
        actin_channel = img
    else:
        return 0.0

    # Ensure uint8 for GLCM calculation (standard practice for texture analysis)
    if actin_channel.dtype != np.uint8:
        # Normalize to 0-255 if not uint8
        min_val = np.min(actin_channel)
        max_val = np.max(actin_channel)
        if max_val > min_val:
            actin_channel = ((actin_channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            actin_channel = np.zeros_like(actin_channel, dtype=np.uint8)

    # 2. Mask Generation
    # We need to compute texture only within the cells, excluding the background.
    # If background is included as a large uniform area, it artificially inflates Energy/Uniformity.
    
    mask = None
    if len(segmentation_masks) > 0:
        # Combine all available masks to get the full biological foreground
        combined_mask = np.zeros(actin_channel.shape, dtype=bool)
        for m in segmentation_masks:
            # Ensure mask matches image shape (handle potential 2D vs 3D mismatch if any)
            if m.shape == actin_channel.shape:
                combined_mask = combined_mask | (m > 0)
        
        if np.any(combined_mask):
            mask = combined_mask

    # Fallback: If no masks provided or masks are empty, use Otsu thresholding on Actin
    if mask is None:
        try:
            thresh = threshold_otsu(actin_channel)
            mask = actin_channel > thresh
        except Exception:
            # Fallback for completely flat images where Otsu fails
            mask = np.ones(actin_channel.shape, dtype=bool)

    # If mask is effectively empty, return 0.0
    if np.sum(mask) < 2:
        return 0.0

    # 3. Quantization and ROI Preparation
    # GLCM on 256 levels is sparse and slow. We bin to 32 levels.
    # We use a specific strategy to ignore background:
    # - Background pixels = 0
    # - Foreground pixels = 1 to 32 (shifted by +1)
    
    n_bins = 32
    # Binning logic: value // (256/32) -> 0..31
    # We add 1 so range becomes 1..32
    binned = (actin_channel // (256 // n_bins)).astype(np.uint8) + 1
    
    # Apply mask: set background to 0
    masked_binned = binned * mask.astype(np.uint8)
    
    # 4. GLCM Computation
    # We compute GLCM for levels 0..32 (size 33x33)
    # Distances: 1 pixel
    # Angles: 0, 45, 90, 135 degrees (average for rotation invariance)
    levels = n_bins + 1
    glcm = graycomatrix(masked_binned, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                        levels=levels, symmetric=True, normed=False)
    
    # 5. ROI Extraction from GLCM
    # The GLCM includes transitions involving 0 (background).
    # We only want transitions between foreground pixels (indices 1 to 32).
    # Slice the matrix to remove row 0 and column 0.
    glcm_roi = glcm[1:, 1:, :, :]
    
    # Check if we have any valid transitions
    if np.sum(glcm_roi) == 0:
        return 0.0
    
    # 6. Feature Calculation: Energy
    # Energy = sum(p_ij^2)
    # We must normalize the ROI sub-matrix manually because 'normed=True' in graycomatrix
    # would have included the background transitions.
    
    energy_values = []
    
    # Iterate over angles and distances (shape: levels, levels, n_dists, n_angles)
    n_dists = glcm_roi.shape[2]
    n_angles = glcm_roi.shape[3]
    
    for d in range(n_dists):
        for a in range(n_angles):
            matrix = glcm_roi[:, :, d, a]
            total = np.sum(matrix)
            if total > 0:
                # Normalize to probability
                prob_matrix = matrix / total
                # Calculate Energy (Angular Second Moment)
                energy = np.sum(prob_matrix ** 2)
                energy_values.append(energy)
            else:
                energy_values.append(0.0)
    
    # Return the average energy across all directions
    result = np.mean(energy_values)
    
    return float(result)
