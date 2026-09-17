def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix
    from skimage.filters import threshold_otsu
    
    # 1. Input Validation and Channel Extraction
    # Convert to appropriate array type if needed, but keep as uint8 for GLCM if possible
    # Dataset spec: (512, 512, 3), uint8. Channel 1 is Tubulin (Green).
    
    img_arr = np.asarray(img)
    
    # Handle dimensionality
    if img_arr.ndim == 3 and img_arr.shape[2] == 3:
        # Standard (H, W, C) format -> Extract Green channel (Index 1)
        tubulin = img_arr[:, :, 1]
    elif img_arr.ndim == 2:
        # Fallback for single channel 2D input
        tubulin = img_arr
    else:
        # Unexpected format
        return 0.0

    # 2. ROI Generation (Masking)
    # We want to compute texture only on the cells, not the black background.
    # If we include the vast black background, the "constant-to-constant" transitions
    # will dominate the GLCM, artificially lowering entropy.
    
    mask = None
    
    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0:
        # Combine all available masks to get a "cellular region" mask
        # Masks are typically labeled integers. Convert to boolean.
        combined_mask = np.zeros(tubulin.shape, dtype=bool)
        for m in segmentation_masks:
            if m is not None and m.shape == tubulin.shape:
                combined_mask = np.logical_or(combined_mask, m > 0)
        
        if np.any(combined_mask):
            mask = combined_mask

    # Fallback: If no masks provided or mask is empty, generate one via Otsu
    if mask is None:
        # Simple background separation
        try:
            thresh = threshold_otsu(tubulin)
            mask = tubulin > thresh
        except Exception:
            # If image is uniform (e.g. all black), otsu fails
            return 0.0

    if not np.any(mask):
        return 0.0

    # 3. Quantization (Binning)
    # GLCM on 256 levels is sparse and slow. Binning to 64 levels is standard for texture.
    # We map background pixels to 0, and foreground pixels to 1-64.
    n_bins = 64
    
    # Get foreground pixels
    fg_pixels = tubulin[mask]
    
    if fg_pixels.size == 0:
        return 0.0
        
    # Normalize foreground to 0-(n_bins-1)
    p_min, p_max = fg_pixels.min(), fg_pixels.max()
    if p_max == p_min:
        # Uniform texture -> Entropy is 0
        return 0.0
        
    # Scale to 0 -> 63
    # We use float for calculation then cast to int
    scaled = (tubulin.astype(np.float32) - p_min) / (p_max - p_min) * (n_bins - 1)
    digitized = np.round(scaled).astype(np.uint8)
    
    # Shift foreground to 1-64, set background to 0
    # This creates a "Background" class at index 0
    binned_img = np.zeros_like(digitized)
    binned_img[mask] = digitized[mask] + 1
    
    # 4. GLCM Computation
    # We compute GLCM including the background class (0), then remove it.
    # Distances: 1, 3, 5 pixels (capture fine to medium texture)
    # Angles: 0, 45, 90, 135 degrees (average for rotation invariance)
    distances = [1, 3, 5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/2]
    levels = n_bins + 1 # 0 is background, 1-64 are texture levels
    
    try:
        # P shape: (levels, levels, num_distances, num_angles)
        P = graycomatrix(binned_img, distances, angles, levels=levels, symmetric=True, normed=False)
    except ValueError:
        return 0.0

    # 5. Background Suppression & Normalization
    # The entry P[0,0] is background-background transitions. We remove this.
    # We also remove transitions between background and foreground (P[0, :] and P[:, 0])
    # to focus purely on the internal texture of the tubulin network.
    
    # Slice to exclude the 0-th row and 0-th column (the background class)
    # New shape: (n_bins, n_bins, num_distances, num_angles)
    P_fg = P[1:, 1:, :, :]
    
    # Sum over distances and angles to get a single aggregate matrix (optional, but robust)
    # Alternatively, compute entropy per (d, theta) and average.
    # Here we compute entropy per (d, theta) then average.
    
    entropy_sum = 0.0
    count = 0
    
    epsilon = 1e-10
    
    for d_idx in range(len(distances)):
        for a_idx in range(len(angles)):
            # Get the matrix for this distance/angle
            p_matrix = P_fg[:, :, d_idx, a_idx]
            
            # Normalize to make it a probability distribution
            total = np.sum(p_matrix)
            if total > 0:
                p_norm = p_matrix / total
                
                # Compute Entropy: - sum(p * log2(p))
                # Mask out zeros to avoid log(0)
                mask_p = p_norm > 0
                vals = p_norm[mask_p]
                entropy = -np.sum(vals * np.log2(vals + epsilon))
                
                entropy_sum += entropy
                count += 1
    
    if count == 0:
        return 0.0
        
    result = entropy_sum / count

    return float(result)
