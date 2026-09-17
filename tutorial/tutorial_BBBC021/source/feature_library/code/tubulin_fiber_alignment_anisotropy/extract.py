def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import structure_tensor, structure_tensor_eigenvalues
    from skimage.filters import threshold_otsu, gaussian
    
    # Convert to float32 for processing
    arr = np.asarray(img, dtype=np.float32)
    
    # 1. Handle Dimensionality and Channel Selection
    # Dataset is (512, 512, 3) where Channel 1 is Tubulin (Green)
    # If the input is 2D (H, W), assume it's a single channel image (unlikely given description, but safe fallback)
    # If 3D (H, W, C), extract channel 1.
    
    tubulin_img = None
    
    if arr.ndim == 3:
        if arr.shape[2] >= 2:
            tubulin_img = arr[:, :, 1]  # Channel 1 is Tubulin
        else:
            tubulin_img = arr[:, :, 0]  # Fallback if fewer channels
    elif arr.ndim == 2:
        tubulin_img = arr
    else:
        return 0.0

    # 2. Intensity Normalization
    # Normalize to [0, 1] for stable gradient calculation
    if tubulin_img.size == 0:
        return 0.0
        
    v_min, v_max = np.min(tubulin_img), np.max(tubulin_img)
    if v_max - v_min > 1e-6:
        tubulin_img = (tubulin_img - v_min) / (v_max - v_min)
    else:
        tubulin_img = np.zeros_like(tubulin_img)

    # 3. Define Region of Interest (ROI)
    # We only want to measure anisotropy inside the cells/cytoplasm, not the background.
    # Background noise often has random high-frequency gradients that can skew anisotropy measures.
    
    roi_mask = None
    
    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0:
        # Combine all masks to get a general cellular foreground mask
        combined_mask = np.zeros(tubulin_img.shape, dtype=bool)
        for mask in segmentation_masks:
            if mask is not None and mask.shape == tubulin_img.shape:
                combined_mask = np.logical_or(combined_mask, mask > 0)
        
        if np.any(combined_mask):
            roi_mask = combined_mask

    # Fallback: If no masks provided or masks are empty, create an intensity-based mask
    if roi_mask is None:
        # Smooth slightly to get a better threshold
        smooth_img = gaussian(tubulin_img, sigma=2)
        try:
            thresh = threshold_otsu(smooth_img)
            roi_mask = smooth_img > thresh
        except Exception:
            # Fallback for extremely low contrast images
            roi_mask = tubulin_img > np.mean(tubulin_img)

    # If ROI is still empty (e.g., black image), return 0
    if not np.any(roi_mask):
        return 0.0

    # 4. Compute Structure Tensor
    # The structure tensor is ideal for analyzing texture orientation and anisotropy.
    # sigma_grad: scale for derivative calculation (detects edges/fibers)
    # sigma_block: scale for integration (averaging tensor over a local window)
    # For microtubules, we want to detect fine fibers (sigma_grad ~ 1) and check local alignment (sigma_block ~ 3-5)
    
    try:
        # Axx, Axy, Ayy
        st = structure_tensor(tubulin_img, sigma=1.0, mode='reflect')
        
        # Compute eigenvalues: l1 >= l2 >= 0
        # l1 corresponds to the gradient magnitude in the dominant direction
        # l2 corresponds to the gradient magnitude in the perpendicular direction
        l1, l2 = structure_tensor_eigenvalues(st)
    except Exception:
        return 0.0

    # 5. Calculate Anisotropy (Coherence)
    # Anisotropy = (l1 - l2) / (l1 + l2)
    # 0 = Isotropic (circle/noise), 1 = Anisotropic (line/bundle)
    # Add epsilon to avoid division by zero
    epsilon = 1e-7
    denominator = l1 + l2 + epsilon
    anisotropy_map = (l1 - l2) / denominator

    # 6. Aggregation
    # We compute the weighted average of anisotropy within the ROI.
    # Weighting by the original intensity (or l1+l2, which is "energy") is often robust 
    # because it emphasizes actual structures over dim background fluctuations within the mask.
    
    # Mask the maps
    masked_anisotropy = anisotropy_map[roi_mask]
    masked_weights = tubulin_img[roi_mask] # Weight by intensity
    
    # Ensure we have valid weights
    if np.sum(masked_weights) < epsilon:
        return 0.0
        
    # Weighted mean
    weighted_anisotropy = np.sum(masked_anisotropy * masked_weights) / np.sum(masked_weights)
    
    return float(weighted_anisotropy)
