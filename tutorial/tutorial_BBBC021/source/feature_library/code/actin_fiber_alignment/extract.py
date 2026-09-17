def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.feature import structure_tensor, structure_tensor_eigenvalues
    from skimage.filters import threshold_otsu
    
    # 1. Data Loading and Preprocessing
    # Convert to float32 for precision
    arr = np.asarray(img, dtype=np.float32)
    
    # Handle dimensionality
    # Expected shape: (512, 512, 3)
    # Channel 0 is Actin (Red), Channel 1 is Tubulin (Green), Channel 2 is DAPI (Blue)
    if arr.ndim == 3 and arr.shape[2] >= 1:
        # Extract Actin channel (Channel 0)
        actin_img = arr[..., 0]
    elif arr.ndim == 2:
        # Fallback if single channel passed (assume it's the relevant one)
        actin_img = arr
    else:
        return 0.0

    # Normalize intensity to [0, 1]
    # Robust normalization using percentiles to handle outliers
    p_min, p_max = np.percentile(actin_img, (1, 99))
    if p_max > p_min:
        actin_norm = (actin_img - p_min) / (p_max - p_min)
    else:
        actin_norm = actin_img  # Should be 0 or constant
    actin_norm = np.clip(actin_norm, 0.0, 1.0)

    # 2. Define Region of Interest (ROI)
    # We only want to calculate alignment where there is actual cytoskeleton signal.
    # Background noise will have random gradients that reduce the score artificially.
    
    roi_mask = None
    
    # Strategy A: Use provided segmentation masks
    if len(segmentation_masks) > 0:
        # Combine all masks to get a general foreground mask
        combined_mask = np.zeros(actin_img.shape, dtype=bool)
        for mask in segmentation_masks:
            if mask is not None and mask.shape == actin_img.shape:
                combined_mask = combined_mask | (mask > 0)
        
        if np.any(combined_mask):
            roi_mask = combined_mask

    # Strategy B: Fallback to intensity thresholding if no masks or empty masks
    if roi_mask is None:
        # Use Otsu's method to separate foreground (cells) from background
        try:
            thresh = threshold_otsu(actin_norm)
            roi_mask = actin_norm > thresh
        except Exception:
            # Fallback for extremely low signal images
            roi_mask = actin_norm > 0.1

    # If mask is still empty (e.g., blank image), return 0
    if not np.any(roi_mask):
        return 0.0

    # 3. Compute Structure Tensor
    # The structure tensor allows estimation of local orientation and coherence.
    # sigma=1.0 for inner scale (derivative smoothing)
    # sigma=3.0 for outer scale (window integration) to capture fiber bundles
    try:
        # Axx, Axy, Ayy = structure_tensor(image, sigma=1, mode='reflect')
        # But skimage's structure_tensor returns the components directly
        # We use a slightly larger sigma to aggregate over fiber bundles
        st = structure_tensor(actin_norm, sigma=1.0, mode='reflect')
        
        # Compute eigenvalues of the structure tensor
        # lambda1 >= lambda2
        # lambda1 corresponds to the gradient magnitude in the dominant direction
        # lambda2 corresponds to the gradient magnitude in the perpendicular direction
        coords = structure_tensor_eigenvalues(st)
        lambda1 = coords[0]
        lambda2 = coords[1]
        
    except Exception:
        return 0.0

    # 4. Calculate Local Coherence (Anisotropy)
    # Coherence = (lambda1 - lambda2) / (lambda1 + lambda2)
    # Ranges from 0 (isotropic/disorganized) to 1 (perfectly aligned 1D structure)
    
    denominator = lambda1 + lambda2
    numerator = lambda1 - lambda2
    
    # Avoid division by zero
    epsilon = 1e-7
    coherence_map = numerator / (denominator + epsilon)
    
    # 5. Aggregate Feature
    # We compute the weighted mean of coherence within the ROI.
    # Weighting by 'energy' (lambda1 + lambda2) ensures that bright, strong fibers 
    # contribute more to the score than faint, dim background textures.
    
    # Extract values within the mask
    mask_indices = np.where(roi_mask)
    valid_coherence = coherence_map[mask_indices]
    valid_energy = denominator[mask_indices]
    
    if valid_coherence.size == 0:
        return 0.0
        
    # Weighted average: Sum(Coherence * Energy) / Sum(Energy)
    total_energy = np.sum(valid_energy)
    
    if total_energy > epsilon:
        weighted_alignment = np.sum(valid_coherence * valid_energy) / total_energy
    else:
        # If energy is effectively zero, just take the mean
        weighted_alignment = np.mean(valid_coherence)

    return float(weighted_alignment)
