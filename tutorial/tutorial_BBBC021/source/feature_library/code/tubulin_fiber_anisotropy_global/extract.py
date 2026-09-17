def extract(img, *segmentation_masks):
    """
    Computes the global anisotropy of the Tubulin channel using structure tensors.
    High anisotropy indicates aligned microtubule bundles (e.g., Taxane effect),
    while low anisotropy suggests a disorganized or depolymerized network.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image. Expected shape (512, 512, 3) or (3, 512, 512).
        Channel 1 (Green) is assumed to be Tubulin.
    segmentation_masks : tuple
        Optional segmentation masks.
        
    Returns:
    --------
    float
        The mean anisotropy value [0, 1].
    """
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu

    # 1. Input Validation and Normalization
    img_arr = np.asarray(img, dtype=np.float32)

    # Handle dimensions: Ensure (H, W, C) or (C, H, W) -> Extract Tubulin channel
    # Dataset info says (512, 512, 3) RGB. Tubulin is Green (Channel 1).
    if img_arr.ndim == 3:
        if img_arr.shape[2] == 3:  # (H, W, C)
            tubulin = img_arr[..., 1]
        elif img_arr.shape[0] == 3:  # (C, H, W)
            tubulin = img_arr[1, ...]
        else:
            return 0.0
    elif img_arr.ndim == 2:
        # Fallback if only 2D image provided (unlikely given dataset desc, but safe)
        tubulin = img_arr
    else:
        return 0.0

    # Normalize intensity to [0, 1]
    # Robust normalization using percentiles to handle outliers
    p_min, p_max = np.percentile(tubulin, (1, 99))
    if p_max > p_min:
        tubulin = (tubulin - p_min) / (p_max - p_min)
    else:
        if p_max > 0:
            tubulin = tubulin / p_max
    tubulin = np.clip(tubulin, 0.0, 1.0)

    # 2. Define Region of Interest (ROI)
    # Use segmentation masks if available, otherwise create a foreground mask
    mask = None
    
    # Check provided masks
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Usually mask 0 might be cells or nuclei. We prefer a cell mask if available.
        # If multiple masks, we combine them or pick the largest coverage.
        # Here we just take the first valid one.
        candidate_mask = segmentation_masks[0]
        if candidate_mask.shape == tubulin.shape:
            mask = candidate_mask > 0
    
    # Fallback: Otsu thresholding on Tubulin channel if no mask provided or mask is empty
    if mask is None or np.sum(mask) == 0:
        try:
            thresh = threshold_otsu(tubulin)
            mask = tubulin > thresh
        except Exception:
            # If Otsu fails (e.g. constant image), use the whole image
            mask = np.ones_like(tubulin, dtype=bool)

    # Final check: if mask is still empty (very rare), use whole image
    if np.sum(mask) == 0:
        mask = np.ones_like(tubulin, dtype=bool)

    # 3. Compute Structure Tensor
    # Gradients
    sigma = 1.0  # Scale for derivative calculation
    Iy, Ix = np.gradient(tubulin)
    
    # Structure tensor components: J = [[Ix^2, IxIy], [IxIy, Iy^2]]
    # Smoothed by a larger sigma (integration scale)
    rho = 2.0
    Ixx = ndimage.gaussian_filter(Ix**2, sigma=rho)
    Ixy = ndimage.gaussian_filter(Ix*Iy, sigma=rho)
    Iyy = ndimage.gaussian_filter(Iy**2, sigma=rho)

    # 4. Compute Eigenvalues and Anisotropy
    # The eigenvalues of the 2x2 matrix [[Ixx, Ixy], [Ixy, Iyy]] are:
    # lambda1,2 = 0.5 * ( (Ixx + Iyy) +/- sqrt((Ixx - Iyy)^2 + 4*Ixy^2) )
    
    trace = Ixx + Iyy
    det = Ixx * Iyy - Ixy**2
    
    # Discriminant for eigenvalue calculation
    # sqrt_delta = sqrt((Ixx - Iyy)^2 + 4*Ixy^2)
    # This is equivalent to sqrt(trace^2 - 4*det)
    # But (Ixx-Iyy)^2 + 4Ixy^2 is numerically safer as it's sum of squares
    delta = (Ixx - Iyy)**2 + 4 * (Ixy**2)
    sqrt_delta = np.sqrt(delta)
    
    l1 = 0.5 * (trace + sqrt_delta)
    l2 = 0.5 * (trace - sqrt_delta)
    
    # Coherence / Anisotropy = (l1 - l2) / (l1 + l2)
    # l1 >= l2 >= 0. 
    # If l1 + l2 == 0 (flat region), anisotropy is 0.
    
    denominator = l1 + l2
    numerator = l1 - l2
    
    # Avoid division by zero
    epsilon = 1e-7
    anisotropy_map = numerator / (denominator + epsilon)
    
    # 5. Aggregate within Mask
    # We only care about anisotropy where there is actually signal (the mask)
    masked_anisotropy = anisotropy_map[mask]
    
    if masked_anisotropy.size == 0:
        return 0.0
        
    # Return mean anisotropy
    score = np.mean(masked_anisotropy)
    
    return float(score)
