def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    
    # 1. Input Validation and Preprocessing
    # Convert to float32 for precision
    arr = np.asarray(img, dtype=np.float32)
    
    # Handle dimensionality
    # Expected shape: (512, 512, 3)
    # Target channel: Channel 1 (Green) = Tubulin
    if arr.ndim == 3 and arr.shape[2] >= 2:
        tubulin = arr[..., 1]
    elif arr.ndim == 2:
        # Fallback if single channel passed (unlikely based on spec but safe)
        tubulin = arr
    else:
        return 0.0

    # Normalize intensity to [0, 1]
    # Robust normalization using percentiles to handle outliers
    p_min, p_max = np.percentile(tubulin, (1, 99))
    if p_max > p_min:
        tubulin_norm = (tubulin - p_min) / (p_max - p_min)
    else:
        if p_max > 0:
            tubulin_norm = tubulin / p_max
        else:
            tubulin_norm = tubulin # All zeros
            
    tubulin_norm = np.clip(tubulin_norm, 0.0, 1.0)

    # 2. Mask Generation (ROI Selection)
    # We only want to measure linearity inside the cells (cytoplasm), not the background.
    
    analysis_mask = None
    
    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0:
        # Combine all masks to get a general cellular foreground mask
        combined_mask = np.zeros(tubulin.shape, dtype=bool)
        for mask in segmentation_masks:
            if mask is not None:
                # Ensure mask matches image shape (handle potential 2D/3D mismatches)
                if mask.shape == tubulin.shape:
                    combined_mask = combined_mask | (mask > 0)
                elif mask.ndim == 3 and mask.shape[:2] == tubulin.shape:
                     # If mask is 3D (e.g. label matrix), project or slice
                     combined_mask = combined_mask | (mask.max(axis=2) > 0)
        
        if np.any(combined_mask):
            analysis_mask = combined_mask

    # Fallback: If no masks provided or mask is empty, generate one using Otsu
    if analysis_mask is None:
        try:
            # Smooth slightly before thresholding to get cleaner regions
            smooth_tubulin = ndimage.gaussian_filter(tubulin_norm, sigma=2.0)
            thresh = threshold_otsu(smooth_tubulin)
            analysis_mask = smooth_tubulin > thresh
        except Exception:
            # If Otsu fails (e.g., uniform image), use all pixels > epsilon
            analysis_mask = tubulin_norm > 0.05

    # If mask is still empty (e.g., black image), return 0
    if not np.any(analysis_mask):
        return 0.0

    # 3. Structure Tensor Analysis for Linearity
    # The structure tensor (J) allows us to estimate local orientation and coherence.
    # J = [[Ix^2, IxIy], [IxIy, Iy^2]] smoothed by a Gaussian window.
    
    # Parameters
    sigma_deriv = 1.0  # Scale for derivative calculation
    sigma_integ = 2.5  # Scale for integration (window size) - tuned for fiber bundles
    
    # Compute gradients
    # Use Gaussian derivatives for robustness against noise
    # ndimage.gaussian_filter1d computes derivatives if order=1
    Iy = ndimage.gaussian_filter1d(tubulin_norm, sigma=sigma_deriv, axis=0, order=1)
    Ix = ndimage.gaussian_filter1d(tubulin_norm, sigma=sigma_deriv, axis=1, order=1)
    
    # Compute tensor components
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    
    # Smooth tensor components (Integration scale)
    # This averages the orientation information over a local neighborhood
    Jxx = ndimage.gaussian_filter(Ixx, sigma=sigma_integ)
    Jyy = ndimage.gaussian_filter(Iyy, sigma=sigma_integ)
    Jxy = ndimage.gaussian_filter(Ixy, sigma=sigma_integ)
    
    # 4. Compute Coherence (Linearity)
    # Coherence = (lambda1 - lambda2) / (lambda1 + lambda2)
    # This can be computed directly from tensor components without explicit eigenvalue decomposition
    # Formula: sqrt((Jxx - Jyy)^2 + 4*Jxy^2) / (Jxx + Jyy)
    
    # Numerator: Measure of anisotropy
    diff_Jxx_Jyy = Jxx - Jyy
    numerator = np.sqrt(diff_Jxx_Jyy**2 + 4 * Jxy**2)
    
    # Denominator: Measure of total energy (trace of the matrix)
    denominator = Jxx + Jyy
    
    # Avoid division by zero
    epsilon = 1e-7
    coherence_map = numerator / (denominator + epsilon)
    
    # 5. Feature Aggregation
    # We calculate the mean coherence specifically within the cellular regions.
    # Taxane treatment (bundling) increases the linearity of the tubulin network.
    
    # Extract values within the mask
    masked_coherence = coherence_map[analysis_mask]
    
    # Optional: Weight by intensity? 
    # Bundles are often brighter. However, pure linearity is a texture feature.
    # Let's stick to the mean coherence of the foreground structure.
    
    result = np.mean(masked_coherence)
    
    return float(result)
