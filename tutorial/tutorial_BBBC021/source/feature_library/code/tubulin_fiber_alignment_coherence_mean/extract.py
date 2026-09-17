def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    
    # 1. Data Loading and Validation
    # Convert to float32 for processing
    arr = np.asarray(img, dtype=np.float32)
    
    # Check dimensions: Expected (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # 2. Channel Selection: Tubulin is Green (Channel 1)
    tubulin_channel = arr[:, :, 1]

    # 3. Normalization
    # Robust normalization to handle potential low contrast or outliers
    p1, p99 = np.percentile(tubulin_channel, (1, 99))
    if p99 > p1:
        tubulin_norm = (tubulin_channel - p1) / (p99 - p1)
    else:
        tubulin_norm = tubulin_channel  # Fallback if image is flat
    
    tubulin_norm = np.clip(tubulin_norm, 0.0, 1.0)

    # 4. Mask Generation (ROI Selection)
    # We want to calculate coherence only within the cell/cytoplasm area, avoiding background noise.
    mask = None
    
    # Try using provided segmentation masks first
    if len(segmentation_masks) > 0:
        # Usually masks are passed. We need a mask that covers the tubulin area.
        # If multiple masks exist, we might combine them or pick the largest coverage.
        # Assuming masks are labeled (0=bg, >0=cell).
        combined_mask = np.zeros(tubulin_channel.shape, dtype=bool)
        for m in segmentation_masks:
            if m is not None and m.shape == tubulin_channel.shape:
                combined_mask = combined_mask | (m > 0)
        
        if np.any(combined_mask):
            mask = combined_mask

    # Fallback: Otsu thresholding on the tubulin channel itself if no valid mask provided
    if mask is None or np.sum(mask) == 0:
        try:
            thresh = threshold_otsu(tubulin_norm)
            mask = tubulin_norm > thresh
        except Exception:
            # If Otsu fails (e.g., uniform image), use a simple mean threshold or process whole image
            mask = tubulin_norm > np.mean(tubulin_norm)
            
    # If mask is still empty (very rare), process the whole image
    if np.sum(mask) == 0:
        mask = np.ones(tubulin_channel.shape, dtype=bool)

    # 5. Structure Tensor Calculation
    # Compute gradients
    sigma = 1.0  # Smoothing scale for derivatives
    Iy, Ix = np.gradient(ndimage.gaussian_filter(tubulin_norm, sigma))

    # Compute structure tensor components (smoothed)
    # J = [[Ix^2, IxIy], [IxIy, Iy^2]] smoothed by a larger window
    rho = 3.0  # Integration scale (window size for averaging tensor)
    
    Ixx = ndimage.gaussian_filter(Ix**2, rho)
    Ixy = ndimage.gaussian_filter(Ix*Iy, rho)
    Iyy = ndimage.gaussian_filter(Iy**2, rho)

    # 6. Coherence Calculation
    # Eigenvalues of the structure tensor:
    # lambda1,2 = 0.5 * ( (Ixx + Iyy) +/- sqrt((Ixx - Iyy)^2 + 4*Ixy^2) )
    # Coherence = ((lambda1 - lambda2) / (lambda1 + lambda2))^2  OR  (lambda1 - lambda2) / (lambda1 + lambda2)
    # Here we use the standard definition: C = sqrt((Ixx - Iyy)^2 + 4*Ixy^2) / (Ixx + Iyy)
    # This ranges from 0 (isotropic) to 1 (perfectly oriented).
    
    numerator = np.sqrt((Ixx - Iyy)**2 + 4 * Ixy**2)
    denominator = Ixx + Iyy
    
    # Avoid division by zero
    epsilon = 1e-7
    coherence_map = numerator / (denominator + epsilon)

    # 7. Aggregation
    # Compute mean coherence only within the masked region (the cells)
    # We weight the coherence by the local intensity (denominator ~ energy) to suppress background noise further,
    # or simply take the mean over the mask.
    # Taking the mean over the mask is safer for "alignment" specifically.
    
    masked_coherence = coherence_map[mask]
    
    if masked_coherence.size == 0:
        return 0.0
        
    mean_coherence = np.mean(masked_coherence)

    return float(mean_coherence)
