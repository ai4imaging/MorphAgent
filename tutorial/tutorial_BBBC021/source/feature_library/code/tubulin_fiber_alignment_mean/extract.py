def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    
    # 1. Data Loading and Preprocessing
    # Convert to float32 for precision
    arr = np.asarray(img, dtype=np.float32)
    
    # Handle dimensionality
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not standard 3-channel image, return 0.0 as fallback
        return 0.0

    # Extract Tubulin Channel (Channel 1 = Green)
    # Channel 0: Actin (Red), Channel 1: Tubulin (Green), Channel 2: DAPI (Blue)
    tubulin = arr[..., 1]

    # Normalize intensity to [0, 1]
    # The input is uint8 (0-255), so we divide by 255.0
    tubulin = tubulin / 255.0

    # 2. Define Region of Interest (ROI)
    # We need to measure alignment only where there is actual tubulin signal.
    # Background noise will produce random, meaningless coherence values.
    
    # Calculate a foreground mask based on intensity
    try:
        thresh = threshold_otsu(tubulin)
        intensity_mask = tubulin > thresh
    except Exception:
        # Fallback if image is uniform (e.g. all black)
        return 0.0

    # Incorporate segmentation masks if available
    # We prefer a cell mask (likely the first one if multiple provided, or specifically named)
    # If masks are provided, we intersect them with the intensity mask to focus on cellular tubulin
    final_mask = intensity_mask
    if segmentation_masks and len(segmentation_masks) > 0:
        # Use the first mask available (assuming it's a cell or nuclei mask)
        # If it's a nuclei mask, it might exclude the cytoplasm where tubulin is, 
        # but usually segmentation directories contain cell masks. 
        # To be safe, we take the union of all provided masks to cover the cellular area,
        # then intersect with intensity.
        seg_union = np.zeros_like(final_mask, dtype=bool)
        for mask in segmentation_masks:
            if mask is not None and mask.shape == final_mask.shape:
                seg_union = seg_union | (mask > 0)
        
        if np.any(seg_union):
            final_mask = final_mask & seg_union

    # If mask is empty, return 0.0
    if np.sum(final_mask) == 0:
        return 0.0

    # 3. Structure Tensor Calculation (Coherence)
    # We compute the local orientation coherence using the Structure Tensor (Second Moment Matrix).
    
    # Parameters
    sigma_deriv = 1.0  # Smoothing for derivative calculation (noise suppression)
    sigma_integ = 2.5  # Smoothing for tensor integration (neighborhood size)
    
    # Compute gradients (Ix, Iy)
    # Gaussian derivatives are more robust than simple finite differences
    Ix = ndimage.gaussian_filter(tubulin, sigma=sigma_deriv, order=(0, 1))
    Iy = ndimage.gaussian_filter(tubulin, sigma=sigma_deriv, order=(1, 0))

    # Compute elements of the Structure Tensor
    # J = [[Ix^2, Ix*Iy], [Ix*Iy, Iy^2]] smoothed by a Gaussian window
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    Jxx = ndimage.gaussian_filter(Ixx, sigma=sigma_integ)
    Jyy = ndimage.gaussian_filter(Iyy, sigma=sigma_integ)
    Jxy = ndimage.gaussian_filter(Ixy, sigma=sigma_integ)

    # 4. Compute Coherence
    # Coherence = (lambda1 - lambda2) / (lambda1 + lambda2)
    # This can be computed directly from tensor elements without explicit eigendecomposition:
    # Coherence = sqrt((Jxx - Jyy)^2 + 4*Jxy^2) / (Jxx + Jyy)
    
    numerator = np.sqrt((Jxx - Jyy)**2 + 4 * Jxy**2)
    denominator = Jxx + Jyy
    
    # Avoid division by zero
    epsilon = 1e-7
    coherence_map = numerator / (denominator + epsilon)

    # 5. Aggregate Feature
    # Calculate the mean coherence only within the valid ROI (cellular tubulin)
    mean_alignment = np.mean(coherence_map[final_mask])

    return float(mean_alignment)
