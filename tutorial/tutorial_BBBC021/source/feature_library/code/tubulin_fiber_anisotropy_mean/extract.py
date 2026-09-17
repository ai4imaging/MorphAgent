def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    
    # 1. Data Loading and Preprocessing
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not the expected 3-channel image, return 0.0
        return 0.0

    # Extract Tubulin channel (Channel 1 - Green)
    tubulin_img = arr[:, :, 1]

    # Normalize intensity to [0, 1]
    # Robust normalization using percentiles to handle outliers
    p_min, p_max = np.percentile(tubulin_img, (1, 99))
    if p_max > p_min:
        tubulin_norm = (tubulin_img - p_min) / (p_max - p_min)
    else:
        tubulin_norm = tubulin_img  # Fallback if image is flat
    
    tubulin_norm = np.clip(tubulin_norm, 0.0, 1.0)

    # 2. Define Region of Interest (ROI) Mask
    # We need to calculate anisotropy only where there is actual cellular material.
    # Background noise will have random anisotropy and dilute the signal.
    
    mask = None
    
    # Check if valid segmentation masks are provided
    if len(segmentation_masks) > 0:
        # Heuristic: Try to find a cell mask (usually larger) or merge all available masks
        # If multiple masks exist, we assume they might be nuclei/cells. 
        # A logical OR of all masks is a safe bet to capture biological foreground.
        combined_mask = np.zeros_like(segmentation_masks[0], dtype=bool)
        for m in segmentation_masks:
            if m.shape == tubulin_img.shape:
                combined_mask = np.logical_or(combined_mask, m > 0)
        
        if np.any(combined_mask):
            mask = combined_mask

    # Fallback: If no masks provided or masks are empty, generate a foreground mask
    if mask is None:
        try:
            # Use Otsu's method to separate foreground (cells) from background
            thresh = threshold_otsu(tubulin_norm)
            mask = tubulin_norm > thresh
        except Exception:
            # Fallback for extremely low contrast or empty images
            mask = tubulin_norm > 0.1

    # If mask is still empty (e.g., blank image), return 0.0
    if not np.any(mask):
        return 0.0

    # 3. Compute Structure Tensor
    # The structure tensor is used to estimate orientation and anisotropy.
    # It involves computing gradients, squaring them, and smoothing.
    
    # Parameters
    sigma_gradient = 1.0  # Scale for derivative calculation (often implicit in Sobel/Scharr)
    sigma_tensor = 1.5    # Scale for integration window (smoothing the tensor components)

    # Compute gradients (Iy, Ix)
    # Using Sobel filter as a robust derivative approximation
    iy = ndimage.sobel(tubulin_norm, axis=0)
    ix = ndimage.sobel(tubulin_norm, axis=1)

    # Compute tensor components per pixel
    ix2 = ix * ix
    iy2 = iy * iy
    ixy = ix * iy

    # Smooth the tensor components with a Gaussian filter
    # This integration step is crucial for the structure tensor to represent local texture
    # rather than just instantaneous edge direction.
    sx2 = ndimage.gaussian_filter(ix2, sigma=sigma_tensor)
    sy2 = ndimage.gaussian_filter(iy2, sigma=sigma_tensor)
    sxy = ndimage.gaussian_filter(ixy, sigma=sigma_tensor)

    # 4. Eigenvalue Analysis
    # The structure tensor matrix at each pixel is [[sx2, sxy], [sxy, sy2]]
    # We calculate eigenvalues L1 and L2 to determine anisotropy.
    # Trace = L1 + L2 = sx2 + sy2
    # Determinant = L1 * L2 = sx2 * sy2 - sxy * sxy
    # Difference = L1 - L2 = sqrt(Trace^2 - 4*Det) = sqrt((sx2 - sy2)^2 + 4*sxy^2)
    
    # Calculate terms for eigenvalues
    trace = sx2 + sy2
    diff_term = np.sqrt((sx2 - sy2)**2 + 4 * sxy**2)
    
    # Eigenvalues
    l1 = (trace + diff_term) / 2.0
    l2 = (trace - diff_term) / 2.0

    # 5. Compute Anisotropy
    # Metric: Coherence or Fractional Anisotropy
    # Formula: (L1 - L2) / (L1 + L2)
    # High value (near 1) -> dominant direction (fibers/bundles)
    # Low value (near 0) -> isotropic (blobs/noise)
    
    # Add epsilon to avoid division by zero
    epsilon = 1e-7
    numerator = l1 - l2
    denominator = l1 + l2 + epsilon
    
    anisotropy_map = numerator / denominator

    # 6. Aggregate Feature
    # Calculate the mean anisotropy only within the cellular mask
    masked_anisotropy = anisotropy_map[mask]
    
    # Calculate mean
    result = np.mean(masked_anisotropy)

    return float(result)
