def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    
    # --- 1. Data Loading and Preprocessing ---
    # Convert to float32 for precision
    arr = np.asarray(img, dtype=np.float32)
    
    # Check dimensions: Expected (H, W, 3) for BBBC021
    if arr.ndim != 3 or arr.shape[2] != 3:
        # Fallback for potential unexpected shapes (e.g., if channels are first)
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        elif arr.ndim == 2:
            # If 2D, assume it's a single channel image, but we can't be sure it's tubulin.
            # However, standard format is (H, W, 3). Return 0.0 if strictly not matching.
            return 0.0
        else:
            return 0.0

    # Extract Tubulin Channel (Channel 1 - Green)
    tubulin = arr[..., 1]

    # Normalize intensity to [0, 1]
    # Robust normalization using percentiles to handle outliers
    p_min, p_max = np.percentile(tubulin, (1, 99))
    if p_max > p_min:
        tubulin = (tubulin - p_min) / (p_max - p_min)
    else:
        if p_max > 0:
            tubulin = tubulin / p_max
    tubulin = np.clip(tubulin, 0.0, 1.0)

    # --- 2. Mask Generation (ROI) ---
    # We need to compute anisotropy only on the cellular structures, not the background.
    # Background noise has random gradients which would lower the average anisotropy score.
    
    mask = None
    
    # Try to use provided segmentation masks
    if len(segmentation_masks) > 0:
        # Check for a valid mask. Often masks are passed as (nuclei, cells) or just one.
        # We prefer a cell mask (larger area) over a nuclei mask.
        # Heuristic: Use the mask with the largest foreground area, assuming it covers the cytoplasm.
        best_mask_area = 0
        for m in segmentation_masks:
            if m is not None and m.shape == tubulin.shape:
                current_area = np.count_nonzero(m)
                if current_area > best_mask_area:
                    mask = m > 0
                    best_mask_area = current_area
    
    # Fallback: Generate mask from the tubulin channel itself if no valid mask provided
    if mask is None or np.count_nonzero(mask) == 0:
        try:
            thresh = threshold_otsu(tubulin)
            mask = tubulin > thresh
            # Fill holes to ensure we capture the internal fiber network
            mask = ndimage.binary_fill_holes(mask)
        except Exception:
            # If otsu fails (e.g. uniform image), use a simple mean threshold
            mask = tubulin > np.mean(tubulin)

    # Ensure mask is boolean
    mask = mask.astype(bool)
    
    # If mask is empty (no cells), return 0.0
    if np.count_nonzero(mask) == 0:
        return 0.0

    # --- 3. Structure Tensor Calculation ---
    # The structure tensor allows us to estimate orientation and anisotropy.
    # J = [[Ix^2, IxIy], [IxIy, Iy^2]] smoothed by a Gaussian.
    
    # Parameters
    sigma_grad = 1.0  # Smoothing before gradient calculation to reduce noise
    sigma_tensor = 2.0 # Integration scale (neighborhood size) for the tensor
    
    # Compute gradients
    # Use Gaussian derivatives for robustness
    Iy, Ix = np.gradient(ndimage.gaussian_filter(tubulin, sigma_grad))
    
    # Compute tensor components
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    
    # Smooth tensor components (integrate over neighborhood)
    # This is critical: without smoothing, anisotropy is always 1 for a single gradient vector.
    Jxx = ndimage.gaussian_filter(Ixx, sigma_tensor)
    Jyy = ndimage.gaussian_filter(Iyy, sigma_tensor)
    Jxy = ndimage.gaussian_filter(Ixy, sigma_tensor)
    
    # --- 4. Eigenvalue Analysis ---
    # Calculate eigenvalues of the 2x2 matrix [[Jxx, Jxy], [Jxy, Jyy]]
    # Trace = Jxx + Jyy = lambda1 + lambda2
    # Det = Jxx*Jyy - Jxy^2 = lambda1 * lambda2
    # Difference = sqrt(Trace^2 - 4*Det) = lambda1 - lambda2
    
    # More numerically stable approach for eigenvalues of symmetric matrix:
    # lambda1,2 = (Trace +/- sqrt((Jxx - Jyy)^2 + 4*Jxy^2)) / 2
    
    tmp = np.sqrt((Jxx - Jyy)**2 + 4 * Jxy**2)
    trace = Jxx + Jyy
    
    lambda1 = (trace + tmp) / 2
    lambda2 = (trace - tmp) / 2
    
    # --- 5. Compute Anisotropy ---
    # Coherence / Anisotropy = (lambda1 - lambda2) / (lambda1 + lambda2)
    # Range [0, 1]. 0 = isotropic (blobs/noise), 1 = anisotropic (lines/fibers)
    
    epsilon = 1e-7 # Avoid division by zero
    anisotropy_map = (lambda1 - lambda2) / (lambda1 + lambda2 + epsilon)
    
    # --- 6. Aggregation ---
    # Compute the mean anisotropy within the cellular mask
    masked_anisotropy = anisotropy_map[mask]
    
    if masked_anisotropy.size == 0:
        return 0.0
        
    result = np.mean(masked_anisotropy)
    
    return float(result)
