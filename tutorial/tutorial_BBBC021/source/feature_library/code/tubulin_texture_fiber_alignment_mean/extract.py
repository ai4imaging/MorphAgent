def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    
    # --- 1. Data Loading and Preprocessing ---
    
    # Convert to float32 for precision
    arr = np.asarray(img, dtype=np.float32)
    
    # Check dimensionality and extract Tubulin channel (Channel 1)
    # Expected shape: (512, 512, 3)
    if arr.ndim == 3 and arr.shape[2] >= 2:
        # Extract Green channel (Tubulin)
        tubulin = arr[..., 1]
    elif arr.ndim == 2:
        # Fallback if single channel passed (unlikely based on spec, but safe)
        tubulin = arr
    else:
        return 0.0

    # Normalize intensity to [0, 1]
    # Robust normalization using percentiles to handle outliers
    p_min, p_max = np.percentile(tubulin, (1, 99))
    if p_max > p_min:
        tubulin = (tubulin - p_min) / (p_max - p_min)
    else:
        tubulin = tubulin  # Keep as is if flat
    tubulin = np.clip(tubulin, 0.0, 1.0)
    
    # --- 2. Mask Generation (ROI) ---
    
    # We need to compute alignment only within the cells to avoid background noise bias.
    mask = None
    
    # Strategy A: Use provided segmentation masks
    if len(segmentation_masks) > 0:
        # Combine all available masks into a single boolean mask
        combined_mask = np.zeros(tubulin.shape, dtype=bool)
        for m in segmentation_masks:
            if m is not None and m.shape == tubulin.shape:
                combined_mask = np.logical_or(combined_mask, m > 0)
        
        if np.any(combined_mask):
            mask = combined_mask

    # Strategy B: Fallback to intensity thresholding if no valid masks provided
    if mask is None:
        # Simple background segmentation on the tubulin channel
        # Use Otsu's method to find a threshold
        try:
            # Check if image has variance
            if np.min(tubulin) == np.max(tubulin):
                return 0.0
            thresh = threshold_otsu(tubulin)
            mask = tubulin > thresh
        except Exception:
            # Fallback for extremely low signal images
            mask = tubulin > 0.1

    # Ensure mask is boolean
    mask = mask.astype(bool)
    
    # If mask is empty (no cells), return 0.0
    if not np.any(mask):
        return 0.0

    # --- 3. Structure Tensor Calculation ---
    
    # The Structure Tensor (Second-Moment Matrix) allows us to estimate local orientation and coherence.
    # J = [[Ix^2, IxIy], [IxIy, Iy^2]] smoothed by a Gaussian window.
    
    # Parameters
    sigma_grad = 1.0  # Scale for gradient calculation (noise suppression)
    sigma_tensor = 2.5 # Scale for tensor integration (window size for averaging orientation)
    
    # Compute gradients
    # We use Gaussian derivatives for robustness against noise
    Iy, Ix = np.gradient(ndimage.gaussian_filter(tubulin, sigma_grad))
    
    # Compute tensor components
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    
    # Smooth tensor components (integrate over local neighborhood)
    # This averaging is crucial to find the dominant direction in a region
    Txx = ndimage.gaussian_filter(Ixx, sigma_tensor)
    Tyy = ndimage.gaussian_filter(Iyy, sigma_tensor)
    Txy = ndimage.gaussian_filter(Ixy, sigma_tensor)
    
    # --- 4. Coherence (Anisotropy) Calculation ---
    
    # Calculate eigenvalues of the structure tensor matrix for each pixel
    # Matrix M = [[Txx, Txy], [Txy, Tyy]]
    # Trace = Txx + Tyy = lambda1 + lambda2
    # Det = Txx*Tyy - Txy*Txy = lambda1 * lambda2
    # Difference of eigenvalues = sqrt(Trace^2 - 4*Det)
    # Coherence = (lambda1 - lambda2) / (lambda1 + lambda2)
    
    # Calculation using trace and difference term directly:
    # Diff term corresponds to sqrt((Txx - Tyy)^2 + 4*Txy^2)
    
    tmp_diff = (Txx - Tyy)
    eigen_diff = np.sqrt(tmp_diff**2 + 4 * Txy**2)
    trace = Txx + Tyy
    
    # Avoid division by zero
    epsilon = 1e-7
    coherence_map = eigen_diff / (trace + epsilon)
    
    # Clip to valid range [0, 1] (mathematically should be, but float errors can occur)
    coherence_map = np.clip(coherence_map, 0.0, 1.0)
    
    # --- 5. Aggregation ---
    
    # We only care about the coherence within the cellular regions (mask).
    # Background noise often has high random coherence or zero coherence depending on smoothness.
    
    # Extract values within the mask
    valid_coherence_values = coherence_map[mask]
    
    # Compute the mean coherence
    # High values indicate aligned fibers (bundles), low values indicate isotropic meshwork
    result = np.mean(valid_coherence_values)
    
    return float(result)
