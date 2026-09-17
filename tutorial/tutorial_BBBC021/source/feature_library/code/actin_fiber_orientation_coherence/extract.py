def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) with channels: 0=Actin, 1=Tubulin, 2=DAPI
    # We need the Actin channel (Channel 0) for fiber orientation analysis
    if arr.ndim == 3 and arr.shape[2] == 3:
        actin_img = arr[..., 0]
    elif arr.ndim == 2:
        # Fallback if single channel passed, assume it's the relevant one
        actin_img = arr
    else:
        return 0.0

    # Intensity normalization
    # Normalize to [0, 1] for stable gradient calculation
    vmax = np.percentile(actin_img, 99.5) if actin_img.size > 0 else 1.0
    if vmax > 0:
        actin_img = actin_img / vmax
    actin_img = np.clip(actin_img, 0.0, 1.0)

    # Define Region of Interest (ROI)
    # We want to measure coherence only within the cellular structure, not the background noise.
    roi_mask = np.zeros_like(actin_img, dtype=bool)
    
    if len(segmentation_masks) > 0:
        # Combine all available segmentation masks
        for mask in segmentation_masks:
            if mask is not None:
                # Ensure mask matches image spatial dimensions
                if mask.shape == actin_img.shape:
                    roi_mask = roi_mask | (mask > 0)
                elif mask.ndim == 3 and mask.shape[:2] == actin_img.shape:
                     # Handle case where mask might be 3D (e.g. one-hot encoded or RGB)
                     roi_mask = roi_mask | (np.any(mask > 0, axis=2))
    
    # Fallback if no valid masks provided or mask is empty
    if not np.any(roi_mask):
        try:
            # Create a simple foreground mask using Otsu thresholding on the actin channel
            thresh = threshold_otsu(actin_img)
            roi_mask = actin_img > thresh
        except Exception:
            # If thresholding fails (e.g., uniform image), use the whole image
            roi_mask = np.ones_like(actin_img, dtype=bool)

    # Structure Tensor Calculation
    # 1. Compute gradients
    sigma_grad = 1.0  # Smoothing scale for derivatives
    Iy = ndimage.gaussian_filter(actin_img, sigma=sigma_grad, order=(1, 0))
    Ix = ndimage.gaussian_filter(actin_img, sigma=sigma_grad, order=(0, 1))

    # 2. Compute tensor components
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # 3. Smooth tensor components (Window integration)
    # This scale determines the neighborhood over which orientation is averaged
    sigma_tensor = 2.0 
    Jxx = ndimage.gaussian_filter(Ixx, sigma=sigma_tensor)
    Jyy = ndimage.gaussian_filter(Iyy, sigma=sigma_tensor)
    Jxy = ndimage.gaussian_filter(Ixy, sigma=sigma_tensor)

    # 4. Eigenvalue Analysis
    # Calculate eigenvalues of the structure tensor matrix [[Jxx, Jxy], [Jxy, Jyy]]
    # Trace = Jxx + Jyy
    # Det = Jxx*Jyy - Jxy*Jxy
    # Delta = sqrt((Jxx - Jyy)^2 + 4*Jxy^2)
    # L1 = (Trace + Delta) / 2
    # L2 = (Trace - Delta) / 2
    
    # We need the coherence metric: C = (L1 - L2) / (L1 + L2)
    # Note: L1 + L2 = Trace
    # L1 - L2 = Delta
    # So C = Delta / (Trace + epsilon)
    
    trace = Jxx + Jyy
    delta = np.sqrt((Jxx - Jyy)**2 + 4 * Jxy**2)
    
    # Small epsilon to avoid division by zero
    epsilon = 1e-7
    coherence_map = delta / (trace + epsilon)

    # Extract feature: Mean coherence within the ROI
    # We only consider pixels within the identified cell/tissue area
    valid_pixels = coherence_map[roi_mask]
    
    if valid_pixels.size == 0:
        return 0.0
        
    # Return the mean coherence
    # High coherence -> aligned fibers (anisotropic)
    # Low coherence -> disorganized meshwork or noise (isotropic)
    result = np.mean(valid_pixels)

    return float(result)
