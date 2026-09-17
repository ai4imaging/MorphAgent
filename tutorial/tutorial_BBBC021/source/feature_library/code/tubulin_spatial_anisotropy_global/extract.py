def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu

    # 1. Data Loading and Preprocessing
    # Convert to float32 for precision
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality based on dataset description
    # Dataset is (512, 512, 3), Channel 1 is Tubulin (Green)
    target_channel_idx = 1
    
    if arr.ndim == 3:
        if arr.shape[2] == 3:
            # Standard (H, W, C) format
            tubulin_img = arr[..., target_channel_idx]
        elif arr.shape[0] == 3:
            # Channel-first (C, H, W) - less likely based on desc but possible
            tubulin_img = arr[target_channel_idx, ...]
        else:
            # Fallback: if channels don't match 3, try to use the middle one or just the image if it's a stack
            # Given the strict dataset desc, we assume (H, W, 3)
            tubulin_img = np.mean(arr, axis=2) # Fallback to grayscale
    elif arr.ndim == 2:
        # Already 2D, assume it's the correct channel or a projection
        tubulin_img = arr
    else:
        return 0.0

    # Normalize intensity to [0, 1]
    # This is critical for structure tensor magnitude stability
    v_min, v_max = np.min(tubulin_img), np.max(tubulin_img)
    if v_max - v_min > 1e-6:
        norm_img = (tubulin_img - v_min) / (v_max - v_min)
    else:
        norm_img = np.zeros_like(tubulin_img)

    # 2. Define Region of Interest (ROI)
    # We only want to calculate anisotropy where there is actual biological signal (cells),
    # not in the background noise.
    mask = None
    
    # Try to use provided segmentation masks
    if len(segmentation_masks) > 0:
        # Combine all available masks (logical OR) to cover all cellular structures
        combined_mask = np.zeros_like(norm_img, dtype=bool)
        for m in segmentation_masks:
            if m is not None:
                # Handle potential shape mismatches (e.g. if mask is 2D and img was 3D)
                if m.shape == norm_img.shape:
                    combined_mask = np.logical_or(combined_mask, m > 0)
                elif m.ndim == 3 and m.shape[:2] == norm_img.shape:
                     combined_mask = np.logical_or(combined_mask, np.max(m, axis=2) > 0)
        
        if np.any(combined_mask):
            mask = combined_mask

    # Fallback: Otsu thresholding if no masks provided or masks were empty
    if mask is None:
        try:
            thresh = threshold_otsu(norm_img)
            mask = norm_img > thresh
        except Exception:
            # If Otsu fails (e.g. uniform image), use entire image
            mask = np.ones_like(norm_img, dtype=bool)

    # If mask is still effectively empty (very rare), return 0.0
    if np.sum(mask) == 0:
        return 0.0

    # 3. Compute Structure Tensor
    # Parameters for structure tensor
    sigma_grad = 1.0  # Scale for derivative calculation
    sigma_tensor = 2.0 # Scale for smoothing the tensor (integration scale)

    # Compute gradients (Ix, Iy)
    # Using Gaussian derivatives is more robust to noise than simple finite differences
    Ix = ndimage.gaussian_filter(norm_img, sigma=sigma_grad, order=(0, 1))
    Iy = ndimage.gaussian_filter(norm_img, sigma=sigma_grad, order=(1, 0))

    # Compute tensor components
    Jxx = Ix**2
    Jxy = Ix * Iy
    Jyy = Iy**2

    # Smooth tensor components
    # This averages the orientation information over a local neighborhood
    Jxx = ndimage.gaussian_filter(Jxx, sigma=sigma_tensor)
    Jxy = ndimage.gaussian_filter(Jxy, sigma=sigma_tensor)
    Jyy = ndimage.gaussian_filter(Jyy, sigma=sigma_tensor)

    # 4. Compute Anisotropy (Coherence)
    # Coherence = (lambda1 - lambda2) / (lambda1 + lambda2)
    # Formula using trace and determinant components:
    # Coherence = sqrt((Jxx - Jyy)^2 + 4*Jxy^2) / (Jxx + Jyy)
    
    numerator = np.sqrt((Jxx - Jyy)**2 + 4 * Jxy**2)
    denominator = Jxx + Jyy + 1e-9 # Add epsilon to avoid division by zero

    coherence_map = numerator / denominator

    # 5. Aggregate Feature
    # Calculate mean coherence only within the cellular mask
    masked_coherence = coherence_map[mask]
    
    if masked_coherence.size == 0:
        return 0.0
        
    global_anisotropy = np.mean(masked_coherence)

    return float(global_anisotropy)
