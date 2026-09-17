def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu

    # 1. Input Validation and Preparation
    # Convert to float32 for processing
    arr = np.asarray(img, dtype=np.float32)

    # Check dimensions: Expected (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # 2. Channel Selection: Actin is Channel 0 (Red)
    actin_channel = arr[..., 0]

    # 3. Robust Normalization
    # Clip outliers to improve contrast for gradient calculation
    p1, p99 = np.percentile(actin_channel, [1, 99])
    if p99 > p1:
        actin_norm = (np.clip(actin_channel, p1, p99) - p1) / (p99 - p1)
    else:
        if p99 > 0:
            actin_norm = actin_channel / p99
        else:
            return 0.0 # Empty image

    # 4. Mask Generation
    # Use segmentation masks if available, otherwise generate a foreground mask
    mask = None
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first available mask (assuming it covers cells)
        # Masks are often labeled integers, convert to boolean
        mask = segmentation_masks[0] > 0
        # Ensure mask shape matches image shape (2D)
        if mask.shape != actin_norm.shape:
            # If mask is 3D or different size, fallback to generated mask
            mask = None

    if mask is None:
        # Generate mask using Otsu thresholding on the normalized actin channel
        try:
            thresh = threshold_otsu(actin_norm)
            mask = actin_norm > thresh
        except Exception:
            # Fallback if Otsu fails (e.g., uniform image)
            mask = np.ones_like(actin_norm, dtype=bool)

    # Safety check: if mask is empty, use whole image
    if np.sum(mask) == 0:
        mask = np.ones_like(actin_norm, dtype=bool)

    # 5. Structure Tensor Calculation (Coherence)
    # Gradients
    sigma = 1.0
    # Gaussian smoothing before gradient
    smooth = ndimage.gaussian_filter(actin_norm, sigma)
    
    # Derivatives
    Iy = ndimage.sobel(smooth, axis=0)
    Ix = ndimage.sobel(smooth, axis=1)

    # Structure tensor components (smoothed)
    # J = [[Ix^2, IxIy], [IxIy, Iy^2]] smoothed
    # Smoothing window for tensor integration (usually larger than gradient sigma)
    tensor_sigma = 3.0
    
    Ixx = ndimage.gaussian_filter(Ix**2, tensor_sigma)
    Iyy = ndimage.gaussian_filter(Iy**2, tensor_sigma)
    Ixy = ndimage.gaussian_filter(Ix*Iy, tensor_sigma)

    # 6. Coherence Computation
    # Coherence = ((Ixx - Iyy)^2 + 4*Ixy^2) / (Ixx + Iyy)^2
    # This measures anisotropy. 1 = perfectly aligned, 0 = isotropic.
    
    numerator = np.sqrt((Ixx - Iyy)**2 + 4 * Ixy**2)
    denominator = Ixx + Iyy
    
    # Avoid division by zero
    # Add a small epsilon to denominator
    epsilon = 1e-7
    coherence_map = numerator / (denominator + epsilon)

    # 7. Aggregation
    # Compute mean coherence only within the masked foreground region
    masked_coherence = coherence_map[mask]
    
    if masked_coherence.size == 0:
        return 0.0
        
    result = np.mean(masked_coherence)

    return float(result)
