def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.feature import structure_tensor, structure_tensor_eigenvalues
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) where Channel 0 is Actin (Red)
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Extract Actin channel (Channel 0)
        actin_channel = arr[..., 0]
    elif arr.ndim == 2:
        # Fallback if only one channel is passed (unlikely based on description but safe)
        actin_channel = arr
    else:
        # Unexpected format
        return 0.0

    # Intensity normalization
    # Normalize to [0, 1] for stable gradient calculation
    vmax = np.percentile(actin_channel, 99.5) if actin_channel.size > 0 else 1.0
    if vmax <= 0:
        vmax = 1.0
    actin_norm = actin_channel / vmax
    actin_norm = np.clip(actin_norm, 0.0, 1.0)

    # Determine Region of Interest (ROI)
    # We only want to compute coherency inside the cells, not on the background
    roi_mask = None
    
    if len(segmentation_masks) > 0:
        # Combine all provided masks into a single binary mask
        combined_mask = np.zeros(actin_norm.shape, dtype=bool)
        for mask in segmentation_masks:
            if mask is not None:
                # Handle potential shape mismatch if masks are squeezed differently
                if mask.shape == actin_norm.shape:
                    combined_mask = np.logical_or(combined_mask, mask > 0)
                elif mask.ndim == 3 and mask.shape[:2] == actin_norm.shape:
                     # If mask is 3D (e.g. one-hot encoded or just extra dim), flatten it
                    combined_mask = np.logical_or(combined_mask, np.any(mask > 0, axis=-1))
        
        if np.any(combined_mask):
            roi_mask = combined_mask

    # Fallback: If no masks provided or masks are empty, use Otsu thresholding on Actin
    if roi_mask is None:
        try:
            thresh = threshold_otsu(actin_norm)
            roi_mask = actin_norm > thresh
        except Exception:
            # Fallback for completely empty/flat images where Otsu fails
            roi_mask = np.ones(actin_norm.shape, dtype=bool)

    # If the mask is empty (no cells), return 0.0
    if np.sum(roi_mask) == 0:
        return 0.0

    # Compute Structure Tensor
    # sigma=1.0 for gradient calculation (noise suppression)
    # mode='reflect' handles boundaries
    # order='xy' is standard for skimage
    Axx, Axy, Ayy = structure_tensor(actin_norm, sigma=1.0, mode='reflect', order='xy')

    # Smooth the structure tensor components
    # This integration scale determines the window over which orientation is averaged.
    # A sigma of ~2.0-3.0 is typical for cellular fibers.
    sigma_tensor = 2.5
    Sxx = ndimage.gaussian_filter(Axx, sigma_tensor)
    Sxy = ndimage.gaussian_filter(Axy, sigma_tensor)
    Syy = ndimage.gaussian_filter(Ayy, sigma_tensor)

    # Compute Eigenvalues of the Structure Tensor
    # structure_tensor_eigenvalues returns (lambda1, lambda2) where lambda1 >= lambda2
    # Note: skimage's structure_tensor_eigenvalues expects the outputs of structure_tensor
    # but we smoothed them manually. We can calculate eigenvalues directly or pass smoothed components.
    # The function signature is structure_tensor_eigenvalues(A) where A is a list of arrays.
    l1, l2 = structure_tensor_eigenvalues([Sxx, Sxy, Syy])

    # Calculate Local Coherency
    # Coherency = (lambda1 - lambda2) / (lambda1 + lambda2)
    # Range is [0, 1]. 1 = perfectly anisotropic (lines), 0 = isotropic (blobs/noise)
    denominator = l1 + l2
    numerator = l1 - l2
    
    # Avoid division by zero
    epsilon = 1e-7
    coherency_map = numerator / (denominator + epsilon)

    # Extract coherency values only within the cellular ROI
    roi_coherency_values = coherency_map[roi_mask]

    # Compute the global mean coherency
    if roi_coherency_values.size > 0:
        result = np.mean(roi_coherency_values)
    else:
        result = 0.0

    return float(result)
