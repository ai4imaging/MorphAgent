def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), Channel 0 = Actin, Channel 1 = Tubulin, Channel 2 = DAPI
    # We need Channel 0 (Actin) for this feature
    if arr.ndim == 3 and arr.shape[2] == 3:
        actin_channel = arr[:, :, 0]
    elif arr.ndim == 2:
        # Fallback if single channel passed, though dataset spec says 3-channel
        actin_channel = arr
    else:
        return 0.0

    # Intensity normalization
    # Normalize to [0, 1] range for consistent contrast calculation
    # Using robust max to handle outliers
    vmax = np.percentile(actin_channel, 99.5) if actin_channel.size > 0 else 1.0
    if vmax <= 0:
        vmax = 1.0
    actin_norm = actin_channel / vmax
    actin_norm = np.clip(actin_norm, 0.0, 1.0)

    # Determine Analysis Mask (Region of Interest)
    # We want to calculate contrast only within the cell/cytoplasm, not the background
    analysis_mask = None

    # 1. Try to use provided segmentation masks
    if len(segmentation_masks) > 0:
        # Combine all available masks to define the cellular region
        # Masks are typically labeled integers (0=bg, 1..N=cells)
        combined_mask = np.zeros(actin_channel.shape, dtype=bool)
        for mask in segmentation_masks:
            if mask is not None and mask.shape == actin_channel.shape:
                combined_mask = np.logical_or(combined_mask, mask > 0)
        
        if np.sum(combined_mask) > 0:
            analysis_mask = combined_mask

    # 2. Fallback: Generate mask via thresholding if no valid masks provided
    if analysis_mask is None:
        try:
            # Simple background exclusion
            thresh = threshold_otsu(actin_norm)
            analysis_mask = actin_norm > thresh
        except Exception:
            # Fallback for extremely low signal images where Otsu might fail
            analysis_mask = actin_norm > 0.05

    # Check for empty mask
    if np.sum(analysis_mask) == 0:
        return 0.0

    # Compute Local Contrast (Local Standard Deviation)
    # Algorithm: Var(X) = E[X^2] - (E[X])^2
    # We use a small kernel to capture texture of filaments (stress fibers)
    kernel_size = 3
    
    # Calculate local mean
    local_mean = ndimage.uniform_filter(actin_norm, size=kernel_size)
    
    # Calculate local mean of squares
    local_sqr_mean = ndimage.uniform_filter(actin_norm**2, size=kernel_size)
    
    # Calculate local variance
    local_var = local_sqr_mean - local_mean**2
    
    # Numerical stability: clip negative values due to floating point errors
    local_var = np.maximum(local_var, 0)
    
    # Calculate local standard deviation (contrast)
    local_std = np.sqrt(local_var)

    # Aggregate Feature: Mean contrast within the cellular region
    # We only care about the texture inside the cells
    masked_contrast = local_std[analysis_mask]
    
    if masked_contrast.size == 0:
        return 0.0
        
    result = np.mean(masked_contrast)

    return float(result)
