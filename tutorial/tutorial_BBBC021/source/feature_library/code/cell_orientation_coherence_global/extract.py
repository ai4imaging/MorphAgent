def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    
    # 1. Data Loading and Validation
    # Convert to float32 for processing
    arr = np.asarray(img, dtype=np.float32)
    
    # Check dimensionality and select Tubulin channel (Channel 1)
    # Dataset is (512, 512, 3) RGB TIFF.
    # Channel 0: Actin (Red)
    # Channel 1: Tubulin (Green) -> Target
    # Channel 2: DAPI (Blue)
    
    if arr.ndim == 3 and arr.shape[2] == 3:
        tubulin = arr[..., 1]
    elif arr.ndim == 2:
        # Fallback if single channel passed
        tubulin = arr
    else:
        return 0.0

    # 2. Normalization
    # Check if image is already normalized to [0, 1] or is uint8 [0, 255]
    # Previous error guidance: check max value to avoid double division
    if tubulin.max() > 1.0:
        tubulin = tubulin / 255.0
    
    # Clip to ensure range [0, 1]
    tubulin = np.clip(tubulin, 0.0, 1.0)

    # 3. Mask Generation (Region of Interest)
    # We want to calculate coherence only within the cellular regions, not the black background.
    # Use segmentation masks if available, otherwise generate a simple mask.
    mask = None
    
    # Try to use provided segmentation masks
    if len(segmentation_masks) > 0:
        # Prefer a mask that covers the cytoplasm/whole cell. 
        # Usually masks are labeled integers. Convert to boolean.
        # If multiple masks, we might just take the union of all non-zero pixels from the first mask
        # assuming it's a cell mask.
        seg = segmentation_masks[0]
        if seg.shape == tubulin.shape:
            mask = seg > 0
    
    # Fallback: Generate mask from image if no valid external mask
    if mask is None or np.sum(mask) == 0:
        # Simple thresholding strategy
        # Gaussian blur to reduce noise before thresholding
        smooth = ndimage.gaussian_filter(tubulin, sigma=2.0)
        try:
            thresh = threshold_otsu(smooth)
            mask = smooth > thresh
        except Exception:
            # Fallback for very low contrast images or empty images
            mask = smooth > np.mean(smooth)

    # Safety check: if mask is still too small (e.g. < 1% of image), use whole image
    if np.sum(mask) < (0.01 * mask.size):
        mask = np.ones_like(tubulin, dtype=bool)

    # 4. Structure Tensor Calculation
    # Coherence is derived from the eigenvalues of the structure tensor.
    # J = [[Ix^2, IxIy], [IxIy, Iy^2]] smoothed by a window.
    
    # Compute gradients
    sigma_deriv = 1.0
    Ix = ndimage.gaussian_filter(tubulin, sigma=sigma_deriv, order=(0, 1))
    Iy = ndimage.gaussian_filter(tubulin, sigma=sigma_deriv, order=(1, 0))

    # Compute tensor components
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # Smooth tensor components (integration scale)
    # This window determines the neighborhood over which orientation is averaged
    sigma_integ = 3.0
    Jxx = ndimage.gaussian_filter(Ixx, sigma=sigma_integ)
    Jyy = ndimage.gaussian_filter(Iyy, sigma=sigma_integ)
    Jxy = ndimage.gaussian_filter(Ixy, sigma=sigma_integ)

    # 5. Eigenvalue Analysis
    # The eigenvalues of the 2x2 matrix [[Jxx, Jxy], [Jxy, Jyy]] are:
    # lambda1,2 = 0.5 * ( (Jxx + Jyy) +/- sqrt((Jxx - Jyy)^2 + 4*Jxy^2) )
    # Coherence = (lambda1 - lambda2) / (lambda1 + lambda2) ^ 2  <-- There are various definitions.
    # A common definition for local coherence (anisotropy) is:
    # C = sqrt((Jxx - Jyy)^2 + 4*Jxy^2) / (Jxx + Jyy)
    # This ranges from 0 (isotropic) to 1 (perfectly oriented 1D structure).

    # Trace = Jxx + Jyy
    trace = Jxx + Jyy
    
    # Difference of eigenvalues (discriminant part)
    diff = np.sqrt((Jxx - Jyy)**2 + 4 * Jxy**2)
    
    # Avoid division by zero
    epsilon = 1e-7
    coherence_map = diff / (trace + epsilon)

    # 6. Aggregation
    # We want the "global coherence". We average the local coherence values 
    # within the masked region.
    
    # Apply mask
    valid_coherence = coherence_map[mask]
    
    if valid_coherence.size == 0:
        return 0.0
        
    # Calculate mean coherence
    # We use nanmean just in case of any numerical instability
    global_coherence = np.nanmean(valid_coherence)

    return float(global_coherence)
