def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    
    # Convert to appropriate array type
    # The input is expected to be (H, W, C) = (512, 512, 3) and uint8
    arr = np.asarray(img)
    
    # Handle dimensionality and extract Actin channel (Channel 0)
    # If the image is 2D (H, W), assume it's a single channel image, but dataset says (H, W, 3)
    if arr.ndim == 3 and arr.shape[2] >= 1:
        actin_channel = arr[:, :, 0]  # Channel 0 is Actin
    elif arr.ndim == 2:
        actin_channel = arr  # Fallback if already single channel
    else:
        return 0.0

    # Ensure uint8 for GLCM calculation efficiency
    if actin_channel.dtype != np.uint8:
        # Normalize to 0-255 if not uint8
        min_val = np.min(actin_channel)
        max_val = np.max(actin_channel)
        if max_val > min_val:
            actin_channel = ((actin_channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            actin_channel = np.zeros_like(actin_channel, dtype=np.uint8)

    # Handle segmentation masks
    # We want to compute texture only within the cellular regions to avoid background noise
    mask = None
    if len(segmentation_masks) > 0:
        # Combine all available masks to define the cellular region
        combined_mask = np.zeros(actin_channel.shape, dtype=bool)
        for m in segmentation_masks:
            if m is not None:
                # Ensure mask matches image shape (handle potential squeezing)
                if m.shape == actin_channel.shape:
                    combined_mask = combined_mask | (m > 0)
                elif m.ndim == 3 and m.shape[:2] == actin_channel.shape:
                     combined_mask = combined_mask | (np.max(m, axis=2) > 0)
        
        if np.any(combined_mask):
            mask = combined_mask

    # Apply mask if available
    if mask is not None:
        # Set background to 0. 
        # Note: In GLCM, 0-0 transitions (background) contribute 0 to Contrast ((i-j)^2 = 0),
        # so they don't affect the numerator sum, but they do affect the normalization factor.
        # However, for standard Haralick implementations on masked images, a common approach 
        # is to compute on the rectangular ROI or masked array. 
        # Here we zero out background.
        masked_actin = actin_channel.copy()
        masked_actin[~mask] = 0
        
        # Optimization: Crop to bounding box of the mask to speed up GLCM
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return 0.0
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Add a small padding to ensure we don't lose edge texture, but stay within bounds
        rmin = max(0, rmin - 1)
        rmax = min(actin_channel.shape[0], rmax + 2)
        cmin = max(0, cmin - 1)
        cmax = min(actin_channel.shape[1], cmax + 2)
        
        roi = masked_actin[rmin:rmax, cmin:cmax]
    else:
        roi = actin_channel

    # Check if ROI is empty or too small
    if roi.size == 0:
        return 0.0

    # Quantization (Binning)
    # Reducing levels from 256 to 64 speeds up computation and improves robustness
    n_levels = 64
    # Integer division to bin: 0-3 -> 0, 4-7 -> 1, ..., 252-255 -> 63
    binned_roi = (roi // (256 // n_levels)).astype(np.uint8)
    
    # Clip to ensure no values exceed n_levels-1 (e.g. if value was 255)
    binned_roi = np.clip(binned_roi, 0, n_levels - 1)

    # Compute GLCM
    # Distances: 1 pixel (capture fine texture of actin fibers)
    # Angles: 0, 45, 90, 135 degrees (average for rotational invariance)
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    try:
        glcm = graycomatrix(binned_roi, distances=distances, angles=angles, 
                            levels=n_levels, symmetric=True, normed=True)
        
        # Compute Contrast
        # Contrast measures the intensity contrast between a pixel and its neighbor over the whole image.
        # Weights are (i-j)^2.
        # High contrast = sharp edges/fibers. Low contrast = smooth/diffuse.
        contrast_props = graycoprops(glcm, 'contrast')
        
        # Return the mean contrast across all 4 directions
        result = np.mean(contrast_props)
        
        return float(result)
        
    except Exception:
        return 0.0
