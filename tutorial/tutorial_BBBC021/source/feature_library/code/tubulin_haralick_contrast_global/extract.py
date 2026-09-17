def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    from skimage.filters import threshold_otsu
    
    # 1. Input Validation and Channel Extraction
    # The dataset description specifies (512, 512, 3) with Channel 1 being Tubulin (Green).
    # We need to handle potential variations or unexpected shapes gracefully.
    
    img_arr = np.asarray(img)
    
    # Check for valid dimensions
    if img_arr.ndim != 3 or img_arr.shape[2] < 2:
        # If not a multi-channel image as expected, return 0.0
        return 0.0
        
    # Extract Tubulin channel (Channel 1)
    tubulin_channel = img_arr[:, :, 1]
    
    # 2. Data Type Handling
    # GLCM requires integer types. The dataset is uint8 (0-255).
    # If the input is float, we need to scale and cast. If it's already uint8, we use it directly.
    if np.issubdtype(tubulin_channel.dtype, np.floating):
        # Normalize to 0-255 if float
        min_val = np.min(tubulin_channel)
        max_val = np.max(tubulin_channel)
        if max_val > min_val:
            tubulin_channel = ((tubulin_channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            tubulin_channel = np.zeros_like(tubulin_channel, dtype=np.uint8)
    elif tubulin_channel.dtype != np.uint8:
        # If it's some other integer type (e.g. uint16), we might need to bin it to 256 levels
        # for performance, or just cast if the range is small. 
        # Given the dataset description says uint8, we ensure it is uint8.
        # Safe conversion for uint16 -> uint8 usually involves scaling, but here we assume standard 8-bit range.
        if np.max(tubulin_channel) > 255:
             tubulin_channel = (tubulin_channel / np.max(tubulin_channel) * 255).astype(np.uint8)
        else:
             tubulin_channel = tubulin_channel.astype(np.uint8)

    # 3. Masking / ROI Definition
    # We want to calculate texture primarily within the cells to avoid background noise affecting the contrast.
    
    mask = None
    if len(segmentation_masks) > 0:
        # Combine all available masks to define the cellular region
        combined_mask = np.zeros(tubulin_channel.shape, dtype=bool)
        for m in segmentation_masks:
            if m is not None and m.shape == tubulin_channel.shape:
                combined_mask = np.logical_or(combined_mask, m > 0)
        
        if np.any(combined_mask):
            mask = combined_mask
    
    # Fallback: If no segmentation masks provided or they are empty, generate a mask using Otsu
    if mask is None:
        try:
            thresh = threshold_otsu(tubulin_channel)
            mask = tubulin_channel > thresh
        except Exception:
            # If otsu fails (e.g. constant image), use the whole image
            mask = np.ones(tubulin_channel.shape, dtype=bool)

    # Apply mask: Set background to 0. 
    # Note: This creates a strong edge at the boundary (signal -> 0). 
    # However, standard GLCM implementations on rectangular arrays usually require a full image.
    # A common approach is to compute GLCM on the masked array. The 0-0 transitions (background)
    # will dominate the (0,0) entry of the GLCM, but 'Contrast' weights the diagonal (i-j=0) by 0,
    # so the background-background texture contributes 0 to the contrast sum.
    # The boundary pixels (signal-background) will contribute to high contrast, which is acceptable
    # as it reflects cell shape complexity, but ideally we want internal texture.
    # To mitigate boundary effects, we can just pass the masked image.
    
    masked_tubulin = tubulin_channel.copy()
    masked_tubulin[~mask] = 0
    
    # 4. Compute GLCM
    # Parameters:
    # - distances=[1]: Pixel adjacency.
    # - angles=[0, 45, 90, 135]: Rotational invariance (average over directions).
    # - levels=256: For uint8.
    
    # Check if image is empty
    if np.sum(masked_tubulin) == 0:
        return 0.0

    try:
        glcm = graycomatrix(masked_tubulin, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                            levels=256, symmetric=True, normed=True)
        
        # 5. Extract Haralick Contrast
        # Contrast: sum_{i,j} |i-j|^2 * p(i,j)
        # Measures local variations. High contrast = sharp transitions (e.g., microtubule bundles).
        # Low contrast = smooth/diffuse (e.g., depolymerized tubulin).
        contrast_props = graycoprops(glcm, 'contrast')
        
        # Average across all angles to get a rotation-invariant global feature
        global_contrast = np.mean(contrast_props)
        
        return float(global_contrast)
        
    except Exception:
        return 0.0
