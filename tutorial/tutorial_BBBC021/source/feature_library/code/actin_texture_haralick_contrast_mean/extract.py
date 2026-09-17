def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    
    # 1. Input Validation and Formatting
    # Ensure input is a numpy array
    img = np.asarray(img)
    
    # Check for valid dimensions. Expected (512, 512, 3)
    if img.ndim != 3 or img.shape[2] < 1:
        return 0.0
        
    # 2. Channel Selection
    # Dataset spec: Channel 0 is Actin (Red)
    # The feature is specifically for Actin texture
    actin_channel = img[:, :, 0]
    
    # 3. Data Type Handling for GLCM
    # GLCM requires integer types. The dataset is uint8 (0-255).
    # If it's not uint8, we need to rescale/cast, but based on specs it is uint8.
    if actin_channel.dtype != np.uint8:
        # If float 0-1, scale to 0-255
        if np.max(actin_channel) <= 1.0 and np.issubdtype(actin_channel.dtype, np.floating):
            actin_channel = (actin_channel * 255).astype(np.uint8)
        else:
            # Just cast safely
            actin_channel = actin_channel.astype(np.uint8)

    # 4. Mask Handling
    # If segmentation masks are provided, we use them to focus on cellular regions.
    # If not, we analyze the whole image (background 0s will contribute to GLCM).
    # Calculating texture on background (0-0 pairs) adds to the denominator (normalization)
    # but adds 0 to the contrast numerator, effectively diluting the value.
    # However, for consistency when masks are available, we should mask out non-cell areas.
    
    final_mask = None
    if segmentation_masks:
        # Combine all available masks (e.g., nuclei + cytoplasm)
        # Masks are labeled integers; convert to boolean
        valid_masks = [m for m in segmentation_masks if m is not None and m.shape == actin_channel.shape]
        if valid_masks:
            # Create a union of all masks
            combined_mask = np.zeros(actin_channel.shape, dtype=bool)
            for m in valid_masks:
                combined_mask = np.logical_or(combined_mask, m > 0)
            final_mask = combined_mask

    # Apply mask if it exists
    if final_mask is not None:
        # Set background pixels to 0. 
        # Note: This creates artificial edges at the boundary of the mask,
        # but is standard practice when rectangular ROIs aren't feasible.
        # Ideally, we would compute GLCM only on pixels within the mask, 
        # but skimage.graycomatrix computes on the full rectangular array.
        # A common workaround is to set background to 0.
        actin_channel = actin_channel * final_mask.astype(np.uint8)
        
        # Optimization: If the mask is empty, return 0
        if not np.any(final_mask):
            return 0.0

    # 5. GLCM Computation
    # Parameters:
    # - distances=[1]: Look at immediate neighbors for fine texture (roughness)
    # - angles=[0, 45, 90, 135]: Average over 4 directions for rotational invariance
    # - levels=256: For uint8 data
    # - symmetric=True: Relationship i->j is same as j->i
    # - normed=True: Normalize matrix to probabilities
    
    try:
        glcm = graycomatrix(
            actin_channel, 
            distances=[1], 
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
            levels=256, 
            symmetric=True, 
            normed=True
        )
    except ValueError:
        # Handle cases where image might be empty or invalid for GLCM
        return 0.0

    # 6. Feature Extraction: Contrast
    # Contrast: sum(|i-j|^2 * p(i,j))
    # Measures local intensity variation. High contrast = rough texture / sharp edges.
    contrast_props = graycoprops(glcm, 'contrast')
    
    # 7. Aggregation
    # Average the contrast across the 4 angles to get a rotation-invariant scalar
    mean_contrast = np.mean(contrast_props)
    
    return float(mean_contrast)
