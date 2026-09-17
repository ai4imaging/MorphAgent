def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    
    # 1. Data Preparation and Validation
    # Ensure input is a numpy array
    arr = np.asarray(img)
    
    # Check dimensionality and extract Actin channel (Channel 0)
    # Dataset spec: (512, 512, 3), Channel 0 is Actin (Red)
    if arr.ndim == 3 and arr.shape[2] >= 1:
        actin_channel = arr[:, :, 0]
    elif arr.ndim == 2:
        # Fallback if passed a single channel image, assuming it's the correct one
        actin_channel = arr
    else:
        # Unexpected format
        return 0.0

    # 2. Type Conversion for GLCM
    # GLCM calculation requires integer types. The dataset is uint8 (0-255).
    # If the input is float (e.g., normalized 0-1), scale it back to 0-255.
    if np.issubdtype(actin_channel.dtype, np.floating):
        # Normalize to 0-255 if it looks like a float image
        if actin_channel.max() <= 1.0:
            actin_channel = (actin_channel * 255).astype(np.uint8)
        else:
            actin_channel = actin_channel.astype(np.uint8)
    else:
        # Ensure it is uint8
        actin_channel = actin_channel.astype(np.uint8)

    # 3. Mask Handling (Optional)
    # If segmentation masks are provided, we can mask out the background.
    # However, standard GLCM is rectangular. Masking usually involves setting background to 0.
    # This creates a strong edge at the boundary, but ensures we don't calculate texture on noise.
    if segmentation_masks:
        # Combine all available masks into a single binary mask
        combined_mask = np.zeros(actin_channel.shape, dtype=bool)
        for mask in segmentation_masks:
            if mask is not None:
                # Handle potential shape mismatches (e.g. if mask is 2D and image is 3D projected)
                if mask.shape == actin_channel.shape:
                    combined_mask = np.logical_or(combined_mask, mask > 0)
        
        # Apply mask if we found valid masks
        if np.any(combined_mask):
            actin_channel = actin_channel * combined_mask.astype(np.uint8)

    # 4. GLCM Computation
    # Parameters:
    # - distances=[1]: Pixel immediate neighbors (captures fine texture)
    # - angles=[0, 45, 90, 135]: All 4 directions for rotational invariance
    # - levels=256: For uint8 image
    # - symmetric=True, normed=True: Standard GLCM settings
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
        # Can happen if image is empty or has invalid values
        return 0.0

    # 5. Feature Extraction: Contrast
    # Contrast measures the local intensity variation.
    # graycoprops returns a 2D array (n_distances, n_angles)
    contrast_matrix = graycoprops(glcm, 'contrast')
    
    # 6. Aggregation
    # Average the contrast across all 4 angles to get a rotation-invariant feature
    result = np.mean(contrast_matrix)

    return float(result)
