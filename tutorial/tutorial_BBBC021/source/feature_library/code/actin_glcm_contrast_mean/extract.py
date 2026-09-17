def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    
    # 1. Input Validation and Channel Extraction
    # The dataset description specifies (512, 512, 3) where Channel 0 is Actin.
    # If the image is not 3D or doesn't have 3 channels, we attempt to handle it or return 0.
    
    img = np.asarray(img)
    
    # Check for valid dimensions
    if img.ndim != 3 or img.shape[2] < 1:
        return 0.0

    # Extract Actin channel (Channel 0)
    # Ensure it is uint8 for GLCM calculation
    if img.dtype != np.uint8:
        # Normalize to 0-255 if not already uint8
        img_float = img.astype(np.float32)
        img_min, img_max = img_float.min(), img_float.max()
        if img_max > img_min:
            img_norm = (img_float - img_min) / (img_max - img_min)
            actin_channel = (img_norm[:, :, 0] * 255).astype(np.uint8)
        else:
            actin_channel = np.zeros(img.shape[:2], dtype=np.uint8)
    else:
        actin_channel = img[:, :, 0]

    # 2. Apply Segmentation Mask (if available)
    # If masks are provided, we use the first one to mask out the background.
    # This ensures texture features are calculated primarily on the cells.
    if segmentation_masks and len(segmentation_masks) > 0:
        mask = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask.shape == actin_channel.shape:
            # Set background pixels to 0
            # We assume labeled mask where 0 is background
            actin_channel = np.where(mask > 0, actin_channel, 0)

    # 3. Quantization (Binning)
    # GLCM calculation on full 256 levels is slow and sparse.
    # We bin the image to 64 levels (reduction factor of 4).
    # This is a standard practice in texture analysis to improve statistical reliability.
    n_levels = 64
    binned_actin = (actin_channel // 4).astype(np.uint8)
    
    # Clip to ensure no values exceed n_levels-1 (though //4 on uint8 max 255 gives 63)
    binned_actin = np.clip(binned_actin, 0, n_levels - 1)

    # 4. Compute GLCM
    # Parameters:
    # - distances: [1] (immediate neighbors for fine texture like actin fibers)
    # - angles: [0, 45, 90, 135] degrees (0, np.pi/4, np.pi/2, 3*np.pi/4) for rotational invariance
    # - levels: 64
    # - symmetric: True
    # - normed: True (returns probabilities)
    try:
        glcm = graycomatrix(
            binned_actin, 
            distances=[1], 
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
            levels=n_levels, 
            symmetric=True, 
            normed=True
        )
    except ValueError:
        # Handle edge cases where image might be empty or invalid
        return 0.0

    # 5. Extract Contrast Feature
    # Contrast measures the local variations in the gray-level co-occurrence matrix.
    # High contrast = sharp transitions (fibers, edges).
    contrast_matrix = graycoprops(glcm, 'contrast')
    
    # 6. Aggregate Results
    # Average the contrast across all 4 directions to get a rotation-invariant scalar.
    mean_contrast = np.mean(contrast_matrix)

    return float(mean_contrast)
