def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    
    # 1. Input Validation and Channel Selection
    # The dataset description specifies (512, 512, 3) for RGB images.
    # Channel 1 is Tubulin (Green).
    
    # Check if image is valid
    if img is None or img.size == 0:
        return 0.0
        
    # Handle dimensionality
    # If 3D (H, W, C), extract Tubulin channel (index 1)
    if img.ndim == 3 and img.shape[2] >= 2:
        tubulin_img = img[..., 1]
    # If 2D, assume it might be a single channel projection or pre-selected channel, 
    # but strictly based on dataset info, it's likely (H, W, C). 
    # If passed a 2D image, we use it directly.
    elif img.ndim == 2:
        tubulin_img = img
    else:
        return 0.0

    # 2. Preprocessing and Quantization
    # GLCM works on discrete integer levels. The input is uint8 (0-255).
    # Computing GLCM on 256 levels can be sparse and slow.
    # We quantize to 64 levels (0-63) to capture texture robustly while reducing noise.
    # 256 / 4 = 64.
    
    # Ensure input is uint8 before bit-shifting or division
    if tubulin_img.dtype != np.uint8:
        # Normalize to 0-255 if float
        if np.issubdtype(tubulin_img.dtype, np.floating):
            vmax = np.max(tubulin_img)
            if vmax > 0:
                tubulin_img = (tubulin_img / vmax * 255).astype(np.uint8)
            else:
                tubulin_img = tubulin_img.astype(np.uint8)
        else:
            # Clip and cast if other integer type
            tubulin_img = np.clip(tubulin_img, 0, 255).astype(np.uint8)

    # Apply Segmentation Mask if available
    # If a mask is provided, we want to ignore the background.
    # However, standard GLCM implementations on rectangular arrays don't support "masked" pixels natively 
    # without treating the boundary as a texture edge (0 to value transition).
    # A common approach in texture analysis for cells is to compute the GLCM on the bounding box 
    # or the whole image, but we can try to minimize background influence.
    # Here, we will zero out the background. While this creates a texture edge at the cell boundary,
    # the contrast of the internal tubulin network is usually the dominant signal in high-content images.
    
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask = segmentation_masks[0]
        # Ensure mask matches image shape
        if mask.shape == tubulin_img.shape:
            # Keep only the segmented regions
            tubulin_img = np.where(mask > 0, tubulin_img, 0)
        
    # Quantize to 64 levels
    n_levels = 64
    # Integer division by 4 maps 0-255 to 0-63
    binned_img = (tubulin_img // 4).astype(np.uint8)
    
    # 3. GLCM Computation
    # Parameters:
    # distances=[1]: We are interested in local texture (fiber sharpness).
    # angles=[0, 45, 90, 135]: Average over all directions for rotational invariance.
    # levels=64: Matches our quantization.
    # symmetric=True, normed=True: Standard GLCM settings.
    
    try:
        glcm = graycomatrix(
            binned_img, 
            distances=[1], 
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
            levels=n_levels, 
            symmetric=True, 
            normed=True
        )
    except ValueError:
        # Can happen if image is empty or has invalid values
        return 0.0

    # 4. Feature Extraction: Contrast
    # Contrast measures the local variations in the gray-level co-occurrence matrix.
    # High contrast = sharp transitions (distinct fibers).
    # Low contrast = smooth transitions (diffuse signal).
    contrast_matrix = graycoprops(glcm, 'contrast')
    
    # 5. Aggregation
    # Average the contrast across all 4 angles to get a single rotation-invariant scalar.
    result = np.mean(contrast_matrix)
    
    return float(result)
