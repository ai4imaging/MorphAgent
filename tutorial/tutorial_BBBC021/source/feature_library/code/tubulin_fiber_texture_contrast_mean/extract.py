def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    from skimage.filters import threshold_otsu
    
    # 1. Input Validation and Preprocessing
    # Convert to appropriate array type
    arr = np.asarray(img)
    
    # Check dimensionality and extract Tubulin channel (Channel 1)
    # Dataset format: (512, 512, 3), Channel 1 is Green/Tubulin
    if arr.ndim == 3 and arr.shape[2] == 3:
        tubulin_channel = arr[:, :, 1]
    elif arr.ndim == 2:
        # Fallback if single channel passed (unlikely based on spec but safe)
        tubulin_channel = arr
    else:
        return 0.0

    # Ensure float for processing, but keep original range logic in mind
    # The input is uint8 (0-255).
    tubulin_channel = tubulin_channel.astype(np.float32)

    # 2. Mask Generation
    # We need to define the "cell boundaries" to compute texture only within cells.
    mask = None
    
    if len(segmentation_masks) > 0:
        # Combine all provided masks
        combined_mask = np.zeros(tubulin_channel.shape, dtype=bool)
        for m in segmentation_masks:
            if m is not None and m.shape == tubulin_channel.shape:
                combined_mask = combined_mask | (m > 0)
        
        if np.any(combined_mask):
            mask = combined_mask

    # Fallback: If no valid mask provided or mask is empty, generate one using Otsu
    if mask is None:
        # Simple background exclusion
        if np.max(tubulin_channel) > np.min(tubulin_channel):
            try:
                thresh = threshold_otsu(tubulin_channel)
                mask = tubulin_channel > thresh
            except Exception:
                # Fallback for extremely low contrast images
                mask = tubulin_channel > np.mean(tubulin_channel)
        else:
            # Flat image, no texture
            return 0.0

    # If mask is still empty (e.g. completely black image), return 0
    if not np.any(mask):
        return 0.0

    # 3. Quantization for GLCM
    # GLCM is sensitive to noise and computationally expensive with 256 levels.
    # We quantize to fewer levels (e.g., 64) to capture robust texture features.
    # Crucially, we set background pixels to 0 and foreground pixels to 1-64.
    
    n_levels = 64
    
    # Normalize foreground pixels to 0-1 range relative to foreground min/max
    fg_pixels = tubulin_channel[mask]
    if fg_pixels.size == 0:
        return 0.0
        
    p_min, p_max = np.min(fg_pixels), np.max(fg_pixels)
    
    # Create a quantized image initialized to 0 (background)
    quantized_img = np.zeros(tubulin_channel.shape, dtype=np.uint8)
    
    if p_max > p_min:
        # Scale foreground to 0-(n_levels-1)
        scaled_fg = (fg_pixels - p_min) / (p_max - p_min)
        # Map to 1-n_levels (reserving 0 for background)
        quantized_fg = (scaled_fg * (n_levels - 1)).astype(np.uint8) + 1
        quantized_img[mask] = quantized_fg
    else:
        # Uniform foreground
        quantized_img[mask] = 1

    # 4. GLCM Calculation
    # We compute GLCM for the whole image including the 0-background.
    # Then we slice the matrix to exclude the background interactions.
    
    # Parameters: distance=1, angles=[0, 45, 90, 135] degrees
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    # Levels = n_levels + 1 (because 0 is background, 1-64 are data)
    try:
        glcm = graycomatrix(quantized_img, distances=distances, angles=angles, 
                            levels=n_levels + 1, symmetric=True, normed=False)
    except ValueError:
        return 0.0

    # 5. Feature Extraction (Contrast)
    # Slice GLCM to remove row 0 and col 0. This leaves only cell-pixel to cell-pixel interactions.
    # glcm shape: (levels, levels, num_distances, num_angles)
    glcm_internal = glcm[1:, 1:, :, :]
    
    # Sum counts across all angles (isotropic texture assumption)
    # Shape becomes (levels-1, levels-1)
    glcm_sum = np.sum(glcm_internal, axis=(2, 3))
    
    total_count = np.sum(glcm_sum)
    if total_count == 0:
        return 0.0
        
    # Normalize to probability distribution
    p_norm = glcm_sum / total_count
    
    # Calculate Contrast: sum_{i,j} |i-j|^2 * p(i,j)
    # Create grid of indices
    rows, cols = p_norm.shape
    i_indices, j_indices = np.indices((rows, cols))
    
    # The indices i, j correspond to quantized levels.
    # Since we mapped min->1 and max->64, the difference |i-j| reflects relative intensity contrast.
    term = (i_indices - j_indices) ** 2
    contrast = np.sum(term * p_norm)
    
    return float(contrast)
