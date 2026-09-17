def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    from skimage import img_as_ubyte
    
    # 1. Input Validation and Channel Extraction
    # Ensure input is a numpy array
    img = np.asarray(img)
    
    # Check dimensionality and extract Actin channel (Channel 0)
    # Dataset spec: (512, 512, 3), Channel 0 = Actin
    if img.ndim == 3 and img.shape[2] >= 1:
        actin_channel = img[..., 0]
    elif img.ndim == 2:
        # Fallback if passed a single channel image
        actin_channel = img
    else:
        return 0.0

    # 2. Preprocessing and Quantization
    # Haralick features are sensitive to the number of gray levels. 
    # Using full 256 levels on 512x512 images can result in sparse matrices.
    # We quantize to 64 levels for robust texture statistics.
    n_levels = 64
    
    # Normalize to [0, 1] first to handle potential float inputs or different ranges
    if actin_channel.dtype != np.uint8:
        min_val, max_val = np.min(actin_channel), np.max(actin_channel)
        if max_val > min_val:
            actin_channel = (actin_channel - min_val) / (max_val - min_val)
        else:
            actin_channel = np.zeros_like(actin_channel)
    else:
        actin_channel = actin_channel.astype(np.float32) / 255.0
        
    # Quantize to integer levels [0, n_levels-1]
    # We use floor to map 0.0-0.99... to bins, and clip 1.0 to n_levels-1
    quantized_img = (actin_channel * n_levels).astype(np.uint8)
    quantized_img = np.clip(quantized_img, 0, n_levels - 1)

    # 3. Mask Handling (ROI Selection)
    # If segmentation masks are provided, we want to compute texture ONLY within the cells.
    # We create a combined mask of all valid cellular regions.
    mask = None
    if segmentation_masks:
        # Combine all masks (logical OR)
        combined_mask = np.zeros(actin_channel.shape, dtype=bool)
        for m in segmentation_masks:
            if m is not None:
                # Ensure mask matches image shape (handle potential 2D vs 3D mismatch if any)
                if m.shape == actin_channel.shape:
                    combined_mask = np.logical_or(combined_mask, m > 0)
        
        if np.any(combined_mask):
            mask = combined_mask

    # 4. GLCM Computation
    # We compute GLCM for 4 directions (0, 45, 90, 135 degrees) at distance 1
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    if mask is not None:
        # MASKING TRICK for GLCM:
        # 1. Set background pixels (outside mask) to a value that is effectively ignored or handled.
        #    However, standard graycomatrix doesn't support a mask directly.
        #    A common approach is to set background to 0, compute GLCM, and then zero out the (0,0) entry.
        #    But since 0 is a valid gray level for the object, we shift object levels by +1.
        
        # Shift valid pixels to range [1, n_levels]
        # Background remains 0
        masked_img = np.zeros_like(quantized_img)
        masked_img[mask] = quantized_img[mask] + 1
        
        # Compute GLCM with n_levels + 1 (0 is background, 1..64 are data)
        glcm = graycomatrix(masked_img, distances=distances, angles=angles, 
                            levels=n_levels + 1, symmetric=True, normed=False)
        
        # Remove the background interaction (index 0)
        # We ignore row 0 and col 0 entirely, effectively slicing out the [1:, 1:] submatrix
        glcm_roi = glcm[1:, 1:, :, :]
        
        # If the ROI is empty after slicing (no valid pixels), return 0
        if np.sum(glcm_roi) == 0:
            return 0.0
            
        # Normalize manually
        glcm_norm = glcm_roi / np.sum(glcm_roi, axis=(0, 1), keepdims=True)
        
    else:
        # No mask: compute on whole image
        glcm = graycomatrix(quantized_img, distances=distances, angles=angles, 
                            levels=n_levels, symmetric=True, normed=True)
        glcm_norm = glcm

    # 5. Entropy Calculation
    # Entropy = - sum(p * log2(p))
    # Add epsilon to avoid log(0)
    epsilon = 1e-10
    
    # Compute entropy for each angle separately
    # glcm_norm shape: (levels, levels, num_distances, num_angles)
    # We iterate over the last two dimensions (distances=1, angles=4)
    entropies = []
    
    for d_idx in range(len(distances)):
        for a_idx in range(len(angles)):
            P = glcm_norm[:, :, d_idx, a_idx]
            # Only consider non-zero probabilities for entropy
            mask_p = P > 0
            if np.any(mask_p):
                entropy = -np.sum(P[mask_p] * np.log2(P[mask_p] + epsilon))
                entropies.append(entropy)
            else:
                entropies.append(0.0)

    # Return the mean entropy across all directions (rotational invariance)
    if not entropies:
        return 0.0
        
    return float(np.mean(entropies))
