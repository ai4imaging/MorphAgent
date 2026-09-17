def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_dilation, disk

    # 1. Data Loading and Validation
    # Ensure image is the correct shape and type
    img = np.asarray(img)
    
    # Handle dimensions: Expecting (H, W, C) or (H, W)
    if img.ndim == 3:
        # Channel 0 is Actin (Red) based on dataset description
        actin_channel = img[..., 0]
        # Channel 2 is DAPI (Blue) for fallback nuclear segmentation
        dapi_channel = img[..., 2] if img.shape[2] > 2 else None
    elif img.ndim == 2:
        # If 2D, assume it's already the channel of interest or a projection
        actin_channel = img
        dapi_channel = None
    else:
        return 0.0

    # 2. Define Region of Interest (Cytoplasm)
    # We need to define the cytoplasm mask: Cell Mask - Nuclear Mask
    
    cytoplasm_mask = None
    
    # Strategy A: Use provided segmentation masks
    if len(segmentation_masks) >= 2:
        # Usually, masks are passed. We need to identify which is which.
        # Heuristic: Nuclei are usually smaller and contained within cells.
        # Or based on typical order: often (nuclei, cells) or (cells, nuclei).
        # Let's try to distinguish by area or containment.
        
        mask1 = segmentation_masks[0]
        mask2 = segmentation_masks[1]
        
        # Ensure masks are boolean/binary
        m1_bool = mask1 > 0
        m2_bool = mask2 > 0
        
        area1 = np.sum(m1_bool)
        area2 = np.sum(m2_bool)
        
        # Assume the larger mask is the whole cell, smaller is nucleus
        if area1 > area2:
            cell_mask = m1_bool
            nuc_mask = m2_bool
        else:
            cell_mask = m2_bool
            nuc_mask = m1_bool
            
        cytoplasm_mask = np.logical_and(cell_mask, np.logical_not(nuc_mask))

    elif len(segmentation_masks) == 1:
        # Only one mask provided. 
        # If it's a cell mask, we need to generate a nuclear mask to subtract.
        # If it's a nuclear mask, we need to generate a cell mask (or just use the area around it).
        # Let's assume it's a cell/nuclei label map.
        
        # Fallback: Generate masks from image channels if only partial or no masks provided
        # This is robust if the provided mask isn't sufficient for "cytoplasm" definition
        pass

    # Strategy B: Fallback if masks are missing or insufficient
    if cytoplasm_mask is None:
        # Generate masks from raw intensity
        # 1. Nuclear mask from DAPI (Channel 2)
        if dapi_channel is not None:
            try:
                thresh_nuc = threshold_otsu(dapi_channel)
                nuc_mask = dapi_channel > thresh_nuc
            except Exception:
                nuc_mask = dapi_channel > np.mean(dapi_channel)
        else:
            # If no DAPI, guess nucleus is high intensity in Actin? Unlikely, but fallback.
            nuc_mask = np.zeros_like(actin_channel, dtype=bool)

        # 2. Cell mask from Actin (Channel 0)
        try:
            thresh_cell = threshold_otsu(actin_channel)
            # Actin often has background noise, so otsu works reasonably well for foreground
            cell_mask = actin_channel > thresh_cell
        except Exception:
            cell_mask = actin_channel > np.mean(actin_channel)
            
        # 3. Cytoplasm = Cell - Nucleus
        cytoplasm_mask = np.logical_and(cell_mask, np.logical_not(nuc_mask))

    # Ensure mask is valid
    if np.sum(cytoplasm_mask) == 0:
        return 0.0

    # 3. Preprocessing for GLCM
    # GLCM requires integer types. 
    # To reduce sparsity and noise, we quantize the image to fewer levels (e.g., 64).
    n_levels = 64
    
    # Normalize actin channel to 0-1 range for quantization
    # Use robust min/max to avoid outliers skewing the bins
    p_low, p_high = np.percentile(actin_channel, (1, 99))
    if p_high - p_low == 0:
        return 0.0 # Flat image, no texture
        
    norm_img = np.clip((actin_channel - p_low) / (p_high - p_low), 0, 1)
    
    # Quantize to 0 -> (n_levels-1)
    # We map valid pixels to 1 -> n_levels. 0 is reserved for background/masked out.
    digitized = (norm_img * (n_levels - 1)).astype(np.uint8) + 1
    
    # Apply mask: Set non-cytoplasm pixels to 0
    digitized[~cytoplasm_mask] = 0

    # 4. Compute GLCM
    # We compute GLCM on the whole image but ignore the background (value 0).
    # Distances: 1 pixel (capture fine actin fibers)
    # Angles: 0, 45, 90, 135 (average for rotation invariance)
    
    # Note: graycomatrix computes the co-occurrence of pixel values.
    # Since we set background to 0, row 0 and col 0 of the GLCM will contain 
    # interactions involving the background. We must exclude these.
    
    try:
        glcm = graycomatrix(digitized, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                            levels=n_levels+1, symmetric=True, normed=False)
    except ValueError:
        return 0.0

    # 5. Extract Texture Feature (Correlation)
    # The GLCM has shape (levels, levels, num_distances, num_angles)
    # We want to exclude index 0 (background).
    # Slice: [1:, 1:, :, :]
    glcm_roi = glcm[1:, 1:, :, :]
    
    # If the ROI is empty (no valid transitions inside cytoplasm), return 0
    if np.sum(glcm_roi) == 0:
        return 0.0
    
    # Normalize the sliced GLCM so it sums to 1 (it becomes a probability matrix)
    # We sum over the first two dimensions (levels x levels) for each angle/distance
    glcm_sums = np.sum(glcm_roi, axis=(0, 1), keepdims=True)
    # Avoid division by zero
    glcm_sums[glcm_sums == 0] = 1
    glcm_norm = glcm_roi / glcm_sums

    # Compute Correlation manually or use skimage's graycoprops on the sliced matrix?
    # graycoprops expects a standard GLCM. Since we sliced it, the levels are now 0 to n_levels-1 relative to the slice.
    # This is mathematically correct for correlation as long as the relative values are linear.
    # However, graycoprops might re-normalize. It's safer to use graycoprops on the un-normalized slice 
    # but graycoprops doesn't accept a probability matrix, it accepts counts.
    # So we pass the raw counts of the slice.
    
    # graycoprops returns (num_distances, num_angles)
    correlation_matrix = graycoprops(glcm_roi, 'correlation')
    
    # 6. Aggregate
    # Average over all angles (and the single distance)
    result = np.mean(correlation_matrix)
    
    # Handle NaN (can happen if variance is 0 in a window)
    if np.isnan(result):
        return 0.0
        
    return float(result)
