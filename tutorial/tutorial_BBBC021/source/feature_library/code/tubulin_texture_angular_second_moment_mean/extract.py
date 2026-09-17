def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix
    from skimage.measure import regionprops, label
    from skimage.filters import threshold_otsu

    # 1. Input Validation and Channel Selection
    # Dataset: (512, 512, 3), Channel 1 is Tubulin (Green)
    # Handle potential dimensionality variations
    if img.ndim == 3 and img.shape[2] >= 2:
        tubulin = img[:, :, 1]
    elif img.ndim == 2:
        tubulin = img
    else:
        # Unexpected format
        return 0.0

    # 2. Quantization
    # GLCM is computationally expensive and sparse on 256 levels.
    # Binning to 64 levels is standard for texture analysis to ensure statistical stability.
    n_levels = 64
    
    # Ensure input is uint8 for bitwise operations or simple division
    if tubulin.dtype != np.uint8:
        # Normalize to 0-255 if not uint8
        min_val, max_val = tubulin.min(), tubulin.max()
        if max_val > min_val:
            tubulin = ((tubulin - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            tubulin = np.zeros_like(tubulin, dtype=np.uint8)
            
    # Quantize 0-255 -> 0-63
    # We use integer division. 256 / 64 = 4.
    img_quantized = (tubulin // (256 // n_levels)).astype(np.uint8)

    # 3. Mask Selection
    selected_mask = None
    
    # Try to find a valid segmentation mask
    if segmentation_masks:
        best_area = -1
        for m in segmentation_masks:
            if m is None:
                continue
            # Handle potential extra dimensions in masks (e.g. 3D or channel dim)
            if m.ndim > 2:
                m = np.max(m, axis=-1) # Flatten
            
            # Ensure mask matches image spatial dimensions
            if m.shape[:2] != tubulin.shape[:2]:
                continue
                
            # Heuristic: Choose the mask with the largest foreground area
            # This prefers whole-cell masks over nuclei masks for tubulin (cytoskeleton) analysis
            foreground_area = np.count_nonzero(m)
            if foreground_area > best_area:
                best_area = foreground_area
                selected_mask = m

    # Fallback: Generate mask if none provided
    if selected_mask is None:
        try:
            # Otsu thresholding to separate foreground
            thresh = threshold_otsu(tubulin)
            binary_mask = tubulin > thresh
            selected_mask = label(binary_mask)
        except Exception:
            # If thresholding fails (e.g. constant image), treat whole image as one object
            selected_mask = np.ones_like(tubulin, dtype=np.int32)

    # 4. Compute Feature (ASM) per Cell
    # We use regionprops to iterate over identified cells
    props = regionprops(selected_mask.astype(int), intensity_image=img_quantized)
    
    asm_values = []
    
    for prop in props:
        # Extract the bounding box of the cell
        # prop.intensity_image is the quantized tubulin within the bbox
        roi_intensity = prop.intensity_image
        # prop.image is the binary mask of the cell within the bbox
        roi_mask = prop.image
        
        # Skip very small regions that can't support GLCM window
        if roi_intensity.size < 4 or roi_intensity.shape[0] < 2 or roi_intensity.shape[1] < 2:
            continue

        # Prepare ROI for GLCM
        # We shift values by +1 (range 1-64) and set background to 0
        # This allows us to compute GLCM on the rectangular bbox and ignore background transitions later
        roi_prepared = (roi_intensity + 1).astype(np.uint8)
        roi_prepared[~roi_mask] = 0
        
        # Compute GLCM
        # Levels=65 (0 for background, 1-64 for signal)
        # Distances=[1] (1 pixel adjacency)
        # Angles=[0, 45, 90, 135] degrees (average for rotation invariance)
        try:
            glcm = graycomatrix(roi_prepared, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                                levels=n_levels+1, symmetric=True, normed=False)
        except ValueError:
            continue

        # Slice the GLCM to remove background (row 0 and col 0)
        # We only want transitions between signal pixels (1-64)
        # glcm shape: (65, 65, 1, 4)
        glcm_signal = glcm[1:, 1:, :, :]
        
        # Normalize the sub-matrix to make it a probability distribution
        glcm_sum = np.sum(glcm_signal, axis=(0, 1), keepdims=True)
        
        # Avoid division by zero (if a cell has no internal transitions)
        valid_sums = glcm_sum > 0
        if not np.any(valid_sums):
            continue
            
        # Normalize
        glcm_normed = np.divide(glcm_signal, glcm_sum, where=valid_sums)
        
        # Compute ASM: Sum of squared probabilities
        # ASM = sum(p_ij ^ 2)
        asm_per_angle = np.sum(glcm_normed**2, axis=(0, 1))
        
        # Average ASM over the 4 directions for rotation invariance
        mean_asm_cell = np.mean(asm_per_angle)
        asm_values.append(mean_asm_cell)

    # 5. Aggregate and Return
    if not asm_values:
        return 0.0
        
    return float(np.mean(asm_values))
