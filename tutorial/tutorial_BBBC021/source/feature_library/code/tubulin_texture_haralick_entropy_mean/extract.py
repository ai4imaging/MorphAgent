def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_opening, disk

    # 1. Data Loading and Validation
    # Ensure image is uint8 for GLCM calculation (0-255 levels)
    # If float, scale to 0-255. If uint16, rescale.
    # The dataset description says input is uint8, but we handle robustness.
    img_arr = np.asarray(img)
    
    # Handle dimensions
    if img_arr.ndim == 3:
        if img_arr.shape[-1] == 3:
            # (H, W, C) - Standard format
            # Channel 1 is Tubulin (Green)
            tubulin = img_arr[..., 1]
        elif img_arr.shape[0] == 3:
            # (C, H, W) - Possible alternative
            tubulin = img_arr[1, ...]
        else:
            # Fallback: take mean or first channel if structure is weird
            tubulin = img_arr[..., 0]
    elif img_arr.ndim == 2:
        # Grayscale image, assume it's the channel of interest
        tubulin = img_arr
    else:
        return 0.0

    # Ensure uint8 for graycomatrix
    if tubulin.dtype != np.uint8:
        # Normalize to 0-1 then scale to 0-255
        min_val = np.min(tubulin)
        max_val = np.max(tubulin)
        if max_val > min_val:
            tubulin = ((tubulin - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            tubulin = np.zeros_like(tubulin, dtype=np.uint8)

    # 2. Mask Handling
    # We need to identify individual cells to compute per-cell texture
    cell_labels = None
    
    if segmentation_masks and len(segmentation_masks) > 0:
        # Use the first available mask
        mask_input = segmentation_masks[0]
        
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[-2:] == tubulin.shape[-2:]:
             # Handle (C, H, W) or (1, H, W) mask
             mask_2d = mask_input.squeeze()
        elif mask_input.shape[:2] == tubulin.shape[:2]:
             # Handle (H, W) or (H, W, 1) mask
             mask_2d = mask_input if mask_input.ndim == 2 else mask_input[..., 0]
        else:
             # Dimension mismatch fallback
             mask_2d = None

        if mask_2d is not None:
            # If mask is already labeled (int > 1), use it. If binary, label it.
            if np.max(mask_2d) > 1:
                cell_labels = mask_2d.astype(int)
            elif np.max(mask_2d) == 1:
                cell_labels = label(mask_2d)
            else:
                # Empty mask
                cell_labels = None

    # Fallback: Generate mask if none provided or invalid
    if cell_labels is None:
        try:
            # Simple background segmentation on Tubulin channel
            # Otsu thresholding
            thresh = threshold_otsu(tubulin)
            binary = tubulin > thresh
            # Clean up noise
            binary = binary_opening(binary, disk(2))
            cell_labels = label(binary)
        except Exception:
            # If thresholding fails (e.g. uniform image), return 0
            return 0.0

    # 3. Feature Extraction: Haralick Entropy
    # We compute GLCM for each cell individually to avoid texture artifacts from cell boundaries/background
    
    props = regionprops(cell_labels, intensity_image=tubulin)
    
    if not props:
        return 0.0

    entropies = []
    
    # GLCM parameters
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4] # 0, 45, 90, 135 degrees
    levels = 256

    for prop in props:
        # Extract the bounding box of the cell
        minr, minc, maxr, maxc = prop.bbox
        
        # Get the sub-image for the cell
        cell_patch = tubulin[minr:maxr, minc:maxc]
        
        # Get the mask for the cell within the bounding box
        # prop.image is the binary mask of the cell in the bbox
        cell_mask = prop.image
        
        # We only want to compute texture on the cell pixels.
        # graycomatrix computes on the rectangular array.
        # Strategy: Mask out background pixels by setting them to 0.
        # Note: This introduces a large peak at (0,0) in the GLCM (background-background).
        # We must handle this during entropy calculation.
        masked_patch = cell_patch.copy()
        masked_patch[~cell_mask] = 0
        
        # Compute GLCM
        # symmetric=True, normed=True
        glcm = graycomatrix(masked_patch, distances=distances, angles=angles, 
                            levels=levels, symmetric=True, normed=True)
        
        # glcm shape: (levels, levels, num_distances, num_angles)
        
        # Calculate Entropy manually
        # Entropy = - sum(p * log2(p))
        
        # To avoid the background influence (the massive count of 0-0 pairs outside the cell),
        # we can zero out the [0,0] entry of the GLCM and re-normalize.
        # This focuses the texture metric on the internal structure and the boundary contrast.
        
        # Iterate over angles to compute mean entropy for this cell
        cell_angle_entropies = []
        for angle_idx in range(len(angles)):
            P = glcm[:, :, 0, angle_idx]
            
            # Remove background-background interactions (index 0,0)
            # This is critical because the rectangular bbox contains many 0s that are not part of the cell
            # However, pixels inside the cell might genuinely be 0.
            # A safer approach for "texture within ROI" is usually to ignore the 0-0 transition 
            # if 0 represents the masked background.
            
            # Since we masked with 0, P[0,0] is dominated by the mask.
            # We set P[0,0] to 0 and re-normalize.
            P_copy = P.copy()
            P_copy[0, 0] = 0
            
            sum_p = np.sum(P_copy)
            if sum_p > 0:
                P_norm = P_copy / sum_p
                
                # Compute entropy on non-zero elements
                mask_p = P_norm > 0
                entropy = -np.sum(P_norm[mask_p] * np.log2(P_norm[mask_p]))
                cell_angle_entropies.append(entropy)
            else:
                cell_angle_entropies.append(0.0)
        
        # Average over angles for this cell
        if cell_angle_entropies:
            entropies.append(np.mean(cell_angle_entropies))

    # 4. Aggregation
    if not entropies:
        return 0.0
        
    result = np.mean(entropies)
    
    return float(result)
