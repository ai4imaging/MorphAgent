def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    
    # 1. Input Validation and Channel Selection
    # Ensure image is numpy array
    img = np.asarray(img)
    
    # Check dimensions. Expected (512, 512, 3)
    if img.ndim != 3 or img.shape[2] != 3:
        return 0.0
        
    # Extract Tubulin Channel (Channel 1 - Green)
    # The dataset description specifies: Ch0=Actin, Ch1=Tubulin, Ch2=DAPI
    tubulin_channel = img[:, :, 1]
    
    # Ensure uint8 for GLCM (0-255)
    if tubulin_channel.dtype != np.uint8:
        # Normalize to 0-255 if not already
        min_val = np.min(tubulin_channel)
        max_val = np.max(tubulin_channel)
        if max_val > min_val:
            tubulin_channel = ((tubulin_channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            tubulin_channel = tubulin_channel.astype(np.uint8)

    # 2. Mask Handling
    # We need a mask to define cell boundaries to compute texture *inside* cells.
    mask = None
    
    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0:
        # Use the first available mask. 
        # In many datasets, if multiple masks exist, one might be nuclei and another cell body.
        # We prefer the largest coverage for tubulin (cytoskeleton), so we iterate to find a suitable one.
        # If we have labeled masks, we convert to boolean for ROI extraction.
        candidate_mask = segmentation_masks[0]
        if candidate_mask is not None and candidate_mask.shape == tubulin_channel.shape:
            mask = candidate_mask > 0
    
    # Fallback: If no mask provided, generate one using Otsu thresholding on the tubulin channel
    if mask is None:
        try:
            thresh = threshold_otsu(tubulin_channel)
            mask = tubulin_channel > thresh
        except Exception:
            # If image is uniform (e.g. all black), otsu fails
            return 0.0

    # 3. Object Identification
    # Label the mask to process individual cells
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask, intensity_image=tubulin_channel)
    
    if not regions:
        return 0.0

    # 4. Feature Computation: GLCM Contrast
    # We compute GLCM per cell to avoid texture artifacts from the black background between cells.
    
    contrast_values = []
    
    # GLCM Parameters
    distances = [1] # 1 pixel offset
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/2] # 0, 45, 90, 135 degrees
    levels = 256 # For uint8
    
    for region in regions:
        # Skip very small artifacts
        if region.area < 50:
            continue
            
        # Extract the bounding box of the cell intensity
        # region.image is the binary mask of the cell within the bbox
        # region.intensity_image is the intensity image within the bbox
        
        # We need to ensure we only calculate texture on the cell pixels, not the bbox background.
        # Strategy:
        # 1. Get the intensity crop.
        # 2. Mask out the background (set to 0).
        # 3. Compute GLCM.
        # 4. IGNORE the 0-index row and column in the GLCM (which represents background interactions).
        
        cell_crop = region.intensity_image.copy()
        # Ensure background in the crop is 0 (it should be by definition of regionprops with intensity_image, 
        # but region.intensity_image keeps original pixels inside bbox. We must apply the mask.)
        cell_crop[~region.image] = 0
        
        # Compute GLCM
        # We use the crop which is uint8. 0 is background.
        try:
            glcm = graycomatrix(cell_crop, distances=distances, angles=angles, levels=levels, symmetric=True, normed=False)
        except ValueError:
            continue

        # CRITICAL STEP: Remove Background Artifacts
        # The transition from cell pixel (Value X) to background (Value 0) creates high contrast
        # at the cell border. We want internal texture only.
        # The GLCM row 0 and col 0 correspond to pixel value 0 (background).
        # We set these to 0 to ignore any pairing involving the background.
        glcm[0, :, :, :] = 0
        glcm[:, 0, :, :] = 0
        
        # Re-normalize the GLCM since we removed counts
        glcm_sum = np.sum(glcm, axis=(0, 1))
        
        # Avoid division by zero if a cell has no internal texture (e.g. single pixel width)
        # glcm_sum shape is (len(distances), len(angles)) -> (1, 4)
        valid_indices = glcm_sum > 0
        
        if np.any(valid_indices):
            # Normalize valid GLCMs
            glcm_norm = np.zeros_like(glcm, dtype=np.float64)
            # Broadcasting normalization
            for d_idx in range(len(distances)):
                for a_idx in range(len(angles)):
                    if glcm_sum[d_idx, a_idx] > 0:
                        glcm_norm[:, :, d_idx, a_idx] = glcm[:, :, d_idx, a_idx] / glcm_sum[d_idx, a_idx]
            
            # Compute Contrast
            # Contrast = sum_{i,j} (i-j)^2 * p(i,j)
            contrasts = graycoprops(glcm_norm, 'contrast')
            
            # Average contrast across all valid angles for this cell
            # contrasts shape is (1, 4)
            mean_cell_contrast = np.mean(contrasts)
            contrast_values.append(mean_cell_contrast)

    # 5. Aggregation
    if not contrast_values:
        return 0.0
        
    # Return the mean contrast across all cells in the image
    return float(np.mean(contrast_values))
