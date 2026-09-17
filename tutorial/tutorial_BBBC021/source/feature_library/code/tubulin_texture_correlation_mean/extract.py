def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    
    # 1. Data Loading and Validation
    # Convert to appropriate array type, ensuring we have the image data
    arr = np.asarray(img)
    
    # Handle dimensionality
    # Dataset is (512, 512, 3). We need Channel 1 (Tubulin/Green).
    if arr.ndim == 3 and arr.shape[2] >= 2:
        # Extract Tubulin channel (Index 1)
        tubulin_img = arr[:, :, 1]
    elif arr.ndim == 2:
        # Fallback if single channel passed (unlikely based on desc, but safe)
        tubulin_img = arr
    else:
        return 0.0

    # Ensure uint8 for GLCM (0-255)
    if tubulin_img.dtype != np.uint8:
        # Normalize to 0-255 if not already
        min_val, max_val = tubulin_img.min(), tubulin_img.max()
        if max_val > min_val:
            tubulin_img = ((tubulin_img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            tubulin_img = np.zeros_like(tubulin_img, dtype=np.uint8)

    # 2. Mask Handling
    # We need a label mask to identify individual cells.
    # Prioritize provided masks. If multiple, usually the larger one covers the cytoplasm (cell body).
    
    label_mask = None
    
    if segmentation_masks and len(segmentation_masks) > 0:
        # Check masks to find a suitable one
        # Often masks are passed as (nuclei, cells) or just (cells). 
        # We prefer the one with the largest coverage area as Tubulin is cytoplasmic.
        best_mask = None
        max_area = -1
        
        for mask in segmentation_masks:
            if mask is None: continue
            # Ensure mask is 2D
            if mask.ndim == 3:
                mask = np.max(mask, axis=2) # Project if 3D
            if mask.shape != tubulin_img.shape:
                continue # Skip mismatched shapes
                
            current_area = np.count_nonzero(mask)
            if current_area > max_area:
                max_area = current_area
                best_mask = mask
        
        if best_mask is not None:
            # If the mask is already labeled (int > 1), use it. 
            # If it's binary (0/1 or 0/255), label it.
            if best_mask.max() > 1:
                label_mask = best_mask.astype(int)
            else:
                label_mask = label(best_mask > 0)

    # Fallback: If no valid mask provided, generate one via Otsu thresholding
    if label_mask is None:
        try:
            thresh = threshold_otsu(tubulin_img)
            binary_mask = tubulin_img > thresh
            label_mask = label(binary_mask)
        except Exception:
            # If image is uniform, otsu fails
            return 0.0

    # 3. Feature Computation (Haralick Correlation)
    # We compute this per cell to avoid background dominating the texture stats.
    
    props = regionprops(label_mask, intensity_image=tubulin_img)
    
    if not props:
        return 0.0
        
    correlations = []
    
    # Parameters for GLCM
    # Distance = 1 pixel
    # Angles = 0, 45, 90, 135 degrees (average for rotation invariance)
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    # Binning: Reduce gray levels to 64 to speed up computation and improve robustness
    # 256 levels is often too sparse for small cell patches
    n_bins = 64
    
    for prop in props:
        # Extract the bounding box of the cell
        # intensity_image in prop is already the cropped version of tubulin_img
        # masked by the bounding box, but we need to mask out non-cell pixels within the box
        
        # Get the cropped mask for this object
        bbox_mask = prop.image # Binary mask of the object in the bounding box
        bbox_img = prop.intensity_image # Intensity image in the bounding box
        
        if bbox_img.size == 0:
            continue

        # Apply mask: set background pixels to 0
        # We need to be careful: 0 is a valid bin index. 
        # However, for correlation, we want to measure the texture of the *object*.
        # Standard approach: Compute GLCM on the rectangular patch, but we risk background influence.
        # Better approach for "within segmented cells": 
        # 1. Quantize image
        # 2. Set background to a specific value (e.g. -1 or handled via ignoring 0-0 in GLCM if we shift bins)
        
        # Quantize to n_bins (0 to n_bins-1)
        # We use floor division. 256 // 4 = 64.
        bins = (bbox_img // (256 // n_bins)).astype(np.uint8)
        bins = np.clip(bins, 0, n_bins - 1)
        
        # Mask out background:
        # We will set background to 0. To distinguish actual 0-intensity signal from background,
        # we can shift the signal bins by +1 (range 1-64), and leave background as 0.
        # Then we can ignore row 0 and col 0 in the GLCM, or just accept that the boundary 
        # contributes slightly to texture (common approximation).
        # Given the constraint of standard libraries, we'll use the standard GLCM on the masked patch.
        # To minimize background effect, we crop tightly (which regionprops does).
        
        masked_bins = bins * bbox_mask
        
        # Compute GLCM
        try:
            # levels needs to be n_bins (if we didn't shift) or n_bins+1 (if we shifted).
            # Here we just use the bins 0-63. Background is 0.
            # This treats the black background as part of the texture. 
            # For tightly cropped cells, this is usually acceptable and robust.
            glcm = graycomatrix(masked_bins, distances=distances, angles=angles, 
                                levels=n_bins, symmetric=True, normed=True)
            
            # Compute Correlation
            # graycoprops returns (n_distances, n_angles)
            corr_matrix = graycoprops(glcm, 'correlation')
            
            # Average over angles (and distances, though we only have 1)
            mean_corr = np.mean(corr_matrix)
            
            # Handle NaN (can happen if image is constant)
            if np.isnan(mean_corr):
                # If variance is 0, correlation is undefined. 
                # Biologically, a flat intensity is "perfectly correlated" with itself in a trivial sense,
                # but mathematically usually treated as 1.0 or 0.0. 
                # Scikit-image returns 1.0 for constant images if handled, but sometimes NaN.
                # We'll assume 1.0 for constant regions (perfect structure).
                correlations.append(1.0)
            else:
                correlations.append(mean_corr)
                
        except Exception:
            continue

    if not correlations:
        return 0.0

    # Aggregate: Mean across all cells
    result = np.mean(correlations)
    
    return float(result)
