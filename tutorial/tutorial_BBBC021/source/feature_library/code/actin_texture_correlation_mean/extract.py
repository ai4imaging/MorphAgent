def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.util import img_as_ubyte
    import warnings

    # Suppress warnings that might arise from empty slices or precision issues
    warnings.filterwarnings("ignore")

    # 1. Data Validation and Preparation
    # Ensure image is numpy array
    img = np.asarray(img)

    # Check dimensionality: Expecting (H, W, C) or (H, W)
    # Dataset description says (512, 512, 3)
    if img.ndim == 3:
        if img.shape[2] >= 1:
            # Channel 0 is Actin (Red) according to dataset description
            actin_channel = img[:, :, 0]
        else:
            return 0.0
    elif img.ndim == 2:
        # Fallback if passed a single channel image
        actin_channel = img
    else:
        return 0.0

    # Ensure uint8 for GLCM calculation (required by skimage.feature.graycomatrix)
    # If float, normalize and convert. If integer but not uint8, cast or rescale.
    if actin_channel.dtype != np.uint8:
        # Normalize to 0-1 then convert to uint8
        min_val = np.min(actin_channel)
        max_val = np.max(actin_channel)
        if max_val > min_val:
            norm_img = (actin_channel - min_val) / (max_val - min_val)
        else:
            norm_img = np.zeros_like(actin_channel)
        actin_channel = img_as_ubyte(norm_img)

    # 2. Segmentation Handling
    # We need to identify cell regions to compute texture within cells, not background.
    labeled_mask = None
    
    # Check if valid segmentation masks are provided
    if len(segmentation_masks) > 0:
        # Iterate through masks to find a suitable one (prefer cell/cytoplasm over nuclei)
        # Assuming masks are passed. We'll take the first non-empty one that looks like a cell mask.
        # If multiple masks exist, usually the larger one covers the cell body.
        
        # Simple heuristic: use the first mask provided.
        # In many pipelines, mask 0 is often the primary object (cell or nuclei).
        # Ideally, we want the whole cell mask for actin.
        candidate_mask = segmentation_masks[0]
        
        if candidate_mask is not None and candidate_mask.size == actin_channel.size:
             # Ensure mask shape matches image (handle potential squeezing)
            if candidate_mask.ndim == actin_channel.ndim and candidate_mask.shape == actin_channel.shape:
                labeled_mask = candidate_mask.astype(int)
            elif candidate_mask.ndim == 2 and actin_channel.ndim == 2:
                 labeled_mask = candidate_mask.astype(int)

    # Fallback: Generate mask if none provided or invalid
    if labeled_mask is None:
        try:
            # Simple Otsu thresholding on the actin channel to find foreground
            # Gaussian blur slightly to reduce noise before thresholding
            from scipy.ndimage import gaussian_filter
            blurred = gaussian_filter(actin_channel.astype(float), sigma=2)
            thresh = threshold_otsu(blurred)
            binary_mask = blurred > thresh
            # Label connected components
            labeled_mask = label(binary_mask)
        except Exception:
            # If thresholding fails (e.g., constant image), return 0
            return 0.0

    # 3. Feature Extraction: Haralick Correlation
    # We compute this per cell and average it.
    
    props = regionprops(labeled_mask, intensity_image=actin_channel)
    
    if not props:
        return 0.0

    correlations = []

    # GLCM parameters
    # Distance 1, Angles 0, 45, 90, 135 degrees (isotropic approximation)
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    levels = 256

    for prop in props:
        # Extract the bounding box of the intensity image (Actin)
        # This is much faster than computing GLCM on the whole image
        roi = prop.image_intensity
        
        # Skip very small regions where texture is meaningless
        if roi.shape[0] < 2 or roi.shape[1] < 2:
            continue

        # Ensure ROI is uint8 (it should be, but safety check)
        if roi.dtype != np.uint8:
            roi = img_as_ubyte(roi)

        try:
            # Compute GLCM
            # We use the rectangular bounding box. 
            # Note: Background pixels (0) in the bounding box but outside the cell mask 
            # will contribute to the texture. Masking inside GLCM is non-trivial in skimage 
            # without custom implementation. Bounding box is standard approximation.
            glcm = graycomatrix(roi, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
            
            # Compute Correlation
            # Returns shape (len(distances), len(angles)) -> (1, 4)
            corr_values = graycoprops(glcm, 'correlation')
            
            # Average across the 4 directions to get a rotation-invariant scalar for this cell
            mean_corr_cell = np.mean(corr_values)
            
            # Check for NaN (can happen if region is perfectly constant intensity)
            if not np.isnan(mean_corr_cell):
                correlations.append(mean_corr_cell)
                
        except Exception:
            continue

    # 4. Aggregation
    if not correlations:
        return 0.0

    # Return the mean correlation across all valid cells
    result = np.mean(correlations)
    
    return float(result)
