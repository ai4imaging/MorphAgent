def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.morphology import skeletonize
    from skimage.filters import threshold_otsu
    
    # 1. Data Loading and Validation
    # Convert to float32 for processing
    arr = np.asarray(img, dtype=np.float32)
    
    # Check dimensionality: Expecting (H, W, C) where C=3
    # If unexpected shape, return 0.0
    if arr.ndim != 3 or arr.shape[2] < 2:
        return 0.0
        
    # Extract Tubulin Channel (Channel 1 - Green)
    tubulin_channel = arr[..., 1]
    
    # Normalize to [0, 1]
    # Check for empty or black image
    if np.max(tubulin_channel) == 0:
        return 0.0
        
    tubulin_norm = tubulin_channel / 255.0
    
    # 2. Define Region of Interest (ROI) / Cell Mask
    # The density should be calculated relative to the cell area, not the whole image.
    cell_mask = None
    
    if len(segmentation_masks) > 0:
        # If segmentation masks are provided, combine them to define the cellular area
        # Masks are typically labeled integers. Convert to binary.
        combined_mask = np.zeros(tubulin_channel.shape, dtype=bool)
        for mask in segmentation_masks:
            if mask is not None and mask.shape == tubulin_channel.shape:
                combined_mask = combined_mask | (mask > 0)
        
        if np.any(combined_mask):
            cell_mask = combined_mask

    # Fallback: If no masks provided or masks are empty, generate a mask from the image
    if cell_mask is None:
        # Use a low threshold on the smoothed image to find general cell body
        # Gaussian blur to reduce noise for mask generation
        smooth_for_mask = ndimage.gaussian_filter(tubulin_norm, sigma=2.0)
        try:
            # Simple thresholding strategy: > mean + small factor, or Otsu
            # Here we use a safe fallback if Otsu fails on uniform images
            thresh_val = threshold_otsu(smooth_for_mask)
            cell_mask = smooth_for_mask > thresh_val
        except Exception:
            # Fallback for extremely low contrast images
            cell_mask = smooth_for_mask > np.mean(smooth_for_mask)

    # Calculate Cell Area (Denominator)
    cell_area = np.sum(cell_mask)
    
    # If no cells detected, return 0.0
    if cell_area == 0:
        return 0.0

    # 3. Skeletonization Pipeline
    # A. Pre-processing: Gaussian blur to smooth fibers and reduce noise artifacts
    # Sigma=1.0 is usually good for preserving fiber structures while removing pixel noise
    tubulin_smooth = ndimage.gaussian_filter(tubulin_norm, sigma=1.0)
    
    # B. Thresholding to identify fiber foreground
    try:
        # Calculate Otsu threshold specifically within the cell mask if possible
        # to avoid background noise influencing the threshold
        pixels_in_cells = tubulin_smooth[cell_mask]
        if pixels_in_cells.size > 0:
            thresh = threshold_otsu(pixels_in_cells)
        else:
            thresh = threshold_otsu(tubulin_smooth)
            
        binary_tubulin = tubulin_smooth > thresh
    except Exception:
        # Fallback if thresholding fails
        binary_tubulin = tubulin_smooth > np.mean(tubulin_smooth)
        
    # Restrict binary tubulin to the cell mask area
    binary_tubulin = binary_tubulin & cell_mask
    
    # C. Skeletonize
    # This reduces the binary fibers to 1-pixel wide lines
    skeleton = skeletonize(binary_tubulin)
    
    # 4. Feature Calculation
    # Calculate total length of skeleton (number of pixels)
    skeleton_pixels = np.sum(skeleton)
    
    # Calculate density: Skeleton Pixels / Cell Area
    # This represents the "fiber length per unit of cytoplasm"
    density = skeleton_pixels / cell_area
    
    return float(density)
