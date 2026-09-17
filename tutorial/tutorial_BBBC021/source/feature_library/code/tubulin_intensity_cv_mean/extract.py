def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label
    from skimage.morphology import dilation, disk
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type and handle dimensionality
    # Dataset is (512, 512, 3), uint8
    # Channel 1 is Tubulin (Green)
    img_arr = np.asarray(img, dtype=np.float32)
    
    # Check dimensionality
    if img_arr.ndim != 3 or img_arr.shape[2] != 3:
        return 0.0

    # Extract Tubulin channel (Channel 1)
    tubulin_channel = img_arr[:, :, 1]

    # Normalize intensity to avoid overflow and ensure consistent scale
    # Although CV is scale-invariant (ratio), working in float is safer.
    # We do NOT normalize to [0,1] for the calculation itself because 
    # if we shift the mean (e.g. by subtracting min), we change the CV.
    # However, raw uint8 values often have a background offset. 
    # A high background increases the Mean, lowering the CV artificially.
    # We will perform a mild background subtraction to make the CV sensitive to structure.
    bg_val = np.percentile(tubulin_channel, 1)
    tubulin_corrected = np.maximum(tubulin_channel - bg_val, 0)

    # Determine Cell Masks
    # We need to identify individual cells to calculate "per cell" statistics.
    labels = None
    
    # Check if valid segmentation masks are provided
    # The system might pass multiple masks. We prioritize a cell mask if available.
    # If masks are provided, we assume they are label masks (0=bg, 1..N=cells)
    if len(segmentation_masks) > 0:
        # Heuristic: Try to find a mask that covers a significant portion of the image 
        # but isn't just small dots (nuclei).
        # If multiple masks, usually the order is specific, but here we inspect.
        
        # Let's try to use the first mask as the primary label source.
        # If it's a nuclei mask (small objects), we might want to dilate it.
        # If it's a cell mask (large objects), we use it directly.
        candidate_mask = segmentation_masks[0]
        
        if candidate_mask is not None and candidate_mask.ndim == 2:
             # Ensure it's integer labeled
            if np.issubdtype(candidate_mask.dtype, np.integer):
                labels = candidate_mask
            else:
                # If boolean or float, label it
                labels = label(candidate_mask > 0)
    
    # Fallback: If no masks provided, generate a mask from the image itself
    if labels is None:
        # 1. Threshold the tubulin channel to find foreground
        try:
            thresh = threshold_otsu(tubulin_channel)
            binary_mask = tubulin_channel > thresh
        except Exception:
            # Fallback for very low contrast images
            binary_mask = tubulin_channel > np.mean(tubulin_channel)
            
        # 2. Label connected components
        # This treats touching cells as one, but it's the best we can do without a mask
        labels = label(binary_mask)

    # If still no labels (empty image), return 0
    if labels.max() == 0:
        return 0.0

    # Compute Statistics per Cell
    # We need Mean and Standard Deviation of intensity for each labeled region
    
    # Get unique labels (excluding background 0)
    unique_labels = np.unique(labels)
    if unique_labels[0] == 0:
        unique_labels = unique_labels[1:]
        
    if len(unique_labels) == 0:
        return 0.0

    # Optimization: Use scipy.ndimage for fast C-based aggregation
    # index argument expects a sequence of labels to compute over
    means = ndimage.mean(tubulin_corrected, labels=labels, index=unique_labels)
    stds = ndimage.standard_deviation(tubulin_corrected, labels=labels, index=unique_labels)

    # Calculate Coefficient of Variation (CV) = std / mean
    # Handle division by zero (if mean is 0)
    # We use a small epsilon for stability, though with corrected data mean could be 0
    epsilon = 1e-6
    
    # Filter out invalid cells (e.g., single pixel cells where std is 0, or empty regions)
    valid_indices = means > epsilon
    
    if not np.any(valid_indices):
        return 0.0
        
    valid_means = means[valid_indices]
    valid_stds = stds[valid_indices]
    
    cv_values = valid_stds / valid_means

    # The feature is the MEAN of the CVs across all cells
    result = np.mean(cv_values)

    return float(result)
