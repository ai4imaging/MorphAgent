def extract(img, *segmentation_masks):
    import numpy as np
    from scipy import stats
    from skimage.filters import threshold_otsu

    # 1. Input Validation and Preparation
    # Convert to float32 for calculations
    img = np.asarray(img, dtype=np.float32)

    # Check dimensionality and extract Actin channel (Channel 0)
    # Expected shape: (H, W, 3) or (H, W) if single channel
    if img.ndim == 3 and img.shape[2] >= 1:
        # Channel 0 is Actin (Red)
        actin_img = img[:, :, 0]
    elif img.ndim == 2:
        # Fallback if passed as single channel
        actin_img = img
    else:
        return 0.0

    # 2. Mask Handling
    # We need to define regions of interest (cells) to compute kurtosis per cell.
    # If we include the black background, kurtosis will be dominated by the background peak.
    
    labels = None
    
    # Check if valid segmentation masks are provided
    if len(segmentation_masks) > 0:
        # Iterate through masks to find a suitable one (preferring cell/cytoplasm over nuclei)
        # We assume masks are passed in order. If multiple, we might get nuclei then cells.
        # We'll try to use the last one assuming it might be the whole cell or cytoplasm, 
        # but any labeled mask is better than none.
        for mask in segmentation_masks:
            if mask is not None and mask.shape == actin_img.shape:
                labels = mask
                # If we find a mask with a significant number of labels, use it.
                if np.max(labels) > 0:
                    break
    
    # Fallback: Generate a mask if none provided
    if labels is None:
        # Create a simple foreground mask using Otsu thresholding on the actin channel itself
        # This is a "global" analysis if no instance segmentation is available.
        try:
            thresh = threshold_otsu(actin_img)
            # Create a binary mask
            binary_mask = actin_img > thresh
            # Label connected components to treat distinct regions as "cells"
            from skimage.measure import label
            labels = label(binary_mask)
        except Exception:
            # If thresholding fails (e.g. constant image), return 0.0
            return 0.0

    # 3. Feature Computation: Kurtosis per object
    kurtosis_values = []
    
    # Get unique labels (excluding background 0)
    unique_labels = np.unique(labels)
    if unique_labels[0] == 0:
        unique_labels = unique_labels[1:]
        
    if len(unique_labels) == 0:
        return 0.0

    # Iterate over each cell
    for label_id in unique_labels:
        # Extract pixels for this cell
        cell_pixels = actin_img[labels == label_id]
        
        # Skip artifacts that are too small for statistical stability
        if cell_pixels.size < 10:
            continue
            
        # Check for zero variance (constant intensity)
        # Kurtosis is undefined if variance is 0
        if np.std(cell_pixels) == 0:
            kurtosis_values.append(0.0) # Uniform distribution equivalent
            continue

        # Calculate Kurtosis (Fisher's definition: excess kurtosis, normal = 0.0)
        # High kurtosis = distinct high-intensity structures (stress fibers) vs background
        # Low kurtosis = diffuse/uniform texture
        k = stats.kurtosis(cell_pixels, fisher=True, bias=False)
        
        if not np.isnan(k) and not np.isinf(k):
            kurtosis_values.append(k)

    # 4. Aggregation
    if not kurtosis_values:
        return 0.0
        
    # Return the mean kurtosis across all cells
    result = np.mean(kurtosis_values)
    
    return float(result)
