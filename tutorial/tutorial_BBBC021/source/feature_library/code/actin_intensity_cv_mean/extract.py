def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.measure import label

    # 1. Input Validation and Preprocessing
    # Check if image is None or empty
    if img is None or img.size == 0:
        return 0.0

    # Handle dimensionality and extract Actin channel
    # Dataset info: (512, 512, 3), Channel 0 = Actin (Red)
    # If the image is 2D (grayscale), assume it's already the channel of interest or a projection
    # If 3D with 3 channels, take index 0.
    if img.ndim == 3 and img.shape[2] == 3:
        actin_img = img[..., 0]  # Channel 0 is Actin
    elif img.ndim == 2:
        actin_img = img
    else:
        # Unexpected shape, return 0.0
        return 0.0

    # Convert to float for statistical calculations to avoid overflow/truncation
    actin_img = actin_img.astype(np.float64)

    # 2. Mask Handling
    # Determine the segmentation mask to use
    labeled_mask = None

    # Check if valid segmentation masks are provided
    if segmentation_masks and len(segmentation_masks) > 0:
        # Use the first available mask
        mask_input = segmentation_masks[0]
        
        # Validate mask shape matches image shape (ignoring channels)
        if mask_input is not None and mask_input.shape == actin_img.shape:
            # If the mask is already labeled (integers > 1), use it directly
            if np.max(mask_input) > 1:
                labeled_mask = mask_input.astype(np.int32)
            else:
                # If binary (0/1), label connected components
                labeled_mask = label(mask_input > 0)
    
    # Fallback: If no valid mask provided, generate one from the image itself
    if labeled_mask is None:
        # Simple background separation using Otsu's method on the actin channel
        # Check if image has variance (if all pixels same, otsu fails)
        if np.min(actin_img) == np.max(actin_img):
            return 0.0
            
        try:
            thresh = threshold_otsu(actin_img)
            binary_mask = actin_img > thresh
            labeled_mask = label(binary_mask)
        except Exception:
            return 0.0

    # 3. Feature Computation: Coefficient of Variation (CV) per cell
    # CV = Standard Deviation / Mean
    
    # Get unique labels (excluding background 0)
    unique_labels = np.unique(labeled_mask)
    unique_labels = unique_labels[unique_labels != 0]

    if len(unique_labels) == 0:
        return 0.0

    # Vectorized calculation using scipy.ndimage
    # This is much faster than iterating through cells in a Python loop
    
    # Calculate mean intensity per cell
    means = ndimage.mean(actin_img, labels=labeled_mask, index=unique_labels)
    
    # Calculate standard deviation of intensity per cell
    stds = ndimage.standard_deviation(actin_img, labels=labeled_mask, index=unique_labels)

    # Calculate CV
    # Handle division by zero: if mean is 0 (unlikely for foreground, but possible), CV is undefined/0
    # We use a small epsilon or mask out zero means
    valid_indices = means > 1e-6
    
    if np.sum(valid_indices) == 0:
        return 0.0
        
    # Filter arrays to only valid cells
    valid_means = means[valid_indices]
    valid_stds = stds[valid_indices]
    
    cv_values = valid_stds / valid_means

    # 4. Aggregation
    # Return the mean CV across all valid cells in the image
    result = np.mean(cv_values)

    return float(result)
