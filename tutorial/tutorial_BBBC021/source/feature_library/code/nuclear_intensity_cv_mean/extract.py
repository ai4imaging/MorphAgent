def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage

    # 1. Input Validation and Channel Selection
    # The dataset description specifies:
    # - Dimensions: (512, 512, 3)
    # - Channel 2 (Index 2) is Blue/DAPI (Nucleus)
    
    # Check if image is valid
    if img is None:
        return 0.0
        
    # Handle dimensionality
    # We expect (H, W, C). If 2D (H, W), it might be a single channel image, but dataset says 3 channels.
    img_arr = np.asarray(img)
    
    if img_arr.ndim == 3:
        if img_arr.shape[2] >= 3:
            # Extract DAPI channel (Channel 2)
            dapi_channel = img_arr[:, :, 2]
        else:
            # Fallback if fewer channels than expected, use the last one or average
            dapi_channel = img_arr[:, :, -1]
    elif img_arr.ndim == 2:
        # If already 2D, assume it's the relevant channel (or a projection)
        dapi_channel = img_arr
    else:
        return 0.0

    # Convert to float for statistical calculations to avoid overflow/truncation
    dapi_channel = dapi_channel.astype(np.float64)

    # 2. Handle Segmentation Masks
    # We need a nuclear mask to compute per-nucleus statistics.
    # If no mask is provided, we cannot compute "per segmented nucleus" stats accurately.
    # However, we can try to generate a crude mask or return NaN/0.
    
    labeled_mask = None
    
    if segmentation_masks and len(segmentation_masks) > 0:
        # Use the first available mask. 
        # In this dataset context, masks are usually passed in order, often starting with nuclei or cells.
        # We assume the mask provided corresponds to the objects we want to measure.
        mask_input = segmentation_masks[0]
        
        # Ensure mask matches image dimensions
        if mask_input.shape == dapi_channel.shape:
            labeled_mask = mask_input
        elif mask_input.ndim == 3 and mask_input.shape[:2] == dapi_channel.shape:
             # If mask is 3D (e.g. one-hot or labeled stack), take max projection or first slice
             labeled_mask = np.max(mask_input, axis=2)
        
        # Ensure mask is integer labeled (0=bg, 1..N=objects)
        if labeled_mask is not None:
            labeled_mask = labeled_mask.astype(np.int32)
    
    # If no valid mask is found, we cannot compute the specific feature requested.
    if labeled_mask is None:
        return 0.0

    # 3. Compute Statistics per Nucleus
    # We need Mean and Standard Deviation for each labeled region (excluding background 0)
    
    # Get unique labels (excluding 0)
    # Note: ndimage functions allow passing 'index' to specify which labels to compute for.
    # Finding unique labels first is safer to ensure we don't process background.
    labels = np.unique(labeled_mask)
    labels = labels[labels > 0] # Remove background
    
    if len(labels) == 0:
        return 0.0

    # Compute Mean intensity per label
    means = ndimage.mean(dapi_channel, labels=labeled_mask, index=labels)
    
    # Compute Standard Deviation intensity per label
    stds = ndimage.standard_deviation(dapi_channel, labels=labeled_mask, index=labels)
    
    # 4. Calculate Coefficient of Variation (CV)
    # CV = std / mean
    # Handle division by zero (if mean is 0, which is rare for valid nuclei but possible)
    
    # Initialize CV array
    cvs = np.zeros_like(means, dtype=np.float64)
    
    # Only divide where mean > 0
    valid_indices = means > 1e-6  # Use small epsilon
    
    if np.any(valid_indices):
        cvs[valid_indices] = stds[valid_indices] / means[valid_indices]
    
    # 5. Aggregate (Mean of CVs)
    # The feature is the mean CV across all nuclei in the image
    if len(cvs) > 0:
        result = np.mean(cvs)
    else:
        result = 0.0

    return float(result)
