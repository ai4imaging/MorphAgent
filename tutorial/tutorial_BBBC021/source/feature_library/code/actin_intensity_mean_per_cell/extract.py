def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    
    # Convert to appropriate array type (float32 for precision)
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset Description: (512, 512, 3), Channel 0 = Actin
    actin_channel = None
    
    if arr.ndim == 3:
        if arr.shape[2] == 3:
            # Standard (H, W, C) format
            actin_channel = arr[:, :, 0]
        elif arr.shape[0] == 3:
            # Channel-first (C, H, W) format - less likely but possible
            actin_channel = arr[0, :, :]
        else:
            # Fallback: if 3D but not clearly 3-channel, take mean or first slice
            actin_channel = arr[:, :, 0]
    elif arr.ndim == 2:
        # If 2D, assume it's a single channel image (possibly pre-sliced)
        actin_channel = arr
    else:
        return 0.0

    # If we failed to extract a valid 2D channel, return 0.0
    if actin_channel is None:
        return 0.0

    # Feature: actin_intensity_mean_per_cell
    # Logic:
    # 1. If segmentation masks are available, use them to identify individual cells.
    #    Calculate the mean actin intensity for each cell, then average those means.
    #    This prevents large cells from dominating the statistic (Mean of Means).
    # 2. If no segmentation masks are available, fallback to the mean intensity of the entire image.

    result = 0.0
    
    # Check for segmentation masks
    # We prioritize the mask that likely represents the whole cell or cytoplasm.
    # If multiple masks are present, we iterate to find a valid one.
    mask = None
    if len(segmentation_masks) > 0:
        # Try to use the first available mask
        # In many pipelines, mask[0] is often the primary object (cell or nucleus)
        candidate_mask = segmentation_masks[0]
        
        # Ensure mask shape matches image shape (H, W)
        if candidate_mask.shape == actin_channel.shape:
            mask = candidate_mask
        elif candidate_mask.ndim == 3 and candidate_mask.shape[:2] == actin_channel.shape:
             # If mask is 3D (e.g. one-hot encoded or RGB mask), take max projection or first channel
             mask = np.max(candidate_mask, axis=2)

    if mask is not None:
        # Ensure mask is integer type for labeling
        mask = mask.astype(np.int32)
        
        # Get unique labels (excluding 0 which is background)
        labels = np.unique(mask)
        labels = labels[labels > 0]
        
        if len(labels) > 0:
            # Calculate mean intensity for each labeled cell
            # ndimage.mean returns a list of means, one for each label index provided
            mean_intensities = ndimage.mean(actin_channel, labels=mask, index=labels)
            
            # Compute the average of these per-cell means
            result = np.mean(mean_intensities)
        else:
            # Mask exists but is empty (no cells)
            result = 0.0
    else:
        # Fallback: No segmentation mask provided or valid
        # Compute mean intensity of the entire image
        # This is a reasonable proxy for "abundance" when segmentation is missing
        result = np.mean(actin_channel)

    return float(result)
