def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    
    # Convert to float32 for calculations to avoid overflow
    arr = np.asarray(img, dtype=np.float32)
    
    # Handle dimensionality and validate input
    # Expected shape is (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract specific channels based on dataset description
    # Channel 0: Actin (Red) -> Cytoskeleton (Denominator)
    # Channel 1: Tubulin (Green) -> Microtubules (Numerator)
    actin_channel = arr[..., 0]
    tubulin_channel = arr[..., 1]

    # Determine the segmentation mask to use
    # We prioritize a mask that likely covers the whole cell (cytoplasm)
    # If multiple masks are provided, we assume the one with the larger total area is the cell mask
    # (Nuclei masks are typically smaller than cell masks)
    labels = None
    
    if segmentation_masks and len(segmentation_masks) > 0:
        best_mask = None
        max_area = -1
        
        for mask in segmentation_masks:
            if mask is None:
                continue
            
            # Ensure mask is 2D and matches image spatial dimensions
            if mask.ndim == 2 and mask.shape == arr.shape[:2]:
                current_area = np.count_nonzero(mask)
                if current_area > max_area:
                    max_area = current_area
                    best_mask = mask
            elif mask.ndim == 3 and mask.shape[:2] == arr.shape[:2]:
                 # Handle case where mask might be 3D (e.g. one-hot or stacked), take max projection or slice
                 # Assuming integer labels, taking 2D slice
                 mask_2d = mask[..., 0]
                 current_area = np.count_nonzero(mask_2d)
                 if current_area > max_area:
                    max_area = current_area
                    best_mask = mask_2d
        
        if best_mask is not None:
            labels = best_mask

    # Fallback if no valid segmentation mask is provided
    # Create a simple intensity-based mask from the Actin channel (usually defines cell shape well)
    if labels is None:
        # Simple background estimation and thresholding
        # Use a low percentile as background estimate
        bg_level = np.percentile(actin_channel, 20)
        threshold = bg_level + (np.max(actin_channel) - bg_level) * 0.1
        mask_bool = actin_channel > threshold
        
        # Label the connected components
        labels, num_features = ndimage.label(mask_bool)
        if num_features == 0:
            return 0.0

    # Get unique object labels (excluding background 0)
    # Using np.unique is safer than assuming 1..N range, though slightly slower
    unique_labels = np.unique(labels)
    if unique_labels[0] == 0:
        unique_labels = unique_labels[1:]
    
    if len(unique_labels) == 0:
        return 0.0

    # Calculate sums of intensity for each cell
    # ndimage.sum is efficient for this
    # index=unique_labels ensures we get a result for each specific cell ID
    actin_sums = ndimage.sum(actin_channel, labels, index=unique_labels)
    tubulin_sums = ndimage.sum(tubulin_channel, labels, index=unique_labels)

    # Calculate ratio per cell
    # Add epsilon to prevent division by zero
    epsilon = 1e-6
    
    # Ensure we are working with arrays
    actin_sums = np.atleast_1d(actin_sums)
    tubulin_sums = np.atleast_1d(tubulin_sums)
    
    # Filter out very small/dark objects to reduce noise
    # (e.g., artifacts that passed thresholding but have negligible intensity)
    valid_indices = actin_sums > epsilon
    
    if not np.any(valid_indices):
        return 0.0
        
    actin_sums = actin_sums[valid_indices]
    tubulin_sums = tubulin_sums[valid_indices]

    # Calculate ratios
    ratios = tubulin_sums / (actin_sums + epsilon)

    # Compute the mean of the ratios
    # This represents the average cellular balance between microtubules and actin
    result = np.mean(ratios)

    return float(result)
