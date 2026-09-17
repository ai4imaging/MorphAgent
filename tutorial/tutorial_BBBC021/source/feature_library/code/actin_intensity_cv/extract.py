def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.measure import label

    # Convert to appropriate array type
    # The input is expected to be (512, 512, 3) uint8
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality and channel selection
    # Dataset description: Channel 0 = Actin (Red), Channel 1 = Tubulin (Green), Channel 2 = DAPI (Blue)
    # We need the Actin channel (Channel 0) for this feature.
    if arr.ndim == 3 and arr.shape[2] >= 1:
        actin_channel = arr[:, :, 0]
    elif arr.ndim == 2:
        # Fallback if only 2D image provided (unlikely given description, but safe)
        actin_channel = arr
    else:
        return 0.0

    # Determine the mask to use
    # We prefer a cell mask (cytoplasm) over a nuclei mask because actin is cytoskeletal.
    # If multiple masks are provided, we assume the one with the larger area is the cell mask.
    final_mask = None

    if len(segmentation_masks) > 0:
        # Check available masks
        candidate_masks = []
        for m in segmentation_masks:
            if m is not None and m.shape == actin_channel.shape:
                candidate_masks.append(m)
        
        if len(candidate_masks) > 0:
            # Heuristic: Pick the mask with the most foreground pixels (likely whole cell vs nucleus)
            mask_areas = [np.sum(m > 0) for m in candidate_masks]
            best_mask_idx = np.argmax(mask_areas)
            final_mask = candidate_masks[best_mask_idx]

    # Fallback: If no valid segmentation masks provided, generate one using Otsu thresholding
    if final_mask is None:
        # Calculate threshold on the actin channel itself
        # Add a small epsilon to avoid errors if image is purely constant
        if actin_channel.max() > actin_channel.min():
            thresh = threshold_otsu(actin_channel)
            binary_mask = actin_channel > thresh
        else:
            binary_mask = actin_channel > 0
        
        # Label the connected components to treat them as individual objects
        final_mask, _ = label(binary_mask, return_num=True)

    # Ensure mask is integer type for indexing
    final_mask = final_mask.astype(int)

    # Get unique labels (excluding background 0)
    unique_labels = np.unique(final_mask)
    if len(unique_labels) <= 1 and unique_labels[0] == 0:
        return 0.0  # Only background found

    # Calculate CV per cell
    # CV = std / mean
    cv_values = []

    # Iterate through labels. 
    # Note: ndimage.mean and ndimage.standard_deviation are efficient, 
    # but iterating allows us to handle the division safely per object.
    
    # Optimization: Extract slices for each object to avoid scanning the whole array repeatedly
    slices = ndimage.find_objects(final_mask)
    
    for i, sl in enumerate(slices):
        if sl is None:
            continue
            
        label_id = i + 1 # find_objects returns slices for labels 1, 2, ...
        
        # Extract local view
        local_mask = (final_mask[sl] == label_id)
        local_intensity = actin_channel[sl][local_mask]
        
        if local_intensity.size == 0:
            continue

        mean_val = np.mean(local_intensity)
        std_val = np.std(local_intensity)

        # Avoid division by zero
        if mean_val > 1e-6:
            cv = std_val / mean_val
            cv_values.append(cv)
        else:
            # If mean is effectively zero, CV is undefined or 0. 
            # In biological context, 0 intensity means no signal, so 0 variation.
            cv_values.append(0.0)

    # Aggregate results
    if not cv_values:
        return 0.0
    
    # We return the mean CV across all cells in the image
    result = np.mean(cv_values)

    return float(result)
