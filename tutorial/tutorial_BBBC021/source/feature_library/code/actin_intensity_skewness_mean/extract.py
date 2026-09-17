def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage, stats
    from skimage.measure import label
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) RGB. Channel 0 is Actin.
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Extract Actin channel (Channel 0)
        actin_channel = arr[..., 0]
    elif arr.ndim == 2:
        # Fallback if single channel passed
        actin_channel = arr
    else:
        # Unexpected format
        return 0.0

    # Determine the segmentation mask to use
    # We need a mask that defines cell boundaries.
    labels = None
    
    if len(segmentation_masks) > 0:
        # If masks are provided, we need to select the best one.
        # Typically, we want the mask that covers the largest area (likely whole cell/cytoplasm)
        # rather than just nuclei.
        best_mask = None
        max_area = -1
        
        for mask in segmentation_masks:
            if mask is None:
                continue
            
            # Ensure mask is 2D matching the image
            if mask.ndim == 3:
                # If mask is 3D (e.g. one-hot or RGB), take max projection or first channel
                curr_mask = mask.max(axis=-1) if mask.shape[-1] < 5 else mask[..., 0]
            else:
                curr_mask = mask
                
            # Check if dimensions match
            if curr_mask.shape != actin_channel.shape:
                continue
                
            # Calculate area to determine if it's likely a cell mask vs nucleus mask
            # Count non-zero pixels
            current_area = np.count_nonzero(curr_mask)
            if current_area > max_area:
                max_area = current_area
                best_mask = curr_mask
        
        if best_mask is not None:
            # Ensure it's labeled (instance segmentation)
            # If the mask is binary (0 and 1 only), label connected components
            if best_mask.max() <= 1:
                labels = label(best_mask > 0)
            else:
                labels = best_mask.astype(int)

    # Fallback: If no valid mask found, generate one using Otsu thresholding on Actin
    if labels is None:
        try:
            thresh = threshold_otsu(actin_channel)
            binary_mask = actin_channel > thresh
            labels = label(binary_mask)
        except Exception:
            # Fallback for completely empty/black images where otsu fails
            return 0.0

    # Get unique object IDs (excluding background 0)
    # Using np.unique is safer than range(1, max+1) in case of gaps, but slower.
    # For labeled masks, usually IDs are contiguous or we can just use range(1, max+1).
    # To be robust and efficient, we'll use ndimage.find_objects or similar, 
    # but labeled_comprehension handles index lists.
    
    # We need to compute skewness per object.
    # Skewness = E[((x - mu)/sigma)^3]
    
    # Define a function to compute skewness on a 1D array of pixels
    def compute_skewness(pixels):
        if pixels.size < 3:
            return np.nan
        # If variance is 0, skewness is undefined/0
        if np.std(pixels) == 0:
            return 0.0
        return stats.skew(pixels, bias=False)

    # Get indices of all objects
    # Note: labels might not be sequential, so we find unique non-zero labels
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2: # Only background exists
        return 0.0
    
    # Remove background (0)
    obj_ids = unique_labels[unique_labels != 0]
    
    # Compute skewness for each labeled region
    # labeled_comprehension is efficient for this
    skewness_values = ndimage.labeled_comprehension(
        input=actin_channel,
        labels=labels,
        index=obj_ids,
        func=compute_skewness,
        out_dtype=np.float64,
        default=np.nan
    )

    # Filter out NaNs (from small cells or errors)
    valid_skewness = skewness_values[~np.isnan(skewness_values)]

    if valid_skewness.size == 0:
        return 0.0

    # Compute the mean of the skewness values across all cells
    result = np.mean(valid_skewness)

    return float(result)
