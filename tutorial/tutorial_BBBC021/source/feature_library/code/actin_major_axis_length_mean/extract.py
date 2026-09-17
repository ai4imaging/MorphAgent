def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu, gaussian
    from skimage.segmentation import clear_border
    from skimage.morphology import remove_small_objects

    # 1. Input Validation and Preparation
    # Ensure input is a numpy array
    img = np.asarray(img)
    
    # Check dimensionality and extract Actin channel
    # Dataset spec: (512, 512, 3), Channel 0 = Actin
    if img.ndim == 3 and img.shape[-1] == 3:
        actin_channel = img[..., 0]  # Channel 0 is Red/Actin
    elif img.ndim == 2:
        # Fallback for single channel image
        actin_channel = img
    else:
        # Unexpected format
        return 0.0

    # 2. Determine Segmentation Mask
    labeled_mask = None
    
    # Strategy: Use provided masks if available, otherwise compute from image
    if len(segmentation_masks) > 0:
        # If multiple masks are provided (e.g., nuclei and cells), we need the cell body mask.
        # Heuristic: The cell mask usually covers a larger area than the nuclei mask.
        best_mask = None
        max_area = -1
        
        for mask in segmentation_masks:
            if mask is None:
                continue
            
            # Ensure mask is 2D (handle potential 3D inputs by max projection or slicing)
            if mask.ndim > 2:
                curr_mask = np.max(mask, axis=0)
            else:
                curr_mask = mask
                
            # Check if mask is labeled or binary
            # If it's binary (0/1 or boolean), label it. If it's integer labels, use as is.
            if curr_mask.dtype == bool or len(np.unique(curr_mask)) <= 2:
                curr_labeled = label(curr_mask > 0)
            else:
                curr_labeled = curr_mask.astype(int)
                
            # Calculate total area to identify the "cell" mask vs "nuclei" mask
            total_area = np.sum(curr_labeled > 0)
            if total_area > max_area:
                max_area = total_area
                best_mask = curr_labeled
        
        labeled_mask = best_mask

    # 3. Fallback: Compute Segmentation from Actin Channel if no valid mask found
    if labeled_mask is None:
        # Normalize actin channel for segmentation
        img_float = actin_channel.astype(np.float32)
        if img_float.max() > 0:
            img_float /= img_float.max()
        
        # Smooth to reduce noise
        blurred = gaussian(img_float, sigma=2)
        
        # Thresholding
        try:
            thresh = threshold_otsu(blurred)
            binary_mask = blurred > thresh
        except ValueError:
            # Handle case where image is uniform (e.g., all black)
            return 0.0
            
        # Labeling
        labeled_mask = label(binary_mask)

    # 4. Refine Mask
    # Clear objects touching the border (incomplete cells skew major axis length)
    labeled_mask = clear_border(labeled_mask)
    
    # Remove small artifacts (noise)
    # Using a conservative threshold (e.g., 50 pixels)
    labeled_mask = remove_small_objects(labeled_mask, min_size=50)
    
    # Re-label to ensure contiguous indices (optional but good for debugging)
    labeled_mask = label(labeled_mask)

    # 5. Compute Feature: Mean Major Axis Length
    regions = regionprops(labeled_mask)
    
    if len(regions) == 0:
        return 0.0
    
    major_axis_lengths = [r.major_axis_length for r in regions]
    
    # Calculate mean
    mean_major_axis = np.mean(major_axis_lengths)
    
    return float(mean_major_axis)
