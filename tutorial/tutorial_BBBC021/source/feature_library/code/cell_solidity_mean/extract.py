def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_closing, binary_opening, disk
    from scipy import ndimage

    # Convert to appropriate array type
    # The input is (512, 512, 3) uint8
    arr = np.asarray(img)
    
    # Check for valid dimensions
    if arr.ndim != 3 or arr.shape[2] != 3:
        # Fallback for unexpected shapes (e.g. if single channel passed by mistake)
        if arr.ndim == 2:
            # Assume it's a single channel image, proceed
            pass
        else:
            return 0.0

    # Determine the label mask to use
    label_mask = None

    # Strategy 1: Use provided segmentation masks
    if len(segmentation_masks) > 0:
        # We need to select the best mask for "cell solidity".
        # Usually, cell segmentation is larger than nuclear segmentation.
        # If multiple masks are provided, we try to find the one with the largest mean area
        # to approximate the "whole cell" mask rather than the "nuclei" mask.
        
        best_mask = None
        max_mean_area = -1
        
        for mask in segmentation_masks:
            if mask is None:
                continue
            
            # Ensure mask is integer labeled
            current_mask = mask.astype(int)
            
            # If the mask is binary (0 and 1 only), label it
            if current_mask.max() <= 1:
                current_mask = label(current_mask)
                
            # Calculate properties to check if this is a good candidate
            # We use a quick check on unique labels count
            unique_labels = np.unique(current_mask)
            if len(unique_labels) <= 1: # Only background
                continue
                
            # Compute mean area of objects
            # We can approximate this quickly by counting non-zero pixels / number of labels
            num_objects = len(unique_labels) - 1
            total_area = np.count_nonzero(current_mask)
            mean_area = total_area / num_objects if num_objects > 0 else 0
            
            if mean_area > max_mean_area:
                max_mean_area = mean_area
                best_mask = current_mask
        
        label_mask = best_mask

    # Strategy 2: Fallback to on-the-fly segmentation if no valid mask found
    if label_mask is None:
        # Use Channel 0 (Actin/Red) as it defines the cytoskeleton and cell boundary best
        # for solidity measurements (blebbing, protrusions).
        if arr.ndim == 3:
            actin_channel = arr[..., 0]
        else:
            actin_channel = arr # Fallback for 2D input

        # Preprocessing
        # Smooth slightly to reduce noise
        smooth = ndimage.gaussian_filter(actin_channel, sigma=2.0)
        
        # Thresholding
        try:
            thresh = threshold_otsu(smooth)
            binary = smooth > thresh
        except Exception:
            # Fallback if image is constant
            return 0.0

        # Morphological cleanup
        # Close holes inside cells
        binary = binary_closing(binary, disk(3))
        # Remove small speckles
        binary = binary_opening(binary, disk(2))
        
        # Label connected components
        label_mask = label(binary)

    # Compute Feature: Mean Solidity
    # Solidity = Area / ConvexHullArea
    
    if label_mask is None or label_mask.max() == 0:
        return 0.0

    props = regionprops(label_mask)
    
    solidity_values = []
    
    for prop in props:
        # Filter out very small artifacts that might skew the mean
        # A typical MCF-7 cell is significantly larger than 50 pixels
        if prop.area < 50:
            continue
            
        solidity_values.append(prop.solidity)
    
    if not solidity_values:
        return 0.0
        
    result = np.mean(solidity_values)

    return float(result)
