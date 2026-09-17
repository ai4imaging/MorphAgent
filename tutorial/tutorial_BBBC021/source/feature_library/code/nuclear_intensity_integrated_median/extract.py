def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.measure import label
    
    # 1. Input Validation and Preprocessing
    # Convert to float32 to prevent overflow during summation of intensities
    img_arr = np.asarray(img, dtype=np.float32)
    
    # Check dimensionality and extract DAPI channel
    # Dataset spec: (512, 512, 3), Channel 2 is DAPI (Blue)
    if img_arr.ndim == 3 and img_arr.shape[2] >= 3:
        dapi_channel = img_arr[:, :, 2]
    elif img_arr.ndim == 2:
        # Fallback if single channel passed (unlikely based on spec but safe)
        dapi_channel = img_arr
    else:
        return 0.0

    # 2. Mask Handling (Instance Segmentation)
    labeled_mask = None
    
    # Check if pre-computed segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Assume the first mask is the primary nuclear/cell mask
        mask_input = segmentation_masks[0]
        
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == dapi_channel.shape:
            # If the mask is already labeled (integers > 1), use it directly
            if np.max(mask_input) > 1:
                labeled_mask = mask_input.astype(int)
            else:
                # If binary mask (0/1 or 0/255), label connected components
                # Threshold > 0 to treat any non-zero value as foreground
                labeled_mask = label(mask_input > 0)
    
    # Fallback: Generate segmentation if no valid mask provided
    if labeled_mask is None:
        # Simple robust background subtraction/thresholding for DAPI
        try:
            # Gaussian blur to reduce noise before thresholding
            blurred = ndimage.gaussian_filter(dapi_channel, sigma=2)
            thresh = threshold_otsu(blurred)
            binary_mask = blurred > thresh
            # Label connected components
            labeled_mask = label(binary_mask)
        except Exception:
            # If Otsu fails (e.g., uniform image), return 0.0
            return 0.0

    # 3. Feature Computation: Integrated Intensity per Nucleus
    # Get unique labels (excluding background 0)
    # Using np.unique is safer but slower; assuming contiguous labels from label() or mask
    # We can use ndimage.sum_labels which is very efficient
    
    # Find the set of object indices present in the mask
    # We use unique to ensure we only query existing labels
    unique_labels = np.unique(labeled_mask)
    if len(unique_labels) <= 1: # Only background exists
        return 0.0
    
    # Remove background label (0)
    unique_labels = unique_labels[unique_labels != 0]
    
    # Calculate sum of pixel intensities for each labeled region
    # index=unique_labels ensures we get a result for each specific nucleus
    integrated_intensities = ndimage.sum_labels(dapi_channel, labeled_mask, index=unique_labels)
    
    # 4. Aggregation: Median
    if integrated_intensities.size == 0:
        return 0.0
        
    result = np.median(integrated_intensities)
    
    return float(result)
