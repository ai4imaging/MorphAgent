def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import closing, square

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3) - Channels: 0=Actin, 1=Tubulin, 2=DAPI
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Valid 3-channel image
        pass
    elif arr.ndim == 2:
        # If 2D, treat as single channel (unlikely given dataset description, but safe fallback)
        arr = arr[:, :, np.newaxis]
        # Duplicate to 3 channels to simplify indexing logic if needed, 
        # or just handle as is. For segmentation fallback, we need specific channels.
    else:
        return 0.0

    # Determine the segmentation mask to use
    labeled_mask = None

    # 1. Try to use provided segmentation masks
    if len(segmentation_masks) > 0:
        # Check masks for validity. We prefer a mask that represents the whole cell (cytoplasm).
        # Often, if multiple masks are provided, one might be nuclei and one cytoplasm.
        # Without explicit metadata on which is which, we check sizes. 
        # Cytoplasm masks usually cover more area than nuclei masks.
        
        best_mask = None
        max_coverage = -1
        
        for mask in segmentation_masks:
            if mask is None:
                continue
            
            # Ensure mask is 2D
            curr_mask = np.array(mask)
            if curr_mask.ndim > 2:
                curr_mask = np.max(curr_mask, axis=-1) # Project if 3D
            
            # Check if it's not empty
            if np.sum(curr_mask > 0) == 0:
                continue

            # Calculate coverage to guess if it's a cell mask (usually larger) vs nucleus
            coverage = np.sum(curr_mask > 0)
            if coverage > max_coverage:
                max_coverage = coverage
                best_mask = curr_mask
        
        if best_mask is not None:
            # If the mask is already labeled (int values > 1), use it.
            # If it's binary (0 and 1/255), label it.
            if np.max(best_mask) > 1 and np.issubdtype(best_mask.dtype, np.integer):
                labeled_mask = best_mask
            else:
                labeled_mask = label(best_mask > 0)

    # 2. Fallback: Generate segmentation from image if no valid mask provided
    if labeled_mask is None:
        # Use Channel 0 (Actin) and Channel 1 (Tubulin) combined for cell body
        # Channel 0 is index 0, Channel 1 is index 1
        if arr.shape[2] >= 2:
            # Combine Actin and Tubulin for robust cell shape
            ch_actin = arr[:, :, 0]
            ch_tubulin = arr[:, :, 1]
            detection_image = np.maximum(ch_actin, ch_tubulin)
        else:
            detection_image = arr[:, :, 0]

        # Normalize detection image
        if detection_image.max() > 0:
            detection_image = detection_image / detection_image.max()
        
        # Smooth to reduce noise
        detection_image = ndimage.gaussian_filter(detection_image, sigma=2)

        # Threshold
        try:
            thresh = threshold_otsu(detection_image)
            binary_mask = detection_image > thresh
        except Exception:
            # Fallback if image is uniform
            return 0.0

        # Morphological cleanup
        binary_mask = closing(binary_mask, square(3))
        
        # Label connected components
        labeled_mask = label(binary_mask)

    # Calculate properties
    # We need the perimeter of each object
    regions = regionprops(labeled_mask)
    
    perimeters = []
    for region in regions:
        # Filter small noise
        if region.area < 50:
            continue
        perimeters.append(region.perimeter)

    if not perimeters:
        return 0.0

    # Calculate mean perimeter
    result = np.mean(perimeters)

    return float(result)
