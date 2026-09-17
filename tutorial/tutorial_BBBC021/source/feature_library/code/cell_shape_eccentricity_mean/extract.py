def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import closing, disk
    from scipy import ndimage

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3)
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Channel 0 is Actin (Cytoskeleton), best for cell shape
        # Channel 2 is Nuclei
        # We prefer Actin for "cell shape", but if we need to segment, we use it.
        actin_channel = arr[..., 0]
    elif arr.ndim == 2:
        # Fallback for single channel images
        actin_channel = arr
    else:
        return 0.0

    # Determine which mask to use
    selected_mask = None
    
    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0:
        # Strategy: Select the mask with the largest foreground area.
        # Rationale: Cell masks (cytoplasm/whole cell) typically cover more area than nuclear masks.
        max_area = -1
        for mask in segmentation_masks:
            if mask is None:
                continue
            
            # Ensure mask is 2D
            current_mask = np.asarray(mask)
            if current_mask.ndim > 2:
                current_mask = np.max(current_mask, axis=-1) # Project if needed
            
            # Calculate area (count non-zero pixels)
            current_area = np.count_nonzero(current_mask)
            
            if current_area > max_area:
                max_area = current_area
                selected_mask = current_mask
    
    # Fallback: Generate segmentation if no valid mask provided
    if selected_mask is None:
        # Normalize actin channel for thresholding
        if actin_channel.max() > actin_channel.min():
            norm_actin = (actin_channel - actin_channel.min()) / (actin_channel.max() - actin_channel.min())
        else:
            norm_actin = actin_channel
            
        # Smooth to reduce noise
        blurred = ndimage.gaussian_filter(norm_actin, sigma=2)
        
        # Threshold
        try:
            thresh = threshold_otsu(blurred)
            binary = blurred > thresh
            # Morphological closing to fill gaps in cytoskeleton
            selected_mask = closing(binary, disk(3))
        except Exception:
            # If thresholding fails (e.g. uniform image), return 0.0
            return 0.0

    # Ensure mask is labeled (integers)
    # If the mask is boolean or just 0/1, label it to identify connected components
    label_img = label(selected_mask)

    # Compute properties
    props = regionprops(label_img)
    
    eccentricities = []
    min_area = 50  # Filter out small noise artifacts
    
    for prop in props:
        if prop.area >= min_area:
            # Eccentricity: 0 = circle, 1 = line
            eccentricities.append(prop.eccentricity)
            
    # Compute mean
    if len(eccentricities) == 0:
        result = 0.0
    else:
        result = np.mean(eccentricities)

    return float(result)
