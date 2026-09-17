def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import regionprops, perimeter
    from skimage.filters import threshold_otsu
    from scipy.ndimage import binary_fill_holes  # Correct import location

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Check dimensions: (512, 512, 3) expected
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract DAPI channel (Channel 2) for nucleus reference
    dapi_channel = arr[:, :, 2]

    # Normalize DAPI for thresholding
    if dapi_channel.max() > 0:
        dapi_norm = dapi_channel / dapi_channel.max()
    else:
        dapi_norm = dapi_channel

    # Generate a reference DAPI mask using Otsu thresholding
    # This is used to identify which of the provided segmentation masks corresponds to nuclei
    try:
        thresh = threshold_otsu(dapi_norm)
        dapi_ref_mask = dapi_norm > thresh
    except Exception:
        # Fallback if image is empty or uniform
        dapi_ref_mask = dapi_norm > 0.1

    # Select the best segmentation mask
    selected_mask = None
    
    if len(segmentation_masks) > 0:
        best_iou = -1.0
        
        for mask in segmentation_masks:
            # Ensure mask is boolean or integer labels
            if mask.ndim != 2:
                continue
            
            # Create a binary version of the mask (foreground vs background)
            binary_mask = mask > 0
            
            # Compute Intersection over Union (IoU) with the DAPI reference mask
            intersection = np.logical_and(binary_mask, dapi_ref_mask).sum()
            union = np.logical_or(binary_mask, dapi_ref_mask).sum()
            
            if union > 0:
                iou = intersection / union
            else:
                iou = 0.0
            
            # We assume the nucleus mask will have the highest overlap with the DAPI channel threshold
            if iou > best_iou:
                best_iou = iou
                selected_mask = mask

    # If no suitable mask found or provided, use the DAPI reference mask
    if selected_mask is None:
        # Label the reference mask if we have to use it
        selected_mask, _ = ndimage.label(dapi_ref_mask)

    # Ensure selected_mask is labeled (integers)
    if selected_mask.dtype == bool:
        labeled_mask, _ = ndimage.label(selected_mask)
    else:
        labeled_mask = selected_mask.astype(int)

    # Compute roughness for each nucleus
    # Roughness metric: (Perimeter^2) / (4 * pi * Area)
    # A perfect circle has a value of 1.0. Higher values indicate roughness/irregularity.
    # We will subtract 1.0 so that 0.0 means perfect circle (smooth).
    
    props = regionprops(labeled_mask)
    roughness_values = []

    for prop in props:
        area = prop.area
        # Filter small artifacts
        if area < 50:
            continue
            
        # Use regionprops perimeter (based on marching squares usually, or boundary pixels)
        perim = prop.perimeter
        
        if area > 0:
            # Circularity ratio: P^2 / (4 * pi * A)
            # For a circle: (2*pi*r)^2 / (4*pi*pi*r^2) = 4*pi^2*r^2 / 4*pi*r^2 = pi? 
            # Wait, standard circularity is 4*pi*Area / Perimeter^2 (which is <= 1).
            # The prompt asks for "perimeter^2 / (4 * pi * area)" which is >= 1.
            
            circularity_inverse = (perim ** 2) / (4 * np.pi * area)
            
            # Normalize: Roughness = Value - 1.0 (so 0 is smooth)
            # Ensure non-negative
            roughness = max(0.0, circularity_inverse - 1.0)
            roughness_values.append(roughness)

    if not roughness_values:
        return 0.0

    # Return the mean roughness across all nuclei in the image
    result = np.mean(roughness_values)

    return float(result)
