def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import regionprops, label
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), uint8
    # Channel 0: Actin (Red) - This is the target channel for this feature
    # Channel 1: Tubulin (Green)
    # Channel 2: DAPI (Blue)
    
    # Check for valid shape
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract Actin channel (Channel 0)
    actin_channel = arr[..., 0]

    # Normalize intensity to [0, 1] for stability, though relative weights matter most for moments
    # We keep the background at 0.
    vmax = np.percentile(actin_channel, 99.5) if actin_channel.size > 0 else 1.0
    if vmax > 0:
        actin_channel = actin_channel / vmax
    actin_channel = np.clip(actin_channel, 0.0, 1.0)

    # Determine Segmentation Mask
    # Strategy: Use provided masks if available, otherwise generate one via Otsu thresholding
    labeled_mask = None
    
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first provided mask (assuming it's a cell or relevant object mask)
        mask_input = segmentation_masks[0]
        
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == actin_channel.shape[:2]:
            # If mask is already labeled (int > 1), use it. If binary, label it.
            if np.max(mask_input) > 1:
                labeled_mask = mask_input.astype(int)
            else:
                labeled_mask = label(mask_input > 0)
    
    # Fallback: Generate mask if none provided or invalid
    if labeled_mask is None:
        try:
            thresh = threshold_otsu(actin_channel)
            binary_mask = actin_channel > thresh
            labeled_mask = label(binary_mask)
        except Exception:
            # Fallback for extremely low contrast or empty images
            return 0.0

    # Compute Region Properties
    # We need 'centroid' (geometric center) and 'weighted_centroid' (intensity center of mass)
    try:
        props = regionprops(labeled_mask, intensity_image=actin_channel)
    except (ValueError, IndexError):
        return 0.0

    displacements = []
    
    for prop in props:
        # Filter small noise
        if prop.area < 50:
            continue

        # Geometric Centroid (y, x)
        yc, xc = prop.centroid
        
        # Weighted Centroid (y, x) based on Actin intensity
        y_w, x_w = prop.weighted_centroid
        
        # Calculate Euclidean distance between the two centroids
        # This quantifies the shift of the "mass" of actin relative to the cell center
        displacement = np.sqrt((yc - y_w)**2 + (xc - x_w)**2)
        displacements.append(displacement)

    # Aggregate results
    if not displacements:
        return 0.0
        
    result = np.mean(displacements)

    return float(result)
