def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from scipy import ndimage

    # Convert to appropriate array type
    # Image is (512, 512, 3), uint8. Channel 2 is DAPI (Blue).
    arr = np.asarray(img)
    
    # Handle dimensionality and channel selection
    # We need the DAPI channel for nuclear intensity analysis
    dapi_channel = None
    
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Standard (H, W, C) format, Channel 2 is DAPI
        dapi_channel = arr[..., 2]
    elif arr.ndim == 2:
        # Grayscale image, assume it's the relevant channel
        dapi_channel = arr
    else:
        # Unexpected format, return 0.0
        return 0.0

    # Ensure dapi_channel is float for calculations, though regionprops handles ints well.
    # Keeping it as is or converting to float doesn't change centroid location logic much,
    # but let's ensure it's clean.
    dapi_channel = dapi_channel.astype(np.float32)

    # Handle segmentation masks
    labeled_mask = None
    
    # Check if valid segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask_input = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == dapi_channel.shape[:2]:
            # If the mask is already labeled (int), use it. 
            # If binary (bool/0-1), label it.
            if np.issubdtype(mask_input.dtype, np.integer) and mask_input.max() > 1:
                labeled_mask = mask_input
            else:
                labeled_mask = label(mask_input > 0)
    
    # Fallback: Generate segmentation if no valid mask provided
    if labeled_mask is None:
        # Simple background subtraction/smoothing
        blurred = ndimage.gaussian_filter(dapi_channel, sigma=2.0)
        try:
            thresh = threshold_otsu(blurred)
            binary_mask = blurred > thresh
            labeled_mask = label(binary_mask)
        except Exception:
            # If thresholding fails (e.g. uniform image), return 0
            return 0.0

    # Compute Region Properties
    # We need 'centroid' (geometric center) and 'weighted_centroid' (intensity center)
    # We pass the dapi_channel as the intensity_image
    props = regionprops(labeled_mask, intensity_image=dapi_channel)

    if not props:
        return 0.0

    displacements = []
    
    for prop in props:
        # Filter out very small artifacts to reduce noise
        if prop.area < 10:
            continue

        # Geometric centroid (y, x)
        yc, xc = prop.centroid
        
        # Weighted centroid (y, x) based on pixel intensity
        # If the region has uniform intensity or sum is 0, this might be equal to centroid or NaN
        # regionprops handles this, but let's be safe
        try:
            yw, xw = prop.weighted_centroid
        except (ValueError, ZeroDivisionError):
            # Fallback if weighted centroid cannot be calculated
            yw, xw = yc, xc

        # Calculate Euclidean distance
        # Distance = sqrt((y2-y1)^2 + (x2-x1)^2)
        dist = np.sqrt((yc - yw)**2 + (xc - xw)**2)
        displacements.append(dist)

    # Aggregate results
    if not displacements:
        return 0.0
    
    # Return the mean displacement across all nuclei
    # This scalar captures the average degree of chromatin asymmetry in the population
    result = np.mean(displacements)

    return float(result)
