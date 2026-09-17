def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import closing, square

    # Convert to appropriate array type
    arr = np.asarray(img)

    # Handle dimensionality and select DAPI channel (Channel 2)
    # Dataset is (512, 512, 3) where Ch2 is DAPI (Nucleus)
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Extract Blue channel (index 2) for nuclei
        nuclei_channel = arr[..., 2]
    elif arr.ndim == 2:
        # Fallback if single channel passed
        nuclei_channel = arr
    else:
        return 0.0

    # Determine Labeled Mask
    labeled_mask = None

    # 1. Try to use provided segmentation masks
    if len(segmentation_masks) > 0:
        # Check the first mask. It might be a nuclei mask.
        # Masks are often passed as labeled arrays or binary arrays.
        mask_candidate = segmentation_masks[0]
        
        # Ensure mask matches image spatial dimensions
        if mask_candidate.shape[:2] == nuclei_channel.shape[:2]:
            if mask_candidate.max() > 1:
                # Already labeled
                labeled_mask = mask_candidate.astype(int)
            elif mask_candidate.max() == 1:
                # Binary mask, need to label
                labeled_mask = label(mask_candidate)
    
    # 2. Fallback: Compute segmentation on the fly if no valid mask provided
    if labeled_mask is None:
        # Normalize for thresholding
        img_float = nuclei_channel.astype(np.float32)
        
        # Basic intensity check to avoid processing empty images
        if np.max(img_float) == np.min(img_float):
            return 0.0
            
        # Otsu Thresholding
        try:
            thresh = threshold_otsu(img_float)
            binary = img_float > thresh
            # Clean up noise
            binary = closing(binary, square(3))
            labeled_mask = label(binary)
        except Exception:
            return 0.0

    # Extract Region Properties
    regions = regionprops(labeled_mask)
    
    if not regions:
        return 0.0

    # Calculate Coherence (Order Parameter)
    # We use the "nematic order parameter" approach for 2D orientation.
    # Orientation is defined on [0, pi] or [-pi/2, pi/2].
    # Because orientation has 2-fold symmetry (0 deg is same as 180 deg),
    # we double the angle for vector averaging.
    
    weighted_cos_sum = 0.0
    weighted_sin_sum = 0.0
    total_weight = 0.0
    
    for props in regions:
        # Filter small noise
        if props.area < 50:
            continue
            
        # Get orientation (-pi/2 to pi/2)
        theta = props.orientation
        
        # Get eccentricity (0=circle, 1=line)
        # We weight the contribution by eccentricity because orientation
        # is meaningless for circular objects.
        weight = props.eccentricity
        
        # Double the angle to map to vector space
        # 2*theta maps the range to [-pi, pi]
        weighted_cos_sum += weight * np.cos(2 * theta)
        weighted_sin_sum += weight * np.sin(2 * theta)
        total_weight += weight

    if total_weight == 0:
        return 0.0

    # Calculate the magnitude of the resultant vector
    resultant_length = np.sqrt(weighted_cos_sum**2 + weighted_sin_sum**2)
    
    # Normalize by total weight to get coherence [0, 1]
    coherence = resultant_length / total_weight

    return float(coherence)
