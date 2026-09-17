def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import regionprops, label
    from skimage.filters import threshold_otsu
    from scipy.spatial.distance import euclidean

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), Channel 0 = Actin (Red)
    if arr.ndim == 3 and arr.shape[2] == 3:
        actin_channel = arr[:, :, 0]  # Extract Actin channel
    elif arr.ndim == 2:
        # Fallback if only 2D image provided, assume it's the relevant channel
        actin_channel = arr
    else:
        return 0.0

    # Intensity normalization
    # Normalize to [0, 1] for weighting calculations
    vmax = np.percentile(actin_channel, 99.5) if actin_channel.size > 0 else 1.0
    if vmax > 0:
        actin_channel = actin_channel / vmax
    actin_channel = np.clip(actin_channel, 0.0, 1.0)

    # Handle segmentation masks
    labeled_mask = None
    
    # Check if valid segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask.shape[:2] == actin_channel.shape[:2]:
            # If mask is not integer labeled, label it
            if mask.dtype == bool or np.max(mask) == 1:
                labeled_mask = label(mask)
            else:
                labeled_mask = mask.astype(int)

    # Fallback: Generate mask from Actin channel if no external mask provided
    if labeled_mask is None:
        try:
            thresh = threshold_otsu(actin_channel)
            binary_mask = actin_channel > thresh
            labeled_mask = label(binary_mask)
        except Exception:
            # If thresholding fails (e.g., empty image), return 0.0
            return 0.0

    # Compute region properties
    # We need 'centroid' (geometric center) and 'weighted_centroid' (intensity center)
    try:
        props = regionprops(labeled_mask, intensity_image=actin_channel)
    except ValueError:
        return 0.0

    if not props:
        return 0.0

    displacements = []
    for prop in props:
        # Geometric centroid (y, x)
        cy, cx = prop.centroid
        
        # Weighted centroid (y, x)
        # Note: weighted_centroid can be NaN if the region has 0 total intensity
        try:
            wcy, wcx = prop.weighted_centroid
        except (ValueError, AttributeError):
            continue
            
        # Check for NaNs which can happen if intensity sum is zero
        if np.isnan(wcy) or np.isnan(wcx):
            continue

        # Calculate Euclidean distance
        dist = np.sqrt((cy - wcy)**2 + (cx - wcx)**2)
        displacements.append(dist)

    if not displacements:
        return 0.0

    # Return the mean displacement across all cells
    result = np.mean(displacements)

    return float(result)
