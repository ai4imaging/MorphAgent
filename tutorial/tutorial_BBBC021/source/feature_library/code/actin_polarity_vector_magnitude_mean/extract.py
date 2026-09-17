def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import regionprops
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), Channel 0 = Actin
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Extract Actin channel (Channel 0)
        actin_channel = arr[:, :, 0]
    elif arr.ndim == 2:
        # Fallback if only 2D image provided (assume it's the relevant channel or a projection)
        actin_channel = arr
    else:
        return 0.0

    # Intensity normalization (optional but good practice for weighted centroids to avoid massive numbers)
    # We don't need strict 0-1 range for centroid calculation, but float is required.
    # arr is already float32.

    # Handle segmentation masks
    # We need at least one mask to define the cells.
    if not segmentation_masks or segmentation_masks[0] is None:
        # Without segmentation, we cannot compute per-cell polarity vectors.
        # Fallback: treat the entire image as one object if it has content, 
        # but technically this feature is defined per-cell.
        # Returning 0.0 is safer than crashing.
        return 0.0

    # Assume the first mask is the primary cell/object mask
    # Masks are typically labeled integers (0=bg, 1=cell1, 2=cell2...)
    mask = segmentation_masks[0]
    
    # Ensure mask shape matches image shape (handle potential 2D mask with 3D image)
    if mask.shape != actin_channel.shape:
        # If mask is 3D and image is 2D or vice versa, try to match
        if mask.ndim == 3 and mask.shape[:2] == actin_channel.shape:
             mask = mask[:, :, 0] # Take first slice if it's a stack
        elif mask.shape != actin_channel.shape:
            return 0.0

    # Ensure mask is integer type for regionprops
    mask = mask.astype(int)

    # Compute properties
    # We need 'centroid' (geometric center) and 'weighted_centroid' (intensity center)
    # regionprops calculates weighted_centroid if intensity_image is provided
    props = regionprops(mask, intensity_image=actin_channel)

    if not props:
        return 0.0

    magnitudes = []
    
    for prop in props:
        # Geometric centroid (y, x)
        cy, cx = prop.centroid
        
        # Weighted centroid (y, x) based on Actin intensity
        # If the region has 0 intensity sum, weighted_centroid might be NaN or equal to centroid depending on implementation version.
        # We check for validity.
        try:
            wcy, wcx = prop.weighted_centroid
        except (ValueError, AttributeError):
            # Fallback if weighted centroid cannot be computed (e.g. all zero intensity)
            wcy, wcx = cy, cx

        # Calculate vector magnitude (Euclidean distance)
        # Vector = (Weighted - Geometric)
        # Magnitude = sqrt((wy - gy)^2 + (wx - gx)^2)
        
        # Handle potential NaNs
        if np.isnan(wcy) or np.isnan(wcx):
            dist = 0.0
        else:
            dist = np.sqrt((wcy - cy)**2 + (wcx - cx)**2)
            
        magnitudes.append(dist)

    if not magnitudes:
        return 0.0

    # Return the mean magnitude across all cells
    result = np.mean(magnitudes)

    return float(result)
