def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_closing, disk, remove_small_objects
    from scipy import ndimage

    # Convert to appropriate array type
    # Image shape is (512, 512, 3), uint8
    arr = np.asarray(img)
    
    # Check for valid image dimensions
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If unexpected format, return 0.0
        return 0.0

    # Determine which mask to use
    labeled_mask = None

    # Strategy 1: Use provided segmentation masks if available
    if len(segmentation_masks) > 0:
        # We need to select the mask that likely corresponds to the whole cell (cytoplasm/actin)
        # rather than just the nucleus, as elongation is a cell-shape feature.
        # Heuristic: The mask with the largest total foreground area is likely the cell mask.
        
        best_mask = None
        max_area = -1
        
        for mask in segmentation_masks:
            if mask is None:
                continue
            
            # Ensure mask is 2D (handle potential 3D or singleton dimensions)
            curr_mask = np.squeeze(mask)
            if curr_mask.ndim != 2:
                continue
                
            # Calculate total foreground area
            foreground_area = np.count_nonzero(curr_mask)
            
            if foreground_area > max_area:
                max_area = foreground_area
                best_mask = curr_mask
        
        if best_mask is not None:
            # If the mask is binary (0/1), label it. If it's already integer labels, use as is.
            if np.max(best_mask) == 1 and np.unique(best_mask).size <= 2:
                labeled_mask = label(best_mask)
            else:
                labeled_mask = best_mask.astype(int)

    # Strategy 2: Fallback to computing segmentation from the image
    if labeled_mask is None:
        # Use Channel 0 (Red/Actin) for cell shape as it defines the cytoskeleton
        # Channel 0 is index 0 in the last dimension
        actin_channel = arr[..., 0]
        
        # Normalize for processing
        actin_float = actin_channel.astype(np.float32)
        
        # Smooth to reduce noise and merge actin filaments
        # A slightly larger sigma helps define the overall cell hull
        smoothed = ndimage.gaussian_filter(actin_float, sigma=2.0)
        
        # Thresholding
        try:
            thresh = threshold_otsu(smoothed)
            binary = smoothed > thresh
        except Exception:
            # Fallback if image is constant
            return 0.0
            
        # Morphological cleaning
        # Close gaps in the cytoskeleton
        binary = binary_closing(binary, disk(3))
        # Remove small noise (debris)
        binary = remove_small_objects(binary, min_size=100)
        
        # Label connected components
        labeled_mask = label(binary)

    # Compute Feature: Mean Eccentricity
    # Eccentricity is the ratio of the focal distance to the major axis length.
    # 0 = circle, -> 1 = ellipse/line
    
    props = regionprops(labeled_mask)
    
    if len(props) == 0:
        return 0.0
    
    eccentricities = []
    for prop in props:
        # Filter out very small artifacts that might have survived
        if prop.area < 50:
            continue
            
        # Eccentricity is robustly calculated by regionprops
        # It handles the ellipse fitting internally
        ecc = prop.eccentricity
        
        # Check for NaN (can happen with singular regions)
        if not np.isnan(ecc):
            eccentricities.append(ecc)
            
    if not eccentricities:
        return 0.0
        
    result = np.mean(eccentricities)
    
    return float(result)
