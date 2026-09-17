def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu, gaussian
    from skimage.morphology import binary_opening, disk
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), Channel 0 = Actin
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not the expected 3-channel image, return 0.0
        return 0.0

    # Extract Actin channel (Channel 0)
    actin_channel = arr[:, :, 0]

    # Normalize Actin channel for thresholding
    # Robust min-max normalization
    p_min, p_max = np.percentile(actin_channel, (1, 99))
    if p_max > p_min:
        actin_norm = (actin_channel - p_min) / (p_max - p_min)
    else:
        actin_norm = actin_channel # Fallback if constant image
    actin_norm = np.clip(actin_norm, 0.0, 1.0)

    # Determine Segmentation Mask
    # Strategy: 
    # 1. If segmentation masks are provided, try to find a cell/cytoplasm mask.
    # 2. If no suitable mask is found, generate one from the Actin channel using Otsu thresholding.
    
    labeled_mask = None
    
    # Check provided masks
    if len(segmentation_masks) > 0:
        # Heuristic: If multiple masks, usually the larger one covers the cytoplasm/cell body.
        # If only one mask, assume it's the one we have.
        # We iterate to find a mask that covers a significant portion of the image but not all (background).
        
        # Try to use the first mask available as a candidate
        candidate_mask = segmentation_masks[0]
        
        # Ensure it's a labeled array (int)
        if candidate_mask.ndim == 2:
             labeled_mask = candidate_mask.astype(int)
        elif candidate_mask.ndim == 3:
             # If 3D mask passed for 2D image, take max projection or slice
             labeled_mask = np.max(candidate_mask, axis=0).astype(int)

    # Fallback: Generate mask from Actin channel if no external mask provided or valid
    if labeled_mask is None or labeled_mask.max() == 0:
        # Smooth to reduce noise
        blurred = gaussian(actin_norm, sigma=2)
        
        # Calculate threshold
        try:
            thresh = threshold_otsu(blurred)
            binary_mask = blurred > thresh
        except ValueError:
            # Handle case where image is uniform (e.g. all black)
            return 0.0
            
        # Clean up mask (remove small noise)
        binary_mask = binary_opening(binary_mask, disk(2))
        
        # Label connected components
        labeled_mask = label(binary_mask)

    # Compute Feature: Mean Tortuosity (Perimeter / Major Axis Length)
    props = regionprops(labeled_mask)
    
    tortuosity_values = []
    
    for prop in props:
        # Filter small artifacts (e.g., debris < 100 pixels)
        if prop.area < 100:
            continue
            
        perimeter = prop.perimeter
        major_axis = prop.major_axis_length
        
        # Avoid division by zero for single-pixel or degenerate objects
        if major_axis > 0:
            # Tortuosity definition: Perimeter / Major Axis Length
            # Circle: pi*D / D = 3.14
            # Ellipse: ~2*L / L = 2
            # Ruffled/Irregular: > 3.14
            val = perimeter / major_axis
            tortuosity_values.append(val)
            
    if not tortuosity_values:
        return 0.0
        
    result = np.mean(tortuosity_values)

    return float(result)
