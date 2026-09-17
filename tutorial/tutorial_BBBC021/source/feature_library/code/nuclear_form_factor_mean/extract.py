def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_opening, disk
    import math

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected shape: (512, 512, 3)
    # If 2D (H, W), expand to (H, W, 1) to handle uniformly, or just check length
    if arr.ndim == 2:
        # If single channel 2D, assume it's a projection or single channel image
        # We need to identify if it's DAPI. If it's just 2D, we treat it as the signal source.
        nuclei_channel = arr
    elif arr.ndim == 3:
        # Multi-channel image: (H, W, C)
        # Dataset info says Channel 2 (Index 2) is DAPI (Nucleus)
        if arr.shape[2] >= 3:
            nuclei_channel = arr[:, :, 2]
        else:
            # Fallback if fewer channels, use the last one or average
            nuclei_channel = np.mean(arr, axis=2)
    else:
        return 0.0

    # Determine the labeled mask for nuclei
    labeled_mask = None

    # Check if valid segmentation masks are provided
    # We prioritize the provided masks. If multiple are provided, we need to guess which one is nuclei.
    # Usually, if masks are provided, they might be [cells, nuclei] or just [nuclei].
    # Without explicit metadata mapping in the function signature, we check the masks.
    if len(segmentation_masks) > 0:
        # Iterate to find a suitable mask. 
        # Heuristic: Nuclei masks often have more distinct small objects than whole-cell masks, 
        # but technically we just take the first available one if we can't distinguish.
        # However, usually the system passes specific masks. Let's try to use the first one 
        # that looks like a label image (integer type).
        for mask in segmentation_masks:
            if mask is not None and np.issubdtype(mask.dtype, np.integer):
                # Check if it's not empty
                if np.max(mask) > 0:
                    labeled_mask = mask
                    break
    
    # Fallback: Perform segmentation on the DAPI channel if no mask provided
    if labeled_mask is None:
        # Normalize nuclei channel for segmentation
        # Robust min/max
        p_min, p_max = np.percentile(nuclei_channel, (1, 99))
        if p_max > p_min:
            norm_nuclei = (nuclei_channel - p_min) / (p_max - p_min)
        else:
            norm_nuclei = nuclei_channel
        norm_nuclei = np.clip(norm_nuclei, 0, 1)

        # Smooth
        smooth_nuclei = ndimage.gaussian_filter(norm_nuclei, sigma=2)

        # Threshold (Otsu)
        try:
            thresh = threshold_otsu(smooth_nuclei)
            binary_mask = smooth_nuclei > thresh
        except ValueError:
            # Handle case where image is uniform (e.g., all black)
            return 0.0

        # Clean up noise
        binary_mask = binary_opening(binary_mask, footprint=disk(2))

        # Label
        labeled_mask = label(binary_mask)

    # Compute Form Factor for each nucleus
    # Form Factor = 4 * pi * Area / Perimeter^2
    # Value range: 0 to 1 (1 is perfect circle)
    
    props = regionprops(labeled_mask)
    
    form_factors = []
    
    for prop in props:
        # Filter small artifacts
        if prop.area < 50:
            continue
            
        # Perimeter of a single pixel is 0 in some implementations or 4 in others depending on connectivity.
        # regionprops perimeter is calculated based on the boundary pixels.
        # Avoid division by zero.
        if prop.perimeter == 0:
            continue
            
        ff = (4 * math.pi * prop.area) / (prop.perimeter ** 2)
        
        # Theoretically max is 1.0, but discrete pixels can cause slight variations.
        # We don't strictly clip to 1.0 to preserve distribution characteristics, 
        # but values > 1.2 are likely artifacts or calculation quirks for very small objects.
        form_factors.append(ff)

    # Aggregate results
    if not form_factors:
        return 0.0
        
    result = np.mean(form_factors)

    return float(result)
