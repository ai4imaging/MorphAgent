def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_closing, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), Channel 0 = Actin (Red), Channel 1 = Tubulin (Green), Channel 2 = DAPI (Blue)
    # We need the Actin channel (Channel 0) for cell boundary definition
    if arr.ndim == 3 and arr.shape[2] == 3:
        actin_channel = arr[..., 0]
    elif arr.ndim == 2:
        # Fallback if passed a single channel image, assume it's the relevant one
        actin_channel = arr
    else:
        return 0.0

    # Intensity normalization
    # Normalize to [0, 1] for thresholding
    vmax = np.percentile(actin_channel, 99.5) if actin_channel.size > 0 else 1.0
    if vmax > 0:
        actin_channel = actin_channel / vmax
    actin_channel = np.clip(actin_channel, 0.0, 1.0)

    # Determine the labeled mask
    labeled_mask = None

    # Scenario 1: Use provided segmentation masks if available
    # We prioritize masks that might represent 'cells' or 'cytoplasm'
    if len(segmentation_masks) > 0:
        # Check masks in order. Usually, if multiple exist, one might be nuclei and one cells.
        # Without specific metadata on which is which, we look for the one with larger coverage 
        # or simply use the first one if it's a valid label map.
        for mask in segmentation_masks:
            if mask is not None and mask.ndim == 2:
                # Ensure it's labeled (integers)
                if np.issubdtype(mask.dtype, np.integer):
                    labeled_mask = mask
                else:
                    # If binary, label it
                    labeled_mask = label(mask > 0)
                
                # If we found a valid mask, break. 
                # (In a real scenario, we might prefer the 'largest' mask to ensure it's cytoplasm not nuclei,
                # but typically the first mask passed is the primary segmentation).
                if labeled_mask.max() > 0:
                    break
    
    # Scenario 2: Fallback to computing segmentation from Actin channel
    if labeled_mask is None or labeled_mask.max() == 0:
        # Apply Gaussian blur to reduce noise
        blurred = ndimage.gaussian_filter(actin_channel, sigma=2.0)
        
        # Determine threshold
        try:
            thresh = threshold_otsu(blurred)
        except Exception:
            thresh = 0.1 # Fallback
            
        binary_mask = blurred > thresh
        
        # Morphological cleanup: close gaps
        binary_mask = binary_closing(binary_mask, disk(3))
        
        # Label connected components
        labeled_mask = label(binary_mask)

    # Compute properties
    regions = regionprops(labeled_mask)
    
    solidity_values = []
    
    for region in regions:
        # Filter small artifacts (e.g., debris)
        if region.area < 100:
            continue
            
        # Solidity = Area / ConvexArea
        # regionprops computes this automatically
        solidity_values.append(region.solidity)

    # Aggregation
    if len(solidity_values) == 0:
        return 0.0
    
    result = np.mean(solidity_values)

    return float(result)
