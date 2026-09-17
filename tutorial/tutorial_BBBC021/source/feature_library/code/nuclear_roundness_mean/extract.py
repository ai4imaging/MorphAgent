def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    import math

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) - Channel 2 is DAPI (Nucleus)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If unexpected shape, try to handle or return 0.0
        if arr.ndim == 2:
            # Assume single channel grayscale, treat as nuclear channel if it's the only info
            nuclear_channel = arr
        else:
            return 0.0
    else:
        # Extract DAPI channel (Channel 2, Blue)
        nuclear_channel = arr[:, :, 2]

    # Determine the segmentation mask to use
    # We need a labeled mask of the nuclei
    labeled_mask = None

    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0:
        # Heuristic: If multiple masks, usually the nuclear mask is distinct. 
        # Without explicit metadata mapping in the function args, we check for the mask 
        # that best overlaps with the DAPI channel or simply use the first one if it looks like a label mask.
        # Given the feature is 'nuclear_roundness', we prioritize a mask that likely corresponds to nuclei.
        
        # If we have masks, we try to use the one that matches the DAPI signal best or just the first one
        # Assuming standard ordering often puts nuclei first or second. 
        # Let's try to use the first mask provided, assuming the system passes relevant masks.
        # If the mask is not labeled (binary), we label it.
        
        candidate_mask = segmentation_masks[0]
        if candidate_mask.shape == nuclear_channel.shape:
             # Ensure it's integer type for labeling
            mask_int = candidate_mask.astype(int)
            if mask_int.max() > 1:
                labeled_mask = mask_int # Already labeled
            else:
                labeled_mask = label(mask_int) # Binary to labeled
    
    # Fallback: If no mask provided, generate one from the DAPI channel
    if labeled_mask is None:
        # Normalize DAPI channel for thresholding
        dapi_norm = nuclear_channel
        if dapi_norm.max() > dapi_norm.min():
            dapi_norm = (dapi_norm - dapi_norm.min()) / (dapi_norm.max() - dapi_norm.min())
        
        # Apply Otsu thresholding
        try:
            thresh = threshold_otsu(dapi_norm)
            binary_mask = dapi_norm > thresh
            
            # Clean up noise (morphological opening)
            binary_mask = ndimage.binary_opening(binary_mask, structure=np.ones((3,3))).astype(int)
            
            # Label the objects
            labeled_mask = label(binary_mask)
        except Exception:
            return 0.0

    # Compute Region Properties
    props = regionprops(labeled_mask)
    
    roundness_values = []
    
    # Image dimensions for border checking
    height, width = nuclear_channel.shape
    
    for prop in props:
        # Filter small artifacts (noise)
        if prop.area < 50:
            continue
            
        # Filter objects touching the border (their shapes are incomplete, leading to incorrect roundness)
        min_row, min_col, max_row, max_col = prop.bbox
        if min_row == 0 or min_col == 0 or max_row == height or max_col == width:
            continue
            
        # Calculate Roundness
        # Formula: 4 * pi * Area / Perimeter^2
        # A perfect circle has roundness 1.0. 
        # Irregular shapes have roundness < 1.0.
        
        area = prop.area
        perimeter = prop.perimeter
        
        if perimeter == 0:
            current_roundness = 1.0 # Single pixel or degenerate case
        else:
            current_roundness = (4 * math.pi * area) / (perimeter ** 2)
            
        roundness_values.append(current_roundness)

    # Aggregate results
    if not roundness_values:
        return 0.0
        
    result = np.mean(roundness_values)

    return float(result)
