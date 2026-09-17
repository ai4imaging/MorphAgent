def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects, remove_small_holes
    from skimage.segmentation import clear_border
    import math

    # Convert to appropriate array type
    # The input image is (512, 512, 3) uint8
    arr = np.asarray(img, dtype=np.float32)

    # Check for valid dimensions
    if arr.ndim != 3 or arr.shape[2] != 3:
        # Fallback for unexpected shapes, though dataset spec says (512, 512, 3)
        return 0.0

    # Extract the DAPI channel (Channel 2 - Blue) which stains the Nucleus
    # Shape becomes (512, 512)
    dapi_channel = arr[:, :, 2]

    # Determine the segmentation mask to use
    mask = None
    
    # Check if external segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first provided mask. 
        # We assume the first mask corresponds to nuclei or cells.
        # If the mask is labeled (int), we use it directly.
        # If it's boolean/binary, we label it later.
        input_mask = segmentation_masks[0]
        
        # Ensure mask matches image spatial dimensions
        if input_mask.shape[:2] == dapi_channel.shape:
            mask = input_mask
    
    # If no valid external mask, perform on-the-fly segmentation
    if mask is None:
        # Normalize DAPI channel for thresholding
        # Robust min/max to avoid hot pixels affecting scaling
        p_min, p_max = np.percentile(dapi_channel, (1, 99))
        if p_max > p_min:
            norm_dapi = (dapi_channel - p_min) / (p_max - p_min)
        else:
            norm_dapi = dapi_channel # Should be flat
        norm_dapi = np.clip(norm_dapi, 0, 1)

        # Calculate Otsu threshold
        try:
            thresh = threshold_otsu(norm_dapi)
            binary_mask = norm_dapi > thresh
            
            # Post-process the binary mask
            # Remove small noise (e.g., < 50 pixels)
            binary_mask = remove_small_objects(binary_mask, min_size=50)
            # Fill holes inside nuclei to ensure accurate area/perimeter
            binary_mask = remove_small_holes(binary_mask, area_threshold=50)
            
            mask = binary_mask
        except Exception:
            # Fallback if thresholding fails (e.g., empty image)
            return 0.0

    # Label the mask if it's not already labeled
    # If mask is integer type and has values > 1, assume it's already labeled
    if np.issubdtype(mask.dtype, np.integer) and mask.max() > 1:
        labeled_mask = mask
    else:
        # Label connected components
        # Convert to integer/bool just in case
        labeled_mask = label(mask > 0)

    # Clear objects touching the border to avoid calculating circularity on cut-off shapes
    labeled_mask = clear_border(labeled_mask)

    # Compute properties
    regions = regionprops(labeled_mask)
    
    circularities = []
    
    for region in regions:
        # Filter very small artifacts that might have survived
        if region.area < 20:
            continue
            
        area = region.area
        perimeter = region.perimeter
        
        # Avoid division by zero
        if perimeter == 0:
            continue
            
        # Circularity formula: 4 * pi * Area / Perimeter^2
        # Value is 1.0 for a perfect circle, approaches 0.0 for elongated shapes
        circ = (4 * math.pi * area) / (perimeter ** 2)
        
        # Theoretical max is 1.0, but discrete pixels can sometimes yield slightly > 1.0
        # due to perimeter estimation methods. We don't clip here to preserve distribution,
        # but it's good to be aware.
        circularities.append(circ)

    # Return the mean circularity
    if len(circularities) == 0:
        return 0.0
    
    result = np.mean(circularities)
    
    return float(result)
