def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import regionprops, label
    import math

    # Define the output variable
    result = 0.0

    # 1. Handle Segmentation Masks
    # The feature relies heavily on segmentation.
    # If masks are provided, we use them.
    # If multiple masks are provided, we need to decide which one represents the "cell".
    # Usually, if two masks are present, one is nuclei and one is cells/cytoplasm.
    # The cell mask is preferred for shape analysis.
    # Without metadata, we assume the mask with the larger average object area or simply the last one
    # (often secondary masks are derived later) is the cell mask, or we just take the first available if unsure.
    # Given the prompt doesn't specify order, we will try to use the first mask that has content.
    
    target_mask = None
    
    if len(segmentation_masks) > 0:
        # Iterate to find a valid non-empty mask
        for mask in segmentation_masks:
            if mask is not None and mask.size > 0 and np.any(mask):
                target_mask = mask
                # If we find a mask, we prefer one that looks like it covers more area (likely cell vs nucleus)
                # But for safety/speed, taking the first valid one is a standard fallback.
                break
    
    # 2. Fallback if no segmentation mask is provided
    # If no mask is provided, we must generate a crude one from the image to avoid returning 0.0 blindly.
    # The dataset is MCF-7 (cells). Channel 0 is Actin (Cytoskeleton), Channel 2 is DAPI (Nucleus).
    # A combined intensity threshold on Actin + Tubulin (Ch 0 + Ch 1) usually gives a decent cell body mask.
    if target_mask is None:
        # Convert image to float for processing
        arr = np.asarray(img, dtype=np.float32)
        
        # Check dimensions
        if arr.ndim == 3 and arr.shape[2] == 3:
            # Use Actin (0) and Tubulin (1) for cell shape
            # Normalize
            actin = arr[..., 0]
            tubulin = arr[..., 1]
            
            # Simple background subtraction / thresholding
            # We use a percentile based threshold to be robust against intensity variations
            combined = actin + tubulin
            threshold = np.percentile(combined, 85) # Conservative threshold to get main cell bodies
            binary_mask = combined > threshold
            
            # Label the binary mask to get instances
            target_mask = label(binary_mask)
        else:
            # If dimensions are unexpected, return NaN or 0.0
            return float(0.0)

    # 3. Ensure the mask is labeled (Instance Segmentation)
    # If the input mask was binary (0 and 1 only), we need to label it to separate objects.
    # If it's already integer labels, label() will re-index them but keep separation.
    # We cast to int to be safe.
    target_mask = target_mask.astype(int)
    
    # If max label is 1, it might be a semantic mask, so we run connected components
    if target_mask.max() == 1:
        labeled_mask = label(target_mask)
    else:
        labeled_mask = target_mask

    # 4. Compute Region Properties
    # We need Area and Perimeter for Form Factor
    props = regionprops(labeled_mask)
    
    if not props:
        return float(0.0)

    form_factors = []
    
    for prop in props:
        area = prop.area
        perimeter = prop.perimeter
        
        # Filter small noise objects
        if area < 50:
            continue
            
        # Avoid division by zero
        if perimeter == 0:
            continue
            
        # Calculate Form Factor
        # Formula: (4 * pi * Area) / (Perimeter^2)
        # 1.0 = perfect circle
        # < 1.0 = irregular/elongated
        ff = (4 * math.pi * area) / (perimeter ** 2)
        
        # Clip to max 1.0 (discretization errors can sometimes result in slightly > 1.0)
        # However, for pixelated circles, perimeter is often overestimated compared to area,
        # so ff is usually < 1.0. If it's > 1.0, it's an artifact.
        if ff > 1.0:
            ff = 1.0
            
        form_factors.append(ff)

    # 5. Aggregate Results
    if not form_factors:
        return float(0.0)
        
    result = np.mean(form_factors)

    return float(result)
