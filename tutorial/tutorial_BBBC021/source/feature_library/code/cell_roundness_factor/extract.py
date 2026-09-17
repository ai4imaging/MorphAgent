def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import closing, square
    from skimage.segmentation import clear_border
    import math

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), Channel 0 = Actin, Channel 1 = Tubulin, Channel 2 = DAPI
    # We need the Actin channel (Channel 0) for cell boundary definition
    if arr.ndim == 3 and arr.shape[2] == 3:
        actin_channel = arr[:, :, 0]
    elif arr.ndim == 2:
        # Fallback if only 2D image provided, assume it's the relevant channel
        actin_channel = arr
    else:
        return 0.0

    # Normalize Actin channel for segmentation logic
    # Range is uint8 [0, 255] usually, but we cast to float.
    # Normalize to [0, 1] for stability
    vmax = np.percentile(actin_channel, 99.5) if actin_channel.size > 0 else 1.0
    if vmax > 0:
        actin_norm = actin_channel / vmax
    else:
        actin_norm = actin_channel
    actin_norm = np.clip(actin_norm, 0.0, 1.0)

    # Determine Segmentation Mask
    # Scenario A: Use provided segmentation mask if available
    labeled_mask = None
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first mask provided. Assuming it matches image dimensions (H, W)
        mask_input = segmentation_masks[0]
        if mask_input.shape == actin_channel.shape:
            labeled_mask = mask_input.astype(int)
        elif mask_input.ndim == 3 and mask_input.shape[:2] == actin_channel.shape:
             # If mask is 3D (e.g. one-hot or RGB), collapse or take first channel
             labeled_mask = mask_input[:, :, 0].astype(int)
    
    # Scenario B: Self-segmentation using Actin channel
    if labeled_mask is None:
        # 1. Thresholding
        try:
            thresh = threshold_otsu(actin_norm)
            binary_mask = actin_norm > thresh
        except Exception:
            # Fallback for very low contrast/empty images
            binary_mask = actin_norm > 0.1

        # 2. Morphological cleanup
        # Close small holes to get solid cell shapes
        binary_mask = closing(binary_mask, square(3))
        
        # 3. Label connected components
        labeled_mask = label(binary_mask)

    # Clear objects touching the border
    # Objects cut by the border have artificial straight edges which distort perimeter/roundness
    labeled_mask = clear_border(labeled_mask)

    # Compute Feature: Roundness (Form Factor)
    # Formula: (4 * pi * Area) / (Perimeter^2)
    # Range: 0.0 (line) to 1.0 (perfect circle)
    
    props = regionprops(labeled_mask)
    
    roundness_values = []
    min_area = 100  # Filter out noise/debris
    
    for prop in props:
        area = prop.area
        perimeter = prop.perimeter
        
        if area < min_area:
            continue
            
        if perimeter == 0:
            continue
            
        # Calculate Form Factor
        # Note: regionprops perimeter calculation accounts for pixel connectivity
        roundness = (4 * math.pi * area) / (perimeter ** 2)
        
        # Clip to theoretical max of 1.0 (discrete pixels can sometimes slightly exceed 1.0 due to approx)
        roundness = min(roundness, 1.0)
        
        roundness_values.append(roundness)

    # Aggregation
    # We return the mean roundness of the cell population
    if len(roundness_values) == 0:
        return 0.0
    
    result = np.mean(roundness_values)

    return float(result)
