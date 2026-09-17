def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from scipy.ndimage import binary_fill_holes  # Correct import location

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Check dimensions: Expected (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract Nuclear Channel (Channel 2 - Blue/DAPI)
    # According to dataset info: Ch0=Actin, Ch1=Tubulin, Ch2=DAPI
    nuclear_channel = arr[:, :, 2]

    # Normalize nuclear channel
    # uint8 range 0-255, but we use robust min/max for safety
    p_min, p_max = np.percentile(nuclear_channel, (1, 99))
    if p_max > p_min:
        nuclear_channel = (nuclear_channel - p_min) / (p_max - p_min)
        nuclear_channel = np.clip(nuclear_channel, 0.0, 1.0)
    else:
        # Flat image, return 0
        return 0.0

    # Determine Segmentation Strategy
    # The prompt warns that segmentation masks might be unordered or missing.
    # It is safer and more robust to segment the nuclei directly from the DAPI channel
    # to ensure we are measuring the correct objects.
    
    # Thresholding
    try:
        thresh = threshold_otsu(nuclear_channel)
        binary_mask = nuclear_channel > thresh
    except Exception:
        # Fallback if otsu fails (e.g. empty image)
        return 0.0

    # Post-processing: Fill holes and label
    binary_mask = binary_fill_holes(binary_mask)
    labeled_nuclei = label(binary_mask)
    
    # Compute properties
    regions = regionprops(labeled_nuclei)
    
    if not regions:
        return 0.0

    circularities = []
    
    for region in regions:
        # Filter out very small artifacts
        if region.area < 50:
            continue
            
        area = region.area
        perimeter = region.perimeter
        
        # Avoid division by zero
        if perimeter == 0:
            continue
            
        # Circularity formula: 4 * pi * Area / Perimeter^2
        # A perfect circle has circularity 1.0. 
        # Irregular shapes have lower values.
        circ = (4 * np.pi * area) / (perimeter ** 2)
        circularities.append(circ)

    if not circularities:
        return 0.0

    # Return mean circularity
    result = np.mean(circularities)
    
    return float(result)
