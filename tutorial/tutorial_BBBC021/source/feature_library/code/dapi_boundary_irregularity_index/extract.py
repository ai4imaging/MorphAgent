def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import closing, disk
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), Channel 2 is DAPI (Nucleus)
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Extract DAPI channel (Index 2)
        dapi_img = arr[:, :, 2]
    elif arr.ndim == 2:
        # Fallback if single channel passed
        dapi_img = arr
    else:
        return 0.0

    # Normalize intensity for processing
    # Although shape features are binary, good segmentation requires good input
    vmax = np.percentile(dapi_img, 99.5) if dapi_img.size > 0 else 1.0
    if vmax > 0:
        dapi_img = dapi_img / vmax
    dapi_img = np.clip(dapi_img, 0.0, 1.0)

    # Determine Segmentation Mask
    labeled_mask = None

    # Check if a valid segmentation mask is provided in arguments
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask_input = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == dapi_img.shape[:2]:
            if mask_input.dtype == bool:
                labeled_mask = label(mask_input)
            else:
                labeled_mask = mask_input.astype(int)

    # Fallback: Perform on-the-fly segmentation if no mask provided
    if labeled_mask is None:
        # 1. Smoothing to reduce noise
        smooth = ndimage.gaussian_filter(dapi_img, sigma=2.0)
        
        # 2. Thresholding (Otsu)
        try:
            thresh = threshold_otsu(smooth)
            binary = smooth > thresh
        except Exception:
            # Fallback for very low contrast images
            binary = smooth > 0.1

        # 3. Morphological cleanup
        binary = closing(binary, disk(3))
        
        # 4. Watershed separation for clustered nuclei (MCF-7 cells often cluster)
        distance = ndimage.distance_transform_edt(binary)
        # Find peaks in distance map to serve as markers
        coords = peak_local_max(distance, min_distance=10, labels=binary)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers = label(mask)
        
        # Apply watershed
        labeled_mask = watershed(-distance, markers, mask=binary)

    # Compute Feature: Boundary Irregularity Index
    # Formula: (Perimeter^2 / (4 * pi * Area)) - 1
    # Value is 0 for a perfect circle, increases with irregularity
    
    props = regionprops(labeled_mask)
    
    irregularity_scores = []
    
    for prop in props:
        # Filter small artifacts
        if prop.area < 50:
            continue
            
        # Filter objects touching the border (shapes are incomplete, metric invalid)
        # Check bounding box coordinates against image shape
        min_row, min_col, max_row, max_col = prop.bbox
        if min_row == 0 or min_col == 0 or max_row == dapi_img.shape[0] or max_col == dapi_img.shape[1]:
            continue

        area = prop.area
        perimeter = prop.perimeter
        
        if area > 0:
            # Calculate Circularity/Form Factor inverse
            # Standard Circularity = 4 * pi * Area / Perimeter^2 (1 for circle, <1 for irregular)
            # Irregularity Index = (1 / Circularity) - 1
            #                    = (Perimeter^2 / (4 * pi * Area)) - 1
            
            # Note: Discrete perimeter estimation can be slightly noisy, but regionprops uses a good estimator.
            val = (perimeter ** 2) / (4 * np.pi * area)
            
            # Theoretically val >= 1.0, but discrete geometry can sometimes yield slightly < 1 for tiny circles
            # We subtract 1 to make 0 the baseline for a circle
            metric = max(0.0, val - 1.0)
            irregularity_scores.append(metric)

    # Aggregation
    if not irregularity_scores:
        return 0.0
        
    # Return the mean irregularity of the population
    result = np.mean(irregularity_scores)

    return float(result)
