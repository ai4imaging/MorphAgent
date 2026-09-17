def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label
    from skimage.filters import threshold_otsu
    from skimage.segmentation import clear_border
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # 1. Extract the Nuclear Channel (DAPI)
    # Based on dataset info: Channel 2 is DAPI (Blue)
    # Image shape is (512, 512, 3)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        dapi_channel = arr[:, :, 2]
    elif arr.ndim == 2:
        # Fallback for single channel images
        dapi_channel = arr
    else:
        return 0.0

    # 2. Determine Segmentation Mask
    labeled_mask = None
    
    # Check if a valid segmentation mask is provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask_input = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == dapi_channel.shape[:2]:
            # If mask is already labeled (int), use it. If binary, label it.
            if np.issubdtype(mask_input.dtype, np.integer) and mask_input.max() > 1:
                labeled_mask = mask_input
            else:
                labeled_mask = label(mask_input > 0)
    
    # Fallback: Perform on-the-fly segmentation if no mask provided
    if labeled_mask is None:
        # Normalize DAPI channel for segmentation
        norm_dapi = dapi_channel
        if norm_dapi.max() > 0:
            norm_dapi = norm_dapi / norm_dapi.max()
        
        # Smooth to reduce noise
        smooth_dapi = ndimage.gaussian_filter(norm_dapi, sigma=2)
        
        try:
            thresh = threshold_otsu(smooth_dapi)
            binary_mask = smooth_dapi > thresh
        except ValueError:
            # Handle case where image is uniform (e.g. all black)
            return 0.0

        # Watershed segmentation to separate touching nuclei
        distance = ndimage.distance_transform_edt(binary_mask)
        # Find peaks in distance map
        coords = peak_local_max(distance, min_distance=7, labels=binary_mask)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndimage.label(mask)
        
        labeled_mask = watershed(-distance, markers, mask=binary_mask)

    # 3. Filter Objects
    # Remove objects touching the border (as their area is incomplete and would skew variance)
    labeled_mask = clear_border(labeled_mask)

    # 4. Calculate Areas
    # Using numpy unique with return_counts is much faster than regionprops for just area
    # labels will contain the label IDs, counts will contain the area (pixel count)
    labels, counts = np.unique(labeled_mask, return_counts=True)
    
    # Filter out background (label 0)
    if len(labels) > 0 and labels[0] == 0:
        counts = counts[1:]
    
    # Filter out very small artifacts (e.g., noise < 20 pixels)
    # This ensures we are measuring cellular heterogeneity, not noise
    valid_areas = counts[counts > 20]

    # 5. Compute Coefficient of Variation (CV)
    if len(valid_areas) < 2:
        # Need at least 2 objects to have meaningful variance, or 1 object has 0 variance
        return 0.0
    
    mean_area = np.mean(valid_areas)
    std_area = np.std(valid_areas)
    
    if mean_area == 0:
        return 0.0
        
    cv = std_area / mean_area

    return float(cv)
