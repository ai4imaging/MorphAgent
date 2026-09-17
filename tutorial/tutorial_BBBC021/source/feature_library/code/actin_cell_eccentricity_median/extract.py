def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu, gaussian
    from skimage.morphology import remove_small_objects, binary_closing, disk
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality
    # Expected shape: (H, W, C) = (512, 512, 3)
    if arr.ndim == 2:
        # If grayscale, assume it's Actin or a composite. 
        # We'll treat it as the primary channel.
        actin_img = arr
        dapi_img = None
    elif arr.ndim == 3:
        if arr.shape[-1] == 3:
            # Standard (H, W, C)
            # Channel 0: Actin (Red) - Target for cell boundary
            # Channel 2: DAPI (Blue) - Target for nuclei (seeds)
            actin_img = arr[..., 0]
            dapi_img = arr[..., 2]
        else:
            # Unexpected channel count, try to use the first channel
            actin_img = arr[..., 0]
            dapi_img = None
    else:
        return 0.0

    # Normalize Actin channel
    vmax_actin = np.percentile(actin_img, 99.5) if actin_img.size > 0 else 1.0
    if vmax_actin > 0:
        actin_img = actin_img / vmax_actin
    actin_img = np.clip(actin_img, 0.0, 1.0)

    # Normalize DAPI channel if available
    if dapi_img is not None:
        vmax_dapi = np.percentile(dapi_img, 99.5) if dapi_img.size > 0 else 1.0
        if vmax_dapi > 0:
            dapi_img = dapi_img / vmax_dapi
        dapi_img = np.clip(dapi_img, 0.0, 1.0)

    # Determine Segmentation Mask
    labeled_cells = None

    # 1. Try using provided segmentation masks
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Assume the first mask is the relevant one (cell mask)
        mask_input = segmentation_masks[0]
        # Ensure it matches image spatial dimensions
        if mask_input.shape[:2] == actin_img.shape[:2]:
            if mask_input.ndim == 2:
                labeled_cells = mask_input.astype(int)
            elif mask_input.ndim == 3:
                # If mask is 3D (e.g. one-hot or RGB), take max projection or first channel
                labeled_cells = mask_input[..., 0].astype(int)
            
            # If the mask is binary (0/1 or 0/255), label it
            if np.max(labeled_cells) <= 1:
                labeled_cells = label(labeled_cells)

    # 2. Fallback: On-the-fly segmentation
    if labeled_cells is None:
        # We need to segment cells based on Actin.
        # Actin often clumps, so we use Marker-Controlled Watershed if DAPI is available.
        
        # Threshold Actin to get cell foreground
        try:
            thresh_actin = threshold_otsu(actin_img)
        except ValueError: # Handle empty images
            thresh_actin = 0.0
            
        binary_actin = actin_img > thresh_actin
        # Clean up noise
        binary_actin = remove_small_objects(binary_actin, min_size=50)
        binary_actin = binary_closing(binary_actin, disk(2))

        if dapi_img is not None:
            # Use Nuclei as seeds
            try:
                thresh_dapi = threshold_otsu(dapi_img)
            except ValueError:
                thresh_dapi = 0.0
            
            binary_dapi = dapi_img > thresh_dapi
            binary_dapi = remove_small_objects(binary_dapi, min_size=20)
            
            # Label nuclei markers
            markers = label(binary_dapi)
            
            # If no nuclei found, fallback to distance transform on actin
            if np.max(markers) == 0:
                distance = ndimage.distance_transform_edt(binary_actin)
                coords = peak_local_max(distance, min_distance=20, labels=binary_actin)
                mask_peaks = np.zeros(distance.shape, dtype=bool)
                mask_peaks[tuple(coords.T)] = True
                markers = label(mask_peaks)

            # Watershed
            # We use the negative intensity of actin as the "elevation map"
            # so the "water" flows from high intensity (nuclei/center) to low intensity
            labeled_cells = watershed(-actin_img, markers, mask=binary_actin)
        else:
            # If no DAPI, just label the binary actin mask
            # This might under-segment touching cells, but it's the best fallback
            labeled_cells = label(binary_actin)

    # Compute Eccentricity
    # Eccentricity ranges from 0 (circle) to 1 (line)
    props = regionprops(labeled_cells)
    
    eccentricities = []
    for prop in props:
        # Filter small debris
        if prop.area < 50:
            continue
            
        # Optional: Filter objects touching the border if the image is large enough
        # to assume we have plenty of central cells. However, for robustness on 
        # small crops or sparse images, we often keep them or apply a loose filter.
        # Here we include them to ensure we get a value, but in strict analysis 
        # one might exclude them.
        
        eccentricities.append(prop.eccentricity)

    if not eccentricities:
        return 0.0

    # Return the median
    result = np.median(eccentricities)
    
    return float(result)
