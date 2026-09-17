def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu, gaussian
    from skimage.measure import label
    from skimage.morphology import remove_small_objects
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    # Convert to appropriate array type
    arr = np.asarray(img)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), Channel 2 is DAPI (Nucleus)
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Extract DAPI channel (Channel 2)
        dapi_channel = arr[..., 2]
    elif arr.ndim == 2:
        # Fallback if single channel passed
        dapi_channel = arr
    else:
        # Unexpected format
        return 0.0

    # Check for provided segmentation masks first
    # If a mask is provided, we assume it's a reliable instance segmentation or binary mask
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask.shape[:2] == dapi_channel.shape[:2]:
            # If mask is integer type and has values > 1, assume it's an instance mask
            if np.issubdtype(mask.dtype, np.integer) and mask.max() > 1:
                # Count unique labels excluding 0 (background)
                # Using max label is faster if labels are sequential, but unique is safer
                unique_labels = np.unique(mask)
                count = len(unique_labels) - 1 if 0 in unique_labels else len(unique_labels)
                return float(count)
            else:
                # If binary mask, label connected components
                labeled_mask = label(mask > 0)
                return float(labeled_mask.max())

    # Fallback: Compute segmentation from raw image (Classic Watershed Pipeline)
    
    # 1. Normalization and Smoothing
    # Convert to float for processing
    img_float = dapi_channel.astype(np.float32)
    if img_float.max() > 0:
        img_float /= img_float.max()
    
    # Gaussian blur to reduce noise and smooth texture within nuclei
    # Sigma=2.0 is appropriate for 512x512 images of cells
    blurred = gaussian(img_float, sigma=2.0)

    # 2. Thresholding (Otsu)
    try:
        thresh = threshold_otsu(blurred)
        binary_mask = blurred > thresh
    except ValueError:
        # Handle case where image is uniform (e.g., all black)
        return 0.0

    # 3. Morphological Cleanup
    # Remove small artifacts (debris) - e.g., < 30 pixels area
    binary_mask = remove_small_objects(binary_mask, min_size=30)
    
    # Fill holes inside nuclei
    binary_mask = ndimage.binary_fill_holes(binary_mask)

    if not np.any(binary_mask):
        return 0.0

    # 4. Instance Separation (Watershed)
    # Compute distance transform (distance from background)
    distance = ndimage.distance_transform_edt(binary_mask)

    # Find peaks in the distance map (centers of nuclei)
    # min_distance ensures we don't over-segment a single nucleus with texture
    # min_distance=7 is roughly a radius of 7 pixels, suitable for MCF-7 nuclei
    coords = peak_local_max(distance, min_distance=7, labels=binary_mask)
    
    # Create markers for watershed
    mask_markers = np.zeros(distance.shape, dtype=bool)
    mask_markers[tuple(coords.T)] = True
    markers = label(mask_markers)

    # Apply watershed
    # We use -distance so watershed floods from the peaks (basins)
    labels = watershed(-distance, markers, mask=binary_mask)

    # 5. Count
    # The number of nuclei corresponds to the maximum label index
    nuclear_count = labels.max()

    return float(nuclear_count)
