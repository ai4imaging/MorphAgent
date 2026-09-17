def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects, binary_closing, disk
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset specifies (512, 512, 3) where Channel 2 is DAPI (Nucleus)
    dapi_channel = None
    
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Standard case: (H, W, C) -> Extract Blue channel (index 2)
        dapi_channel = arr[:, :, 2]
    elif arr.ndim == 2:
        # Fallback: Grayscale image, assume it's the relevant channel
        dapi_channel = arr
    else:
        # Unexpected format
        return 0.0

    # Normalize DAPI channel for processing
    # Although Otsu handles raw values well, normalization ensures consistency
    if dapi_channel.max() > dapi_channel.min():
        dapi_channel = (dapi_channel - dapi_channel.min()) / (dapi_channel.max() - dapi_channel.min())
    else:
        return 0.0 # Flat image, no features

    labeled_mask = None

    # Strategy 1: Use provided segmentation masks if available
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask_input = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == dapi_channel.shape[:2]:
            # If mask is already labeled (int type with values > 1), use it directly
            if np.issubdtype(mask_input.dtype, np.integer) and mask_input.max() > 1:
                labeled_mask = mask_input
            else:
                # If binary mask, label it
                labeled_mask = label(mask_input > 0)

    # Strategy 2: Compute segmentation internally if no valid mask provided
    if labeled_mask is None:
        # 1. Preprocessing: Gaussian blur to reduce noise
        blurred = ndimage.gaussian_filter(dapi_channel, sigma=2.0)

        # 2. Thresholding: Otsu's method
        try:
            thresh = threshold_otsu(blurred)
            binary = blurred > thresh
        except ValueError:
            # Handle case where image is uniform (e.g. all black)
            return 0.0

        # 3. Morphological cleanup
        # Fill holes to ensure accurate area calculation
        binary = ndimage.binary_fill_holes(binary)
        # Remove small artifacts (noise) - e.g., < 50 pixels
        binary = remove_small_objects(binary, min_size=50)
        # Smooth edges
        binary = binary_closing(binary, disk(2))

        # 4. Instance Separation (Watershed)
        # This is crucial for "Mean Area" to prevent merged nuclei from skewing the average
        distance = ndimage.distance_transform_edt(binary)
        
        # Find peaks in the distance map
        # min_distance ensures we don't over-segment textured nuclei
        coords = peak_local_max(distance, min_distance=7, labels=binary)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers = label(mask)
        
        # Apply watershed
        labeled_mask = watershed(-distance, markers, mask=binary)

    # Calculate Feature: Mean Area
    regions = regionprops(labeled_mask)
    
    if len(regions) == 0:
        return 0.0

    # Extract areas
    areas = [r.area for r in regions]
    
    # Compute mean
    mean_area = np.mean(areas)

    return float(mean_area)
