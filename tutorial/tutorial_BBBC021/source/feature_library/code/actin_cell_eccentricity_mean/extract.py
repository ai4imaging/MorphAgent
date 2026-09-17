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

    # Handle dimensionality and validate input
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract relevant channels
    # Channel 0: Actin (Cytoskeleton) - Defines cell boundary
    # Channel 2: DAPI (Nucleus) - Used as seeds for watershed if segmentation is missing
    actin_ch = arr[..., 0]
    dapi_ch = arr[..., 2]

    # Normalize channels to [0, 1]
    def normalize(ch):
        vmax = np.percentile(ch, 99.5) if ch.size > 0 else 1.0
        if vmax <= 0: vmax = 1.0
        return np.clip(ch / vmax, 0.0, 1.0)

    actin_norm = normalize(actin_ch)
    dapi_norm = normalize(dapi_ch)

    # Determine Segmentation Strategy
    labeled_cells = None

    # 1. Try to use provided segmentation masks
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask.shape[:2] == arr.shape[:2]:
            # If mask is boolean or binary (0/1), label it
            if mask.dtype == bool or len(np.unique(mask)) <= 2:
                labeled_cells = label(mask > 0)
            else:
                # Assume it's already an instance segmentation (integer labels)
                labeled_cells = mask.astype(int)

    # 2. Fallback: On-the-fly segmentation if no valid mask provided
    if labeled_cells is None:
        # Step A: Detect Nuclei (Seeds)
        # Smooth DAPI slightly
        dapi_smooth = gaussian(dapi_norm, sigma=2)
        try:
            thresh_nuc = threshold_otsu(dapi_smooth)
        except ValueError:  # Handle empty images
            thresh_nuc = 0.1
        
        nuclei_mask = dapi_smooth > thresh_nuc
        nuclei_mask = remove_small_objects(nuclei_mask, min_size=50)
        
        # Generate markers for watershed
        # Use distance transform to find centers of nuclei for better separation
        distance = ndimage.distance_transform_edt(nuclei_mask)
        # Find peaks in distance map
        local_maxi = peak_local_max(distance, labels=label(nuclei_mask), 
                                    footprint=np.ones((3, 3)), indices=False)
        markers = label(local_maxi)

        # Step B: Detect Cell Bodies (Actin)
        # Smooth Actin
        actin_smooth = gaussian(actin_norm, sigma=2)
        try:
            thresh_actin = threshold_otsu(actin_smooth)
        except ValueError:
            thresh_actin = 0.1
            
        # Create cell mask, slightly dilated/closed to fill gaps
        cell_mask = actin_smooth > thresh_actin
        cell_mask = binary_closing(cell_mask, disk(3))
        cell_mask = remove_small_objects(cell_mask, min_size=200)

        # Step C: Watershed Segmentation
        # Use the inverse of actin intensity as the "elevation map"
        # We restrict watershed to the cell_mask area
        if np.any(markers):
            labeled_cells = watershed(-actin_smooth, markers, mask=cell_mask)
        else:
            # Fallback if no nuclei found but actin exists
            labeled_cells = label(cell_mask)

    # Feature Computation: Mean Eccentricity
    # If segmentation failed or image is empty
    if labeled_cells is None or labeled_cells.max() == 0:
        return 0.0

    regions = regionprops(labeled_cells)
    eccentricities = []

    for region in regions:
        # Filter out very small artifacts that might skew shape analysis
        if region.area < 100:
            continue
        
        # Eccentricity: 0 = circle, 1 = line
        # Measures elongation of the fitted ellipse
        eccentricities.append(region.eccentricity)

    if not eccentricities:
        return 0.0

    result = np.mean(eccentricities)

    return float(result)
