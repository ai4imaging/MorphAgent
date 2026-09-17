def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_opening, disk
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality and validate input
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not 3 channels, we can't reliably separate actin/nuclei as described
        # Fallback: if 2D, treat as single channel intensity
        if arr.ndim == 2:
            actin_channel = arr
            nuclei_channel = arr
        else:
            return 0.0
    else:
        # Channel 0: Actin (Cytoskeleton) - Defines cell shape/orientation
        # Channel 2: DAPI (Nuclei) - Defines cell centers
        actin_channel = arr[..., 0]
        nuclei_channel = arr[..., 2]

    # Normalize channels
    def normalize(c):
        vmax = np.percentile(c, 99.5) if c.size > 0 else 1.0
        if vmax <= 0: vmax = 1.0
        return np.clip(c / vmax, 0.0, 1.0)

    actin_norm = normalize(actin_channel)
    nuclei_norm = normalize(nuclei_channel)

    # Determine the labeled mask
    labeled_mask = None

    # 1. Try to use provided segmentation masks
    if len(segmentation_masks) > 0:
        # Check masks in order. We prefer a cell/cytoplasm mask (usually larger objects)
        # over a nuclei mask for orientation calculation.
        # Without metadata, we pick the first valid one, assuming it's the primary segmentation.
        for mask in segmentation_masks:
            if mask is not None and mask.ndim == 2 and np.max(mask) > 0:
                labeled_mask = mask.astype(int)
                break
    
    # 2. Fallback: Perform on-the-fly segmentation if no mask provided
    if labeled_mask is None:
        # Step A: Threshold Actin to get cell bodies (foreground)
        try:
            thresh_actin = threshold_otsu(actin_norm)
            mask_body = actin_norm > thresh_actin
        except ValueError: # Handle empty image
            return 0.0
        
        # Clean up noise
        mask_body = binary_opening(mask_body, disk(2))

        # Step B: Identify markers from Nuclei channel
        try:
            thresh_nuc = threshold_otsu(nuclei_norm)
            mask_nuc = nuclei_norm > thresh_nuc
        except ValueError:
            mask_nuc = mask_body # Fallback if nuclei channel is empty

        # Use distance transform on nuclei to find centers (better separation)
        distance = ndimage.distance_transform_edt(mask_nuc)
        # Find peaks in distance map (nuclei centers)
        # min_distance=7 corresponds to ~7 pixels radius (approx 4um), reasonable for nuclei
        coords = peak_local_max(distance, min_distance=7, labels=mask_nuc)
        mask_peaks = np.zeros(distance.shape, dtype=bool)
        mask_peaks[tuple(coords.T)] = True
        markers = label(mask_peaks)

        # Step C: Watershed segmentation
        # Use negative actin intensity as "elevation" so bright actin regions are basins
        # Or simply use distance transform of the body mask. 
        # Here, distance transform of body is robust for shape.
        distance_body = ndimage.distance_transform_edt(mask_body)
        labeled_mask = watershed(-distance_body, markers, mask=mask_body)

    # Feature Extraction: Orientation Coherence
    # Get properties of all regions
    props = regionprops(labeled_mask)

    if not props:
        return 0.0

    # Collect orientation vectors
    # We use the "doubled angle" method for axial data (orientation is undirected [0, 180])
    # Mapping: theta -> (cos(2*theta), sin(2*theta))
    
    sum_sin = 0.0
    sum_cos = 0.0
    count = 0

    for prop in props:
        # Filter out small debris
        if prop.area < 50:
            continue
        
        # Filter out round cells (eccentricity near 0)
        # Round cells have unstable orientation; including them adds noise.
        # Eccentricity range: [0 (circle), 1 (line)]
        # Threshold 0.4 allows slightly elongated cells but excludes perfect circles.
        if prop.eccentricity < 0.4:
            continue

        # prop.orientation is in radians [-pi/2, pi/2]
        angle = prop.orientation
        
        # Double the angle
        double_angle = 2 * angle
        
        # Accumulate unit vectors
        sum_cos += np.cos(double_angle)
        sum_sin += np.sin(double_angle)
        count += 1

    if count == 0:
        return 0.0

    # Calculate Mean Resultant Vector Length (R)
    # R = sqrt( (sum_cos/N)^2 + (sum_sin/N)^2 )
    avg_cos = sum_cos / count
    avg_sin = sum_sin / count
    
    coherence = np.sqrt(avg_cos**2 + avg_sin**2)

    return float(coherence)
