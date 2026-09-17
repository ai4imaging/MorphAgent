def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import closing, disk
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), Channel 0 = Actin, Channel 1 = Tubulin, Channel 2 = DAPI
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Extract Actin channel (Channel 0)
        actin_channel = arr[..., 0]
    elif arr.ndim == 2:
        # Fallback if single channel passed (unlikely based on desc, but safe)
        actin_channel = arr
    else:
        return 0.0

    # Intensity normalization
    # Normalize to [0, 1] for processing
    vmax = np.percentile(actin_channel, 99.5) if actin_channel.size > 0 else 1.0
    if vmax > 0:
        actin_channel = actin_channel / vmax
    actin_channel = np.clip(actin_channel, 0.0, 1.0)

    # Segmentation Logic
    # Check if pre-computed segmentation masks are available
    # We prioritize a mask that looks like a cell body mask.
    # If multiple masks are provided, we need a heuristic. 
    # Usually, if masks are provided, they might be (nuclei, cells) or just (nuclei).
    
    labeled_cells = None
    
    if len(segmentation_masks) > 0:
        # Heuristic: Try to find a mask that covers a significant portion of the image 
        # (likely cell body) vs a small portion (likely nuclei).
        # Or simply use the last mask if multiple, assuming standard order often puts whole-cell last.
        # However, without strict metadata on mask order, a robust fallback is safer.
        # Let's check if any mask is suitable.
        
        for mask in segmentation_masks:
            if mask is None: continue
            
            # Ensure mask is integer for labeling
            mask_int = mask.astype(int)
            
            # Calculate coverage
            coverage = np.count_nonzero(mask_int) / mask_int.size
            
            # If coverage is reasonable for cell bodies (e.g., > 5% and < 95%)
            # This is a loose heuristic. If a mask exists, we generally trust it over raw segmentation.
            if coverage > 0.01: 
                labeled_cells = mask_int
                # If it's a binary mask (0 and 1), label it to separate instances
                if np.max(labeled_cells) == 1:
                    labeled_cells = label(labeled_cells)
                break
    
    # Fallback: De novo segmentation on Actin channel if no suitable mask found
    if labeled_cells is None:
        # 1. Smooth the image to reduce noise and merge actin fibers
        smoothed = ndimage.gaussian_filter(actin_channel, sigma=2)
        
        # 2. Thresholding (Otsu)
        try:
            thresh = threshold_otsu(smoothed)
            binary_mask = smoothed > thresh
        except Exception:
            # Fallback for very low signal images
            binary_mask = smoothed > 0.1

        # 3. Morphological operations to fill gaps in cytoskeleton
        # Actin often looks like a mesh; we want a solid body.
        binary_mask = closing(binary_mask, disk(3))
        
        # 4. Instance Segmentation (Watershed)
        # Use distance transform to find cell centers
        distance = ndimage.distance_transform_edt(binary_mask)
        
        # Find peaks (potential cell centers)
        # min_distance ensures we don't over-segment
        coords = peak_local_max(distance, min_distance=20, labels=binary_mask)
        mask_peaks = np.zeros(distance.shape, dtype=bool)
        mask_peaks[tuple(coords.T)] = True
        markers = label(mask_peaks)
        
        # Watershed
        labeled_cells = watershed(-distance, markers, mask=binary_mask)

    # Feature Computation: Mean Area
    # Get properties of labeled regions
    regions = regionprops(labeled_cells)
    
    areas = []
    for props in regions:
        # Filter out very small artifacts (noise)
        if props.area > 50:
            areas.append(props.area)
            
    if not areas:
        return 0.0
        
    result = np.mean(areas)

    return float(result)
