def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu, gaussian
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    from skimage.morphology import remove_small_objects
    import math

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) - Channels: 0=Actin, 1=Tubulin, 2=DAPI
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Define channels
    ch_actin = arr[..., 0]  # Best for cell body shape
    ch_dapi = arr[..., 2]   # Best for nuclei seeds

    # Normalize channels to [0, 1]
    def normalize(c):
        vmax = np.percentile(c, 99.5) if c.size > 0 else 1.0
        if vmax > 0:
            c = c / vmax
        return np.clip(c, 0.0, 1.0)

    ch_actin = normalize(ch_actin)
    ch_dapi = normalize(ch_dapi)

    labeled_cells = None

    # Strategy 1: Use provided segmentation masks if available
    # We look for a mask that likely represents the whole cell (usually larger area than nuclei)
    if len(segmentation_masks) > 0:
        # Heuristic: Try to find a mask that isn't empty. 
        # If multiple, we might default to the first one or try to guess based on coverage.
        # Assuming the system passes relevant masks.
        for mask in segmentation_masks:
            if mask is not None and np.sum(mask) > 0:
                # Ensure it's labeled
                if mask.max() == 1 and mask.dtype == bool: # Binary mask
                    labeled_cells = label(mask)
                elif mask.ndim == 2: # Already labeled or integer mask
                    labeled_cells = mask.astype(int)
                    # If it's a binary mask stored as int (0, 255) or (0, 1) without distinct labels
                    if len(np.unique(labeled_cells)) <= 2:
                        labeled_cells = label(labeled_cells > 0)
                break
    
    # Strategy 2: Fallback internal segmentation (Watershed)
    if labeled_cells is None:
        try:
            # 1. Detect Nuclei (Seeds)
            # Smooth DAPI
            dapi_smooth = gaussian(ch_dapi, sigma=2)
            # Threshold DAPI
            thresh_dapi = threshold_otsu(dapi_smooth)
            mask_nuclei = dapi_smooth > thresh_dapi
            
            # Distance transform for seed separation
            distance = ndimage.distance_transform_edt(mask_nuclei)
            # Find peaks (seeds)
            # min_distance ensures we don't over-segment fragmented nuclei
            coords = peak_local_max(distance, min_distance=7, labels=mask_nuclei)
            mask_seeds = np.zeros(distance.shape, dtype=bool)
            mask_seeds[tuple(coords.T)] = True
            markers = label(mask_seeds)

            # 2. Detect Cell Body (Basin)
            # Smooth Actin
            actin_smooth = gaussian(ch_actin, sigma=2)
            # Threshold Actin (Cell mask)
            try:
                thresh_actin = threshold_otsu(actin_smooth)
            except ValueError: # Handle uniform image
                thresh_actin = 0.1
            mask_cells = actin_smooth > thresh_actin
            
            # Clean up cell mask
            mask_cells = remove_small_objects(mask_cells, min_size=50)

            # 3. Watershed
            # Invert intensity for watershed (basins are dark)
            # We use the actin channel gradient or intensity as the topological surface
            labeled_cells = watershed(-actin_smooth, markers, mask=mask_cells)
            
        except Exception:
            # If segmentation fails completely (e.g. empty image), return 0.0
            return 0.0

    # Feature Extraction: Compactness
    # Formula: (4 * pi * Area) / (Perimeter^2)
    # Range: 0 (line) to 1 (circle)
    
    if labeled_cells is None or labeled_cells.max() == 0:
        return 0.0

    regions = regionprops(labeled_cells)
    compactness_values = []

    for props in regions:
        # Filter small artifacts
        if props.area < 50:
            continue
            
        # Filter border cells (optional but good practice for shape features)
        # If a cell touches the border, its perimeter is artificial, skewing compactness.
        minr, minc, maxr, maxc = props.bbox
        if minr == 0 or minc == 0 or maxr == labeled_cells.shape[0] or maxc == labeled_cells.shape[1]:
            continue

        area = props.area
        perimeter = props.perimeter

        if perimeter == 0:
            continue

        # Calculate Compactness (Isoperimetric quotient)
        comp = (4.0 * math.pi * area) / (perimeter ** 2)
        
        # Clip to theoretical max of 1.0 (discrete pixels can sometimes cause slight >1)
        comp = min(comp, 1.0)
        
        compactness_values.append(comp)

    if not compactness_values:
        return 0.0

    result = np.mean(compactness_values)
    return float(result)
