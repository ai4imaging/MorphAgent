def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.segmentation import watershed, clear_border
    from skimage.morphology import binary_opening, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not standard 3-channel image, return 0.0
        return 0.0

    # Extract channels
    # Channel 0: Actin (Red) -> Best for cell body shape
    # Channel 2: DAPI (Blue) -> Best for seeding nuclei
    actin_ch = arr[..., 0]
    dapi_ch = arr[..., 2]

    # Normalize channels to [0, 1] for processing
    def normalize(ch):
        vmax = np.percentile(ch, 99.5) if ch.size > 0 else 1.0
        if vmax <= 0: vmax = 1.0
        return np.clip(ch / vmax, 0.0, 1.0)

    actin_norm = normalize(actin_ch)
    dapi_norm = normalize(dapi_ch)

    # Determine the label mask to use
    labeled_cells = None

    # Check if valid segmentation masks are provided
    if len(segmentation_masks) > 0:
        # If masks are provided, we need to identify the cell body mask.
        # Often, datasets provide [nuclei_mask, cell_mask] or just one.
        # We prefer the mask with the larger total area as it likely represents the cytoplasm.
        best_mask = None
        max_area = -1
        
        for mask in segmentation_masks:
            if mask is None: continue
            # Ensure mask is 2D
            if mask.ndim == 3:
                mask = np.max(mask, axis=0) # Project if 3D
            if mask.ndim != 2: continue
            
            # Check if it's a labeled mask
            current_area = np.count_nonzero(mask)
            if current_area > max_area:
                max_area = current_area
                best_mask = mask
        
        if best_mask is not None:
            # Ensure it is labeled (integers 1..N), not binary
            if best_mask.max() == 1 and best_mask.dtype != bool:
                 labeled_cells = label(best_mask)
            else:
                 labeled_cells = best_mask.astype(int)

    # Fallback: Perform instant segmentation if no mask provided
    if labeled_cells is None:
        # 1. Detect Nuclei (Seeds)
        try:
            thresh_nuc = threshold_otsu(dapi_norm)
        except ValueError: # Handle uniform image
            thresh_nuc = 0.5
            
        nuclei_mask = dapi_norm > thresh_nuc
        nuclei_mask = binary_opening(nuclei_mask, footprint=disk(2))
        nuclei_seeds = label(nuclei_mask)

        # 2. Detect Cell Body (Foreground)
        # Smooth actin to get a better general shape
        actin_smooth = ndimage.gaussian_filter(actin_norm, sigma=2)
        try:
            thresh_cell = threshold_otsu(actin_smooth)
        except ValueError:
            thresh_cell = 0.5
        cell_mask = actin_smooth > thresh_cell

        # 3. Watershed Segmentation
        # Use negative intensity as topographic map (basins)
        # Mask restricts watershed to foreground
        if np.any(nuclei_seeds):
            labeled_cells = watershed(-actin_smooth, nuclei_seeds, mask=cell_mask)
        else:
            # If no nuclei found, just label the cell mask blobs
            labeled_cells = label(cell_mask)

    # If still no objects, return 0.0
    if labeled_cells is None or labeled_cells.max() == 0:
        return 0.0

    # Remove objects touching the border
    # Cells at the border are cut off, artificially reducing roundness (flat edges)
    labeled_cells = clear_border(labeled_cells)

    # Compute Roundness
    # Formula: (4 * pi * Area) / (Perimeter^2)
    # Range: 0.0 (line) to 1.0 (circle)
    
    props = regionprops(labeled_cells)
    roundness_values = []

    for prop in props:
        # Filter tiny noise
        if prop.area < 50:
            continue
            
        area = prop.area
        perimeter = prop.perimeter
        
        if perimeter == 0:
            val = 0.0
        else:
            val = (4 * np.pi * area) / (perimeter ** 2)
        
        # Clip to valid range (pixelation can cause slight >1.0)
        val = min(max(val, 0.0), 1.0)
        roundness_values.append(val)

    # Aggregate results
    if not roundness_values:
        return 0.0
        
    result = np.mean(roundness_values)

    return float(result)
