def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract relevant channels
    # Channel 0: Actin (Cytoskeleton)
    # Channel 2: DAPI (Nucleus)
    actin_channel = arr[:, :, 0]
    dapi_channel = arr[:, :, 2]

    # Normalize channels to [0, 1] for processing
    def normalize(c):
        p99 = np.percentile(c, 99.5)
        if p99 > 0:
            c = c / p99
        return np.clip(c, 0.0, 1.0)

    actin_norm = normalize(actin_channel)
    dapi_norm = normalize(dapi_channel)

    # --- Segmentation Logic ---
    # We need instance segmentation to calculate the ratio "per cell".
    # If masks are provided, we use them. Otherwise, we perform a basic watershed segmentation.
    
    labeled_cells = None
    labeled_nuclei = None

    # Check if valid segmentation masks are provided
    # We expect masks to potentially separate nuclei and cells, but the prompt implies
    # we might just get a tuple of masks. We'll prioritize generating our own consistent
    # segmentation to ensure the Nucleus vs Cytoplasm definition is robust for this specific metric.
    # However, if a mask is passed, we can try to use it.
    # Given the complexity of matching external masks to specific compartments without metadata,
    # and the requirement for a specific N/C ratio, a self-contained watershed is often more reliable
    # unless the masks are explicitly labeled 'nuclei' and 'cells'.
    # Let's implement a robust internal segmentation pipeline as the primary method, 
    # as this ensures the "Actin minus DAPI" logic holds on the same set of pixels.

    # 1. Segment Nuclei (Seeds)
    # Smooth DAPI
    dapi_smooth = ndimage.gaussian_filter(dapi_norm, sigma=2)
    try:
        thresh_nuc = threshold_otsu(dapi_smooth)
    except ValueError: # Handle empty images
        thresh_nuc = 0.1
        
    mask_nuc = dapi_smooth > thresh_nuc
    # Fill holes and remove small objects
    mask_nuc = ndimage.binary_fill_holes(mask_nuc)
    # Label nuclei
    labeled_nuclei, num_nuclei = label(mask_nuc, return_num=True)
    
    # Filter small nuclei
    sizes = ndimage.sum(mask_nuc, labeled_nuclei, range(num_nuclei + 1))
    mask_nuc = mask_nuc & (sizes[labeled_nuclei] > 50) # Min nucleus size
    labeled_nuclei, num_nuclei = label(mask_nuc, return_num=True)

    # 2. Segment Cell Body (Basin)
    # Smooth Actin
    actin_smooth = ndimage.gaussian_filter(actin_norm, sigma=2)
    try:
        thresh_actin = threshold_otsu(actin_smooth)
    except ValueError:
        thresh_actin = 0.1
        
    # The cell body should contain the nucleus. Union of signals helps robustness.
    # Sometimes actin is weak over the nucleus.
    combined_signal = np.maximum(actin_smooth, dapi_smooth)
    mask_cell = combined_signal > thresh_actin
    mask_cell = ndimage.binary_fill_holes(mask_cell)

    # 3. Watershed to separate cells
    # Use nuclei as markers
    if num_nuclei > 0:
        # Distance transform for topology
        distance = ndimage.distance_transform_edt(mask_cell)
        # We use the labeled nuclei as markers for the watershed
        # The mask defines the area to fill
        labeled_cells = watershed(-distance, labeled_nuclei, mask=mask_cell)
    else:
        return 0.0

    # --- Feature Computation ---
    # Calculate N/C ratio per cell
    
    ratios = []
    
    # Iterate through each detected cell
    # regionprops is efficient for this
    props = regionprops(labeled_cells)
    
    for prop in props:
        # Get the label ID
        label_id = prop.label
        
        # Extract the bounding box to minimize computation
        min_row, min_col, max_row, max_col = prop.bbox
        
        # Extract local masks
        cell_local = labeled_cells[min_row:max_row, min_col:max_col] == label_id
        nuc_local = labeled_nuclei[min_row:max_row, min_col:max_col] == label_id
        
        # Calculate areas
        area_total = np.sum(cell_local)
        
        # The nucleus mask might slightly bleed out of the cell mask due to threshold diffs,
        # or be smaller. We strictly define nucleus area as the intersection of the 
        # specific nucleus label and the cell body, though usually the watershed ensures containment.
        # However, to be safe and follow "Actin minus DAPI" logic:
        # We use the DAPI mask defined earlier for the numerator.
        area_nuc = np.sum(nuc_local)
        
        # Cytoplasm area = Total Cell Area - Nucleus Area
        # We enforce non-negative cytoplasm area
        area_cyto = max(0.0, area_total - area_nuc)
        
        # Avoid division by zero
        if area_cyto > 1.0 and area_nuc > 1.0:
            ratio = area_nuc / area_cyto
            
            # Filter unrealistic ratios (e.g., if segmentation failed and cyto is tiny)
            # A typical N/C ratio is 0.1 - 2.0. Extreme values usually indicate debris.
            if 0.01 < ratio < 10.0:
                ratios.append(ratio)

    # --- Aggregation ---
    if not ratios:
        return 0.0
        
    result = np.mean(ratios)

    return float(result)
