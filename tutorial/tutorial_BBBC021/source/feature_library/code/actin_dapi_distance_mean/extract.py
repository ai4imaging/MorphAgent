def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.segmentation import watershed
    from skimage.morphology import remove_small_objects

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not the expected 3-channel image, return 0.0
        return 0.0

    # Extract relevant channels
    # Channel 0: Actin (Cytoskeleton/Cell Body)
    # Channel 2: DAPI (Nucleus)
    actin_ch = arr[..., 0]
    dapi_ch = arr[..., 2]

    # Normalize intensities to [0, 1]
    def normalize(ch):
        vmax = np.percentile(ch, 99.5) if ch.size > 0 else 1.0
        if vmax <= 0: vmax = 1.0
        return np.clip(ch / vmax, 0.0, 1.0)

    actin_norm = normalize(actin_ch)
    dapi_norm = normalize(dapi_ch)

    # --- Segmentation Logic ---
    # We need matched pairs of (Nucleus, Cell Body) to compute the distance between their centroids.
    # Strategy:
    # 1. Identify Nuclei (Seeds).
    # 2. Identify Cell Body (Mask).
    # 3. Use Watershed to propagate Nuclei labels to the Cell Body boundaries.
    #    This ensures every cell body has exactly one corresponding nucleus ID.

    nuclei_labels = None
    cell_labels = None

    # Check if segmentation masks are provided
    # We assume if masks are provided, they might be separate files for nuclei and cells,
    # or a single file. However, without guaranteed matching IDs in external masks,
    # re-segmenting using the image channels is often more robust for this specific
    # "distance between organelles" feature to ensure 1-to-1 correspondence.
    # Given the complexity of matching external masks without metadata, we will prioritize
    # a robust internal watershed segmentation based on the clear DAPI/Actin signals.
    
    # --- Internal Segmentation Pipeline ---
    
    # 1. Detect Nuclei (Seeds)
    try:
        thresh_dapi = threshold_otsu(dapi_norm)
    except ValueError: # Handle empty images
        thresh_dapi = 0.1
        
    nuclei_mask = dapi_norm > thresh_dapi
    nuclei_mask = remove_small_objects(nuclei_mask, min_size=50)
    nuclei_labels = label(nuclei_mask)

    # 2. Detect Cell Body (Basin)
    try:
        thresh_actin = threshold_otsu(actin_norm)
    except ValueError:
        thresh_actin = 0.1
        
    # Actin signal is often weaker/more diffuse, so we might lower the threshold slightly
    # or just use Otsu. Let's stick to Otsu for robustness.
    cell_mask = actin_norm > thresh_actin
    # The cell body must contain the nucleus
    cell_mask = np.logical_or(cell_mask, nuclei_mask)
    
    # 3. Watershed
    # We use the negative actin intensity as the topographic map, seeded by nuclei
    # This expands the nuclei labels into the actin-defined cytoplasm
    if np.max(nuclei_labels) > 0:
        # watershed(image, markers, mask)
        # image: -actin_norm (basins are high intensity actin regions)
        # markers: nuclei_labels
        # mask: cell_mask (limit growth to actin signal)
        cell_labels = watershed(-actin_norm, nuclei_labels, mask=cell_mask)
    else:
        return 0.0

    # --- Feature Computation ---
    
    # We now have:
    # nuclei_labels: labeled image of nuclei
    # cell_labels: labeled image of entire cells (cytoplasm + nucleus), where ID matches nuclei_labels
    
    props_nuclei = regionprops(nuclei_labels)
    props_cells = regionprops(cell_labels)
    
    # Create a lookup for cell properties by label
    # Note: regionprops returns a list sorted by label, but let's be safe with a dict
    cell_props_map = {p.label: p for p in props_cells}
    
    distances = []
    
    image_height, image_width = arr.shape[:2]

    for n_prop in props_nuclei:
        label_id = n_prop.label
        
        # Check if we have a corresponding cell body
        if label_id not in cell_props_map:
            continue
            
        c_prop = cell_props_map[label_id]
        
        # Filter: Exclude border cells
        # If the cell body touches the edge, the centroid calculation is biased.
        min_row, min_col, max_row, max_col = c_prop.bbox
        if min_row == 0 or min_col == 0 or max_row == image_height or max_col == image_width:
            continue
            
        # Get Centroids
        # regionprops centroid is (row, col)
        ny, nx = n_prop.centroid
        cy, cx = c_prop.centroid
        
        # Calculate Euclidean distance
        dist = np.sqrt((nx - cx)**2 + (ny - cy)**2)
        distances.append(dist)

    # --- Aggregation ---
    if len(distances) == 0:
        return 0.0
        
    result = np.mean(distances)

    return float(result)
