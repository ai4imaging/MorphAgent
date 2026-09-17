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
        # If not standard 3-channel, try to adapt or return 0
        if arr.ndim == 2:
            # Cannot compute polarity without distinct channels for nucleus and cell body
            return 0.0
        return 0.0

    # Extract relevant channels
    # Channel 0: Actin (Red) -> Cell Body
    # Channel 2: DAPI (Blue) -> Nucleus
    actin_channel = arr[..., 0]
    dapi_channel = arr[..., 2]

    # Normalize channels to 0-1 range for processing
    def normalize(ch):
        vmax = np.percentile(ch, 99.5) if ch.size > 0 else 1.0
        if vmax <= 0: vmax = 1.0
        return np.clip(ch / vmax, 0.0, 1.0)

    actin_norm = normalize(actin_channel)
    dapi_norm = normalize(dapi_channel)

    # --- Segmentation Logic ---
    # We need to identify individual cells to calculate per-cell polarity.
    # A global centroid difference would be biased by cell distribution in the image.
    
    # 1. Segment Nuclei (Seeds)
    try:
        thresh_nuc = threshold_otsu(dapi_norm)
    except ValueError: # Handle empty images
        return 0.0
        
    nuclei_mask = dapi_norm > thresh_nuc
    nuclei_mask = binary_opening(nuclei_mask, footprint=disk(2))
    nuclei_labels = label(nuclei_mask)

    # 2. Segment Cell Body (Mask)
    try:
        # Use a slightly lower threshold relative to Otsu for actin to capture the full spread
        thresh_actin = threshold_otsu(actin_norm) * 0.8
    except ValueError:
        thresh_actin = 0.1
        
    cell_mask = actin_norm > thresh_actin
    # Ensure nuclei are part of the cell mask
    cell_mask = np.logical_or(cell_mask, nuclei_mask)

    # 3. Instance Segmentation via Watershed
    # Use nuclei as markers and inverted actin intensity as the "basin"
    # This separates touching cells based on their nuclei
    if np.max(nuclei_labels) == 0:
        return 0.0

    # Create markers for watershed
    markers = nuclei_labels
    
    # Elevation map: inverted intensity so bright actin regions are "valleys"
    elevation_map = -actin_norm
    
    # Run watershed
    # mask=cell_mask restricts the watershed to the foreground
    cell_labels = watershed(elevation_map, markers, mask=cell_mask)

    # 4. Clear Border
    # Remove cells touching the border as their centroids might be artificially shifted
    cell_labels = clear_border(cell_labels)
    
    # We also need to clear the corresponding nuclei labels to ensure matching
    # Create a mask of valid cells
    valid_cells_mask = cell_labels > 0
    # Filter nuclei labels to only include those within valid cells
    nuclei_labels_valid = np.where(valid_cells_mask, nuclei_labels, 0)

    # --- Feature Computation ---
    
    # Get properties for nuclei and cells
    # Note: regionprops returns a list sorted by label index
    # We need to match label X in nuclei_props to label X in cell_props
    
    nuclei_props = regionprops(nuclei_labels_valid)
    cell_props = regionprops(cell_labels)
    
    # Create a lookup for nuclei centroids by label
    nuclei_centroids = {prop.label: np.array(prop.centroid) for prop in nuclei_props}
    
    displacements = []

    for cell_prop in cell_props:
        label_id = cell_prop.label
        
        # Check if we have a corresponding nucleus for this cell body
        if label_id in nuclei_centroids:
            # Get centroids (row, col)
            nuc_cen = nuclei_centroids[label_id]
            cell_cen = np.array(cell_prop.centroid)
            
            # Calculate Euclidean distance (magnitude of displacement vector)
            dist = np.linalg.norm(nuc_cen - cell_cen)
            
            # Filter out tiny debris that might have passed through
            if cell_prop.area > 50: 
                displacements.append(dist)

    # --- Aggregation ---
    if not displacements:
        return 0.0

    # Return the mean displacement across all valid cells
    result = np.mean(displacements)

    return float(result)
