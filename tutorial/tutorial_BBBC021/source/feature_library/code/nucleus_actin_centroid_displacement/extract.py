def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.segmentation import watershed
    from skimage.morphology import binary_closing, disk
    from skimage.feature import peak_local_max

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract Channels
    # Channel 0: Actin (Red) - Cytoskeleton/Cell Body
    # Channel 2: DAPI (Blue) - Nucleus
    actin_ch = arr[:, :, 0]
    dapi_ch = arr[:, :, 2]

    # Normalize channels to [0, 1]
    def normalize(ch):
        vmax = np.percentile(ch, 99.5) if ch.size > 0 else 1.0
        if vmax <= 0: return ch
        return np.clip(ch / vmax, 0.0, 1.0)

    actin_norm = normalize(actin_ch)
    dapi_norm = normalize(dapi_ch)

    # --- Segmentation Logic ---
    # We need instance segmentation to link a specific nucleus to its specific cell body.
    # We will perform a marker-controlled watershed.

    # 1. Identify Nuclei (Markers)
    try:
        thresh_dapi = threshold_otsu(dapi_norm)
    except Exception:
        return 0.0 # Image likely empty or uniform

    nuclei_mask = dapi_norm > thresh_dapi
    # Clean up nuclei mask
    nuclei_mask = binary_closing(nuclei_mask, disk(2))
    # Label nuclei
    nuclei_labels = label(nuclei_mask)
    
    # Filter small artifacts from nuclei
    nuclei_props = regionprops(nuclei_labels)
    valid_nuclei_mask = np.zeros_like(nuclei_labels, dtype=int)
    current_label = 1
    
    # Map old labels to new labels to keep them contiguous
    label_map = {}
    
    for prop in nuclei_props:
        if prop.area > 50: # Minimum nucleus size
            valid_nuclei_mask[nuclei_labels == prop.label] = current_label
            label_map[current_label] = prop.label # Keep track if needed, though we just need the mask
            current_label += 1
    
    nuclei_labels = valid_nuclei_mask
    num_nuclei = current_label - 1

    if num_nuclei == 0:
        return 0.0

    # 2. Identify Cell Body (Basin)
    # Combine Actin and DAPI to ensure the cell covers the nucleus
    # Using a lower threshold for the cell body
    combined_signal = np.maximum(actin_norm, dapi_norm)
    try:
        # Use a lower threshold for cell boundaries, often mean or Li works well, 
        # here we approximate with a fraction of Otsu on the combined or just a low fixed value if normalized
        thresh_cell = threshold_otsu(combined_signal) * 0.7
    except:
        thresh_cell = 0.1

    cell_mask = combined_signal > thresh_cell
    cell_mask = binary_closing(cell_mask, disk(3))

    # 3. Watershed
    # We use the inverse of the combined intensity as the topographic map
    # The nuclei are the seeds. The cell_mask limits the basin.
    elevation_map = -combined_signal
    cell_labels = watershed(elevation_map, nuclei_labels, mask=cell_mask)

    # --- Feature Computation ---
    # Calculate displacement for each cell
    
    displacements = []
    
    # Get properties for nuclei and cells
    # Note: regionprops returns properties sorted by label
    nuclei_props = regionprops(nuclei_labels)
    cell_props = regionprops(cell_labels)
    
    # Create a dictionary for fast lookup of cell properties by label
    cell_prop_dict = {prop.label: prop for prop in cell_props}

    image_height, image_width = arr.shape[:2]

    for n_prop in nuclei_props:
        label_id = n_prop.label
        
        # Check if corresponding cell exists (it should, by definition of watershed)
        if label_id not in cell_prop_dict:
            continue
            
        c_prop = cell_prop_dict[label_id]
        
        # Check for border touching
        # If a cell touches the border, its centroid is unreliable.
        min_row, min_col, max_row, max_col = c_prop.bbox
        if min_row <= 0 or min_col <= 0 or max_row >= image_height or max_col >= image_width:
            continue

        # Get Centroids
        # regionprops centroid is (row, col)
        n_center = np.array(n_prop.centroid)
        c_center = np.array(c_prop.centroid)
        
        # Calculate Euclidean distance
        dist = np.linalg.norm(n_center - c_center)
        displacements.append(dist)

    # --- Aggregation ---
    if not displacements:
        return 0.0

    # Return the mean displacement across the population
    return float(np.mean(displacements))
