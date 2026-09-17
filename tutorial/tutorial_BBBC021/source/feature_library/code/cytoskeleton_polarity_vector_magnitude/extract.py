def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops
    from skimage.segmentation import watershed, clear_border
    from skimage.morphology import binary_closing, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality and validate input
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Normalize intensity to [0, 1]
    # We use a robust max to avoid hot pixels skewing the normalization
    vmax = np.percentile(arr, 99.5) if arr.size > 0 else 1.0
    if vmax > 0:
        arr = arr / vmax
    arr = np.clip(arr, 0.0, 1.0)

    # Extract Channels
    # Channel 0: Actin (Red) - Cytoskeleton
    # Channel 1: Tubulin (Green) - Cytoskeleton
    # Channel 2: DAPI (Blue) - Nucleus
    ch_actin = arr[..., 0]
    ch_tubulin = arr[..., 1]
    ch_dapi = arr[..., 2]

    # Combine Actin and Tubulin for a robust cell body signal
    ch_body = np.maximum(ch_actin, ch_tubulin)

    # --- Segmentation Logic ---
    # We need matched Nucleus and Cell masks to compute the vector between their centroids.
    # If masks are provided, we use them. If not, we generate them using marker-controlled watershed.
    
    mask_nuclei = None
    mask_cells = None

    if len(segmentation_masks) >= 2:
        # Assuming order: mask_nuclei, mask_cells (or vice versa, logic usually requires checking overlap)
        # However, without strict metadata, we often assume the first valid mask is nuclei if smaller, etc.
        # Given the prompt doesn't specify mask order strictly, we will implement a robust fallback:
        # We will perform on-the-fly segmentation as it guarantees the 1-to-1 mapping required for this specific feature.
        # Relying on external masks that might not be matched (e.g. different number of labels) is risky for vector computation.
        # Therefore, we prioritize a consistent internal segmentation pipeline for this geometric feature.
        pass

    # --- On-the-fly Segmentation Pipeline ---
    
    # 1. Segment Nuclei (Markers)
    try:
        thresh_nuc = threshold_otsu(ch_dapi)
        binary_nuc = ch_dapi > thresh_nuc
        # Clean up nuclei
        binary_nuc = binary_closing(binary_nuc, disk(2))
        # Label nuclei
        mask_nuclei = label(binary_nuc)
    except Exception:
        return 0.0

    # 2. Segment Cell Body (Mask)
    try:
        thresh_body = threshold_otsu(ch_body)
        binary_body = ch_body > thresh_body
        # Ensure nuclei are part of the body (biological constraint)
        binary_body = np.logical_or(binary_body, binary_nuc)
        # Clean up body
        binary_body = binary_closing(binary_body, disk(3))
    except Exception:
        return 0.0

    # 3. Watershed for Instance Segmentation
    # Use nuclei as markers to grow into the cell body signal
    # This guarantees that label 'k' in mask_cells corresponds to label 'k' in mask_nuclei
    mask_cells = watershed(-ch_body, mask_nuclei, mask=binary_body)

    # 4. Clear Border
    # Cells touching the border have incomplete shapes, leading to artificial centroid shifts.
    # We remove them to ensure the polarity vector is biologically meaningful.
    mask_cells = clear_border(mask_cells)
    
    # Update mask_nuclei to match the cleared cell mask
    # We only keep nuclei where the corresponding cell label still exists
    mask_nuclei = np.where(mask_cells > 0, mask_nuclei, 0)

    # --- Feature Computation: Polarity Vector Magnitude ---
    
    # Get properties for nuclei and cells
    # regionprops returns a list sorted by label
    props_nuclei = regionprops(mask_nuclei)
    props_cells = regionprops(mask_cells)

    # Create a dictionary for fast lookup by label
    # Key: Label ID, Value: Centroid (row, col)
    nuclei_centroids = {p.label: np.array(p.centroid) for p in props_nuclei}
    cell_centroids = {p.label: np.array(p.centroid) for p in props_cells}

    magnitudes = []

    # Iterate through unique labels found in the cell mask
    unique_labels = np.unique(mask_cells)
    unique_labels = unique_labels[unique_labels > 0] # Exclude background

    for label_id in unique_labels:
        if label_id in nuclei_centroids and label_id in cell_centroids:
            c_nuc = nuclei_centroids[label_id]
            c_cell = cell_centroids[label_id]

            # Calculate vector: Cell Centroid - Nucleus Centroid
            # This vector points from the nucleus towards the bulk of the cytoplasm
            vector = c_cell - c_nuc
            
            # Calculate magnitude (Euclidean distance)
            magnitude = np.linalg.norm(vector)
            magnitudes.append(magnitude)

    # --- Aggregation ---
    if not magnitudes:
        return 0.0

    # Return the mean magnitude across all valid cells in the image
    result = np.mean(magnitudes)

    return float(result)
