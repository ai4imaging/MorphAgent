def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.segmentation import watershed
    from skimage.morphology import binary_opening, disk

    # Convert to appropriate array type and normalize
    # Image is (512, 512, 3), uint8
    arr = np.asarray(img, dtype=np.float32)
    
    # Handle dimensionality
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not standard 3-channel image, return 0.0
        return 0.0

    # Normalize to [0, 1]
    vmax = np.percentile(arr, 99.5) if arr.size > 0 else 1.0
    if vmax > 0:
        arr = arr / vmax
    arr = np.clip(arr, 0.0, 1.0)

    # Extract channels
    # Channel 0: Actin (Cytoplasm)
    # Channel 2: DAPI (Nucleus)
    ch_cyto = arr[..., 0]
    ch_nuc = arr[..., 2]

    # Initialize masks
    nuclei_labels = None
    cell_labels = None

    # Logic to obtain labeled masks for Nuclei and Cells
    # We need a 1-to-1 mapping between a nucleus and its cell body to compare centroids.
    
    if len(segmentation_masks) >= 2:
        # Strategy: If 2 masks provided, assume one is nuclei and one is cells.
        # We need to identify which is which. Usually nuclei are smaller and contained within cells.
        mask1 = segmentation_masks[0]
        mask2 = segmentation_masks[1]
        
        # Ensure they are labeled integer arrays
        if mask1.ndim == 3: mask1 = mask1.max(axis=0) # Handle potential 3D masks
        if mask2.ndim == 3: mask2 = mask2.max(axis=0)
        
        lbl1 = label(mask1) if mask1.max() <= 1 else mask1.astype(int)
        lbl2 = label(mask2) if mask2.max() <= 1 else mask2.astype(int)
        
        # Heuristic: Nuclei usually cover less area
        area1 = np.sum(lbl1 > 0)
        area2 = np.sum(lbl2 > 0)
        
        if area1 < area2:
            nuclei_labels = lbl1
            cell_labels = lbl2
        else:
            nuclei_labels = lbl2
            cell_labels = lbl1

    elif len(segmentation_masks) == 1:
        # Strategy: Assume provided mask is Nuclei (most common in this dataset context),
        # and generate Cell mask via watershed on Actin channel.
        mask = segmentation_masks[0]
        if mask.ndim == 3: mask = mask.max(axis=0)
        nuclei_labels = label(mask) if mask.max() <= 1 else mask.astype(int)
        
        # Generate cell mask
        try:
            thresh_cyto = threshold_otsu(ch_cyto)
        except:
            thresh_cyto = 0.1
        binary_cyto = ch_cyto > thresh_cyto
        
        # Watershed to propagate nuclei labels to cell boundaries
        cell_labels = watershed(-ch_cyto, nuclei_labels, mask=binary_cyto)

    else:
        # Strategy: No masks provided. Generate both from scratch.
        # 1. Detect Nuclei
        try:
            thresh_nuc = threshold_otsu(ch_nuc)
        except:
            thresh_nuc = 0.1
        binary_nuc = ch_nuc > thresh_nuc
        binary_nuc = binary_opening(binary_nuc, disk(2))
        nuclei_labels = label(binary_nuc)
        
        # 2. Detect Cytoplasm
        try:
            thresh_cyto = threshold_otsu(ch_cyto)
        except:
            thresh_cyto = 0.1
        binary_cyto = ch_cyto > thresh_cyto
        
        # 3. Watershed for Cell Labels (using nuclei as seeds)
        # This ensures 1-to-1 mapping: Cell ID 1 contains Nucleus ID 1
        cell_labels = watershed(-ch_cyto, nuclei_labels, mask=binary_cyto)

    # If segmentation failed completely
    if nuclei_labels is None or cell_labels is None or nuclei_labels.max() == 0:
        return 0.0

    # Compute Centroids
    # We use regionprops to get centroids.
    # Note: If we used watershed with nuclei as markers, the labels match (ID 1 in nuclei matches ID 1 in cell).
    # If we used provided masks that weren't generated together, IDs might not match.
    # We will implement a robust matching strategy: for each nucleus, find the enclosing cell.

    nuc_props = regionprops(nuclei_labels)
    cell_props = regionprops(cell_labels)
    
    # Create a map for cell properties for fast lookup if IDs match, 
    # but we'll use spatial matching to be safe.
    # However, spatial matching is O(N*M). 
    # Optimization: If we generated cell_labels via watershed from nuclei_labels, IDs match perfectly.
    # If masks came from external source, we check overlap.
    
    # Let's assume the standard case where we want to measure valid cells.
    distances = []

    # Map cell labels to their props for O(1) access
    cell_prop_map = {prop.label: prop for prop in cell_props}

    for n_prop in nuc_props:
        n_label = n_prop.label
        n_centroid = np.array(n_prop.centroid) # (row, col)
        
        target_cell_prop = None
        
        # Case A: IDs match (Watershed approach)
        if n_label in cell_prop_map:
            # Verify containment roughly (centroid of nucleus should be inside bounding box of cell)
            # or just trust the ID if it came from watershed.
            # To be robust against mismatched external masks:
            c_prop = cell_prop_map[n_label]
            
            # Check if this cell actually overlaps/contains the nucleus
            # We can check if the nucleus centroid is within the cell mask
            # int coordinates
            nr, nc = int(n_centroid[0]), int(n_centroid[1])
            if 0 <= nr < cell_labels.shape[0] and 0 <= nc < cell_labels.shape[1]:
                if cell_labels[nr, nc] == c_prop.label:
                    target_cell_prop = c_prop
        
        # Case B: IDs don't match or validation failed (External mismatched masks)
        if target_cell_prop is None:
            nr, nc = int(n_centroid[0]), int(n_centroid[1])
            if 0 <= nr < cell_labels.shape[0] and 0 <= nc < cell_labels.shape[1]:
                enclosing_cell_id = cell_labels[nr, nc]
                if enclosing_cell_id > 0 and enclosing_cell_id in cell_prop_map:
                    target_cell_prop = cell_prop_map[enclosing_cell_id]

        if target_cell_prop is not None:
            c_centroid = np.array(target_cell_prop.centroid)
            
            # Euclidean distance
            dist = np.linalg.norm(n_centroid - c_centroid)
            distances.append(dist)

    if not distances:
        return 0.0

    # Return the mean distance across all valid cells in the image
    result = np.mean(distances)
    return float(result)
