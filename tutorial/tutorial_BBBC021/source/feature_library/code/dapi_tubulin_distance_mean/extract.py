def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import regionprops
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), Channels: 0=Actin, 1=Tubulin, 2=DAPI
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract relevant channels
    # Channel 1: Tubulin (Green) - for intensity weighted center
    # Channel 2: DAPI (Blue) - for nuclear centroid (fallback if no nuclear mask)
    tubulin_ch = arr[:, :, 1]
    dapi_ch = arr[:, :, 2]

    # Normalize channels to [0, 1] for robust center of mass calculation
    def normalize_ch(ch):
        vmax = np.percentile(ch, 99.5) if ch.size > 0 else 1.0
        if vmax <= 0: vmax = 1.0
        return np.clip(ch / vmax, 0.0, 1.0)

    tubulin_ch = normalize_ch(tubulin_ch)
    # DAPI normalization is useful if we need to use it for intensity-based centroid
    dapi_ch = normalize_ch(dapi_ch)

    # Handle segmentation masks
    # We expect at least one mask (cells) to define the cellular boundaries.
    # Ideally, we have two: cells and nuclei.
    # Common convention in these datasets: mask 0 is often nuclei, mask 1 is cells, or vice versa.
    # We need to heuristically determine which is which or use a robust fallback.
    
    cells_mask = None
    nuclei_mask = None

    if len(segmentation_masks) > 0:
        # Heuristic: usually the mask with larger average object size is the cell mask, 
        # and the one with smaller objects is the nuclei mask.
        # Or simply: if 2 masks, assume one is nuclei and one is cells.
        
        masks = [np.asarray(m, dtype=np.int32) for m in segmentation_masks]
        
        if len(masks) == 1:
            # Only one mask provided. Assume it's the cell mask (cytoplasm).
            # We will infer nuclear position from DAPI intensity within this cell mask.
            cells_mask = masks[0]
        elif len(masks) >= 2:
            # Two masks. Let's try to distinguish them.
            # Calculate total area of foreground
            area_0 = np.count_nonzero(masks[0])
            area_1 = np.count_nonzero(masks[1])
            
            # Usually cell mask covers more area than nuclei mask
            if area_0 > area_1:
                cells_mask = masks[0]
                nuclei_mask = masks[1]
            else:
                cells_mask = masks[1]
                nuclei_mask = masks[0]
    else:
        # No segmentation provided. 
        # We cannot compute a "per cell" metric accurately without segmentation.
        # Fallback: Treat the entire image as one "cell" (not ideal but returns a value)
        # or return 0.0. Returning 0.0 is safer to indicate feature failure.
        return 0.0

    # Ensure masks match image shape (2D)
    if cells_mask.shape != arr.shape[:2]:
        return 0.0
    if nuclei_mask is not None and nuclei_mask.shape != arr.shape[:2]:
        nuclei_mask = None # Discard invalid mask

    # Get unique cell labels
    cell_labels = np.unique(cells_mask)
    cell_labels = cell_labels[cell_labels > 0] # Remove background

    if len(cell_labels) == 0:
        return 0.0

    # Strategy:
    # 1. Calculate Tubulin Center of Mass (weighted by intensity) for each cell.
    # 2. Calculate Nuclear Centroid for each cell.
    #    - If nuclei_mask exists: find the nucleus associated with the cell and get its geometric centroid.
    #    - If no nuclei_mask: use DAPI intensity weighted center within the cell mask.
    
    # Pre-calculate centers for efficiency
    # ndimage.center_of_mass(input, labels, index)
    # input: weights (intensity), labels: mask, index: label IDs
    
    # 1. Tubulin Centers (Intensity Weighted)
    # Returns list of (row, col) tuples
    tubulin_centers = ndimage.center_of_mass(tubulin_ch, cells_mask, cell_labels)
    
    # 2. Nuclear Centers
    nuclear_centers = []
    
    if nuclei_mask is not None:
        # We need to map cell labels to nuclei labels.
        # A cell contains a nucleus.
        # For each cell label, find the corresponding nucleus label.
        
        # Optimization: Instead of masking for every cell, iterate once.
        # However, simple overlap is robust.
        
        # Let's get properties of nuclei to quickly access centroids by label
        nuc_props = regionprops(nuclei_mask)
        nuc_centroids_map = {p.label: p.centroid for p in nuc_props} # label -> (row, col)
        
        for cell_idx, cell_lbl in enumerate(cell_labels):
            # Find which nucleus is inside this cell.
            # Create a mask for the current cell
            # This can be slow if many cells. 
            # Faster approach: Look at the nuclei_mask where cells_mask == cell_lbl
            # But we can't easily slice without bounding box.
            
            # Let's use the bounding box of the cell to speed up search
            slices = ndimage.find_objects(cells_mask == cell_lbl)
            if not slices:
                nuclear_centers.append(None)
                continue
                
            sl = slices[0]
            # Extract local masks
            local_cell_mask = (cells_mask[sl] == cell_lbl)
            local_nuc_mask = nuclei_mask[sl]
            
            # Find nuclei labels inside this cell region
            # We look for the most frequent nucleus label inside the cell area
            # (excluding background 0)
            intersecting_nuclei = local_nuc_mask[local_cell_mask]
            intersecting_nuclei = intersecting_nuclei[intersecting_nuclei > 0]
            
            if intersecting_nuclei.size > 0:
                # Find mode (most common nucleus label)
                # np.bincount is fast for integers
                counts = np.bincount(intersecting_nuclei)
                dom_nuc_label = np.argmax(counts)
                
                if dom_nuc_label in nuc_centroids_map:
                    nuclear_centers.append(nuc_centroids_map[dom_nuc_label])
                else:
                    # Fallback if label not in props (weird edge case)
                    nuclear_centers.append(None)
            else:
                # No nucleus found in this cell
                nuclear_centers.append(None)
                
    else:
        # No nuclei mask, use DAPI intensity centroid within cell mask
        # This assumes the nucleus is the brightest part of DAPI in the cell
        nuclear_centers = ndimage.center_of_mass(dapi_ch, cells_mask, cell_labels)

    # Calculate Distances
    distances = []
    
    # tubulin_centers is a list of tuples or a single tuple if len=1. 
    # ndimage returns list of tuples if index is a sequence.
    if isinstance(tubulin_centers, tuple) and not isinstance(tubulin_centers[0], tuple):
         # Single object case, wrap in list
         tubulin_centers = [tubulin_centers]
    
    # Same for nuclear_centers if it came from ndimage
    if isinstance(nuclear_centers, tuple) and not isinstance(nuclear_centers[0], tuple):
         nuclear_centers = [nuclear_centers]

    for i in range(len(cell_labels)):
        t_cen = tubulin_centers[i]
        n_cen = nuclear_centers[i]
        
        if t_cen is None or n_cen is None:
            continue
            
        # Check for NaN (can happen if intensity is 0)
        if np.any(np.isnan(t_cen)) or np.any(np.isnan(n_cen)):
            continue
            
        # Euclidean distance
        dist = np.sqrt((t_cen[0] - n_cen[0])**2 + (t_cen[1] - n_cen[1])**2)
        distances.append(dist)

    if not distances:
        return 0.0

    result = np.mean(distances)
    return float(result)
