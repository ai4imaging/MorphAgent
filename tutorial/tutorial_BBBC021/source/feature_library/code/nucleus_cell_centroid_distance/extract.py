def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.segmentation import watershed, clear_border
    from skimage.morphology import binary_closing, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract channels based on dataset description
    # Channel 0: Actin (Red) -> Cytoskeleton
    # Channel 1: Tubulin (Green) -> Cytoskeleton
    # Channel 2: DAPI (Blue) -> Nucleus
    ch_actin = arr[..., 0]
    ch_tubulin = arr[..., 1]
    ch_dapi = arr[..., 2]

    # Combine Actin and Tubulin for a robust cell body signal
    ch_cell = ch_actin + ch_tubulin

    # --- Segmentation Logic ---
    nuclei_labels = None
    cell_labels = None

    # Check if valid segmentation masks are provided
    # We expect at least two masks for this feature: one for nuclei, one for cells
    # If only one is provided, we can't reliably compute this feature without generating the other
    if len(segmentation_masks) >= 2:
        # Assume the first two masks correspond to nuclei and cells respectively, 
        # or try to infer based on size/overlap if possible. 
        # Given the prompt doesn't specify order, we will rely on a fallback if the provided masks are not clearly distinct or labeled.
        # However, usually, masks are passed in a specific order. Let's try to use them.
        # A common convention is nuclei first, then cells.
        
        mask1 = segmentation_masks[0]
        mask2 = segmentation_masks[1]
        
        if mask1 is not None and mask2 is not None:
            # Ensure they are labeled integer arrays
            if mask1.ndim == 2 and mask2.ndim == 2:
                # Heuristic: Nuclei are usually smaller and contained within cells.
                # Let's just assume mask1=nuclei, mask2=cells for now, but verify containment later.
                nuclei_labels = mask1.astype(int)
                cell_labels = mask2.astype(int)

    # --- Fallback Segmentation (if masks not provided or invalid) ---
    if nuclei_labels is None or cell_labels is None:
        # 1. Nuclei Segmentation (Channel 2 - DAPI)
        # Smooth to reduce noise
        dapi_smooth = ndimage.gaussian_filter(ch_dapi, sigma=2)
        try:
            thresh_nuc = threshold_otsu(dapi_smooth)
        except ValueError: # Handle empty images
            thresh_nuc = 0
        
        mask_nuc = dapi_smooth > thresh_nuc
        # Clean up binary mask
        mask_nuc = binary_closing(mask_nuc, disk(2))
        # Label nuclei
        nuclei_labels = label(mask_nuc)
        
        # 2. Cell Segmentation (Watershed seeded by Nuclei)
        # Smooth cell channel
        cell_smooth = ndimage.gaussian_filter(ch_cell, sigma=2)
        try:
            thresh_cell = threshold_otsu(cell_smooth)
        except ValueError:
            thresh_cell = 0
            
        mask_cell = cell_smooth > thresh_cell
        
        # Use watershed to separate cells, seeded by nuclei
        # This ensures 1-to-1 correspondence: Label 1 in nuclei_labels corresponds to Label 1 in cell_labels
        cell_labels = watershed(-cell_smooth, nuclei_labels, mask=mask_cell)

    # --- Feature Calculation ---
    
    # Get properties
    props_nuc = regionprops(nuclei_labels)
    props_cell = regionprops(cell_labels)
    
    # Create dictionaries for fast lookup by label
    # Note: regionprops excludes label 0 (background)
    nuc_centroids = {p.label: np.array(p.centroid) for p in props_nuc}
    cell_centroids = {p.label: np.array(p.centroid) for p in props_cell}
    
    distances = []

    # Iterate through nuclei to find matching cells
    # If we used the watershed fallback, labels match directly.
    # If we used provided masks, labels might not match directly (e.g. Nucleus 5 inside Cell 3).
    
    # We need a robust way to link them.
    # Strategy: For each nucleus, find the cell label at the nucleus centroid location.
    
    for nuc_label, nuc_centroid in nuc_centroids.items():
        r, c = int(nuc_centroid[0]), int(nuc_centroid[1])
        
        # Ensure coordinates are within bounds
        if 0 <= r < cell_labels.shape[0] and 0 <= c < cell_labels.shape[1]:
            # Identify which cell this nucleus belongs to
            target_cell_label = cell_labels[r, c]
            
            # If it falls on background (0) or the cell label doesn't exist in our props, skip
            if target_cell_label > 0 and target_cell_label in cell_centroids:
                cell_centroid = cell_centroids[target_cell_label]
                
                # Calculate Euclidean distance
                dist = np.linalg.norm(nuc_centroid - cell_centroid)
                distances.append(dist)

    # --- Aggregation ---
    if not distances:
        return 0.0
    
    # Return the mean distance across all valid cells in the image
    result = np.mean(distances)

    return float(result)
