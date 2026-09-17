def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu, threshold_triangle
    from skimage.morphology import closing, square, remove_small_objects
    from skimage.segmentation import watershed

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # Fallback for unexpected shapes (e.g., if channels are first or 2D)
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        elif arr.ndim == 2:
            # If 2D, we can't distinguish nucleus/cyto channels properly, return 0
            return 0.0
        else:
            return 0.0

    # Extract channels based on dataset description
    # Ch0: Actin (Cyto), Ch1: Tubulin (Cyto), Ch2: DAPI (Nucleus)
    # We combine Ch0 and Ch1 for a robust cytoplasm signal
    cyto_signal = np.maximum(arr[..., 0], arr[..., 1])
    nuc_signal = arr[..., 2]

    # Normalize signals to [0, 1] for processing
    def normalize(c):
        vmax = np.percentile(c, 99.5) if c.size > 0 else 1.0
        if vmax > 0:
            c = c / vmax
        return np.clip(c, 0.0, 1.0)

    cyto_signal = normalize(cyto_signal)
    nuc_signal = normalize(nuc_signal)

    # --- Segmentation Logic ---
    # We need matched instance labels for nuclei and cells to compute per-cell ratios.
    
    nuc_labels = None
    cell_labels = None

    # Check if valid segmentation masks are provided
    # We expect at least two masks to do this reliably if provided externally:
    # one for nuclei, one for cells.
    if len(segmentation_masks) >= 2:
        # Assuming mask 0 is cells/cyto and mask 1 is nuclei, or vice versa.
        # However, without strict metadata on mask order, on-the-fly segmentation 
        # is often more robust for this specific ratio metric to ensure 1-to-1 mapping.
        # If the provided masks are just binary or not instance-matched, 
        # calculating the ratio per cell is difficult.
        # Given the "optional" nature and potential variability, we will prioritize
        # a robust internal watershed segmentation to guarantee matched instances.
        pass 

    # Perform On-the-fly Segmentation (Robust Fallback & Primary Method for Consistency)
    # 1. Segment Nuclei
    try:
        thresh_nuc = threshold_otsu(nuc_signal)
        nuc_mask = nuc_signal > thresh_nuc
        # Clean up nucleus mask
        nuc_mask = closing(nuc_mask, square(3))
        nuc_mask = remove_small_objects(nuc_mask, min_size=50)
        nuc_labels = label(nuc_mask)
    except Exception:
        return 0.0

    # 2. Segment Cytoplasm / Cells using Watershed
    # Use nuclei as seeds to ensure every cell has exactly one nucleus
    try:
        # Threshold cytoplasm (Triangle is often better for faint tails than Otsu)
        thresh_cyto = threshold_triangle(cyto_signal[cyto_signal > 0])
        cyto_mask = cyto_signal > thresh_cyto
        
        # Ensure nucleus is part of the cell
        combined_mask = np.logical_or(cyto_mask, nuc_mask)
        combined_mask = remove_small_objects(combined_mask, min_size=100)

        # Watershed
        # We use the inverse intensity as the topographic surface
        # Seeds are the nucleus labels
        # Mask restricts the growth to the combined binary mask
        cell_labels = watershed(-cyto_signal, nuc_labels, mask=combined_mask)
    except Exception:
        return 0.0

    # --- Feature Computation ---
    
    if nuc_labels is None or cell_labels is None or nuc_labels.max() == 0:
        return 0.0

    # We iterate through properties. Since we used watershed with nuc_labels as seeds,
    # the label indices in cell_labels correspond to the same indices in nuc_labels.
    
    props_nuc = regionprops(nuc_labels)
    props_cell = regionprops(cell_labels)

    # Create a dictionary for fast lookup of cell areas by label
    cell_areas = {p.label: p.area for p in props_cell}

    ratios = []

    for p_nuc in props_nuc:
        label_id = p_nuc.label
        
        # Get corresponding cell area
        if label_id in cell_areas:
            area_nuc = p_nuc.area
            area_total_cell = cell_areas[label_id]
            
            # Cytoplasm area = Total Cell Area - Nucleus Area
            # Note: In watershed, the nucleus pixels are included in the cell label region.
            area_cyto = area_total_cell - area_nuc
            
            # Filter out artifacts where cyto area is impossibly small
            if area_cyto > 5:  # Minimum 5 pixels of cytoplasm to be valid
                ratio = area_nuc / float(area_cyto)
                ratios.append(ratio)

    if not ratios:
        return 0.0

    # Return the mean ratio
    result = np.mean(ratios)
    
    return float(result)
