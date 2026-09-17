def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops
    from skimage.morphology import binary_closing, disk

    # 1. Data Loading and Validation
    # The image is expected to be (512, 512, 3) uint8
    arr = np.asarray(img)
    
    if arr.ndim != 3 or arr.shape[2] != 3:
        # Fallback for unexpected dimensions, though dataset spec says (H, W, 3)
        return 0.0

    # 2. Channel Extraction
    # Channel 0: Actin (Cytoskeleton) - Red
    # Channel 1: Tubulin (Microtubules) - Green
    # Channel 2: DAPI (Nucleus) - Blue
    actin_ch = arr[..., 0]
    tubulin_ch = arr[..., 1]
    dapi_ch = arr[..., 2]

    # 3. Segmentation Logic
    # We need to define the Nucleus area and the Total Cell area.
    # Total Cell area is typically defined by the union of Cytoskeleton (Actin/Tubulin) and Nucleus signals.
    
    # Check if pre-computed segmentation masks are provided
    # The prompt mentions masks might be available, but we must handle the case where they are not.
    # If masks are provided, we assume a standard order or check their properties. 
    # However, without specific metadata on which mask is which, it's safer to compute fresh masks 
    # from the specific channels known to be present (DAPI, Actin, Tubulin) to ensure accuracy for this specific feature.
    # If a robust instance segmentation mask was passed (e.g. labeled cells), we could use it, 
    # but often "segmentation_masks" might just be binary probability maps.
    # Given the instruction to implement the feature logic, I will implement a robust threshold-based segmentation pipeline.

    # --- Nucleus Segmentation (DAPI) ---
    # Smooth slightly to reduce noise
    dapi_smooth = ndimage.gaussian_filter(dapi_ch.astype(float), sigma=2)
    try:
        thresh_nuc = threshold_otsu(dapi_smooth)
    except ValueError: # Handle empty images
        thresh_nuc = 0
    
    mask_nuc = dapi_smooth > thresh_nuc
    # Fill holes to make solid objects
    mask_nuc = ndimage.binary_fill_holes(mask_nuc)
    # Label nuclei to handle them as individual objects
    labeled_nuc, num_nuc = label(mask_nuc, return_num=True)

    # --- Cell Body Segmentation (Actin + Tubulin + Nucleus) ---
    # Combine channels for a "whole cell" signal
    # Max projection across channels is often effective for defining the cellular footprint
    cell_signal = np.maximum(actin_ch, tubulin_ch)
    # Include DAPI in cell signal to ensure nucleus is contained in cell
    cell_signal = np.maximum(cell_signal, dapi_ch)
    
    cell_smooth = ndimage.gaussian_filter(cell_signal.astype(float), sigma=2)
    try:
        thresh_cell = threshold_otsu(cell_smooth)
    except ValueError:
        thresh_cell = 0
        
    mask_cell = cell_smooth > thresh_cell
    # Morphological closing to connect fragmented cytoskeleton
    mask_cell = binary_closing(mask_cell, disk(3))
    mask_cell = ndimage.binary_fill_holes(mask_cell)

    # 4. Feature Computation: Nucleus to Cell Area Ratio
    # Strategy:
    # We need to associate nuclei with cell bodies.
    # In a simple 2D projection without instance segmentation of touching cells, 
    # the "Cell" area corresponding to a specific nucleus is hard to define exactly if cells touch.
    # However, the feature is "nucleus_to_cell_area_ratio_mean".
    # Approach A: Global Ratio (Total Nuc Area / Total Cell Area) - robust but less informative about population variance.
    # Approach B: Per-object Ratio. We can use the nuclear labels as seeds to partition the cell mask (Watershed).
    
    if num_nuc == 0:
        return 0.0

    # Use Watershed to partition the cell mask based on nuclei seeds
    # This assigns every pixel in the cell mask to the nearest nucleus
    # Distance transform for watershed
    distance = ndimage.distance_transform_edt(mask_cell)
    
    # We use the labeled nuclei as markers. 
    # We need to ensure markers are within the mask_cell.
    markers = labeled_nuc.copy()
    # If a nucleus is outside the cell mask (unlikely due to construction), clip it.
    markers[~mask_cell] = 0
    
    # Watershed
    # In skimage, watershed works on the negative distance or gradient.
    # We want to expand from nuclei to fill the cell mask.
    from skimage.segmentation import watershed
    labeled_cells = watershed(-distance, markers, mask=mask_cell)

    # Now we have:
    # labeled_nuc: labeled nuclei
    # labeled_cells: labeled cell bodies corresponding to those nuclei
    
    ratios = []
    
    # Iterate through properties
    # We can use regionprops on the labeled_cells.
    # Since labeled_cells indices match labeled_nuc indices (from watershed seeding),
    # we can compute areas directly.
    
    props_cell = regionprops(labeled_cells)
    
    # Create a lookup for nucleus areas
    # It's faster to just measure them.
    # Note: regionprops ignores background (0), so index 0 in props corresponds to label 1.
    
    # Get areas for nuclei
    # We need to be careful: watershed might have merged or changed labels if seeds were weird, 
    # but usually it preserves the seed labels.
    # Let's compute areas explicitly to be safe.
    
    # Optimization: bincount is much faster than regionprops for just area
    # Counts pixels for each label value.
    # minlength ensures we cover all labels up to the max label
    max_label = max(labeled_nuc.max(), labeled_cells.max())
    
    nuc_areas = np.bincount(labeled_nuc.ravel(), minlength=max_label+1)
    cell_areas = np.bincount(labeled_cells.ravel(), minlength=max_label+1)
    
    # Iterate over valid labels (1 to max_label)
    valid_ratios = []
    for i in range(1, max_label + 1):
        c_area = cell_areas[i]
        n_area = nuc_areas[i]
        
        # Filter artifacts
        # A valid cell must have some area and a valid nucleus
        if c_area > 10 and n_area > 5:
            # The nucleus area should theoretically be <= cell area because cell mask includes nucleus signal
            # However, due to thresholding differences, n_area could slightly exceed c_area partition locally.
            # We clamp the ratio at 1.0.
            ratio = n_area / c_area
            if ratio > 1.0:
                ratio = 1.0
            valid_ratios.append(ratio)

    if not valid_ratios:
        return 0.0

    result = np.mean(valid_ratios)
    return float(result)
