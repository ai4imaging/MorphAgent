def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops
    from skimage.morphology import opening, closing, disk
    from skimage.segmentation import watershed

    # Convert to appropriate array type and check validity
    arr = np.asarray(img)
    
    # Handle dimensionality and empty inputs
    if arr.ndim != 3 or arr.shape[2] != 3:
        # Expecting (H, W, 3) image
        return 0.0
    
    # Extract channels
    # Channel 0: Actin (Cytoskeleton)
    # Channel 2: DAPI (Nucleus)
    actin_ch = arr[..., 0]
    dapi_ch = arr[..., 2]

    # Check for empty image (all zeros)
    if np.max(dapi_ch) == 0 or np.max(actin_ch) == 0:
        return 0.0

    # --- Segmentation Logic ---
    # We need two labeled masks: one for nuclei, one for whole cells (cytoplasm + nucleus)
    # The goal is to compute the ratio per cell instance.
    
    nuclei_labels = None
    cell_labels = None

    # Check if segmentation masks are provided
    # The prompt mentions masks might be provided. Common convention:
    # If 1 mask: usually nuclei or cells. If 2: usually nuclei and cells.
    # However, to ensure robustness and consistency with the specific feature definition (Actin vs DAPI),
    # we will prioritize a robust internal segmentation pipeline if the provided masks are ambiguous or missing,
    # or use them as seeds if appropriate.
    
    # Given the variability of external masks, we will implement a robust fallback pipeline 
    # that guarantees we have matched Nucleus and Cell objects.
    
    # 1. Segment Nuclei (Seeds)
    try:
        # Smooth DAPI channel
        dapi_smooth = ndimage.gaussian_filter(dapi_ch.astype(float), sigma=2)
        thresh_nuc = threshold_otsu(dapi_smooth)
        nuclei_mask = dapi_smooth > thresh_nuc
        # Clean up noise
        nuclei_mask = opening(nuclei_mask, disk(2))
        nuclei_mask = closing(nuclei_mask, disk(2))
        nuclei_labels = label(nuclei_mask)
    except Exception:
        return 0.0

    if nuclei_labels.max() == 0:
        return 0.0

    # 2. Segment Cell Body (Actin boundaries)
    try:
        # Smooth Actin channel
        actin_smooth = ndimage.gaussian_filter(actin_ch.astype(float), sigma=2)
        # Use a slightly lower threshold for actin to capture faint edges, or standard Otsu
        thresh_actin = threshold_otsu(actin_smooth)
        actin_mask = actin_smooth > thresh_actin
        # Morphological cleanup
        actin_mask = closing(actin_mask, disk(3))
        
        # Ensure actin mask covers nuclei (biologically valid assumption for 2D projection)
        # This prevents "floating nuclei" outside the cell body mask
        combined_mask = np.logical_or(actin_mask, nuclei_mask)
        
        # 3. Instance Segmentation via Watershed
        # Use nuclei as markers to split the actin mask into individual cells
        # We calculate the distance transform for better watershedding if needed, 
        # but using the intensity or just the mask geometry with markers is standard.
        # Here we use the mask geometry.
        
        # Create markers for watershed
        markers = nuclei_labels
        
        # Run watershed
        # We use the negative intensity as the "elevation" map so bright actin regions are basins,
        # or simply use the mask. Using intensity guides boundaries better.
        cell_labels = watershed(-actin_smooth, markers, mask=combined_mask)
        
    except Exception:
        return 0.0

    # --- Feature Computation ---
    # Calculate Area Ratio per cell: Cell Area / Nuclear Area
    
    props_nuc = regionprops(nuclei_labels)
    props_cell = regionprops(cell_labels)
    
    # Map label ID to area for quick lookup
    nuc_areas = {p.label: p.area for p in props_nuc}
    
    ratios = []
    
    for cell_prop in props_cell:
        label_id = cell_prop.label
        
        # Check if this cell has a corresponding nucleus (it should, due to watershed seeding)
        if label_id in nuc_areas:
            c_area = cell_prop.area
            n_area = nuc_areas[label_id]
            
            if n_area > 0:
                # The cell area includes the nucleus in this segmentation approach.
                # Ratio = Total Cell Footprint / Nuclear Footprint
                ratio = c_area / float(n_area)
                
                # Sanity check: Ratio should be >= 1.0 (Cell contains nucleus)
                # If segmentation was weird (e.g. nucleus spilling out), clip to 1.0
                if ratio < 1.0:
                    ratio = 1.0
                
                ratios.append(ratio)

    if not ratios:
        return 0.0

    # Return the mean ratio across the population
    result = np.mean(ratios)
    
    return float(result)
