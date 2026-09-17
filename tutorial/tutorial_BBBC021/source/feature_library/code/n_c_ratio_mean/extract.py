def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops
    from skimage.segmentation import watershed
    from skimage.morphology import remove_small_objects, opening, disk

    # Convert to appropriate array type and handle dimensionality
    # Dataset: (512, 512, 3), uint8. Channels: 0=Actin, 1=Tubulin, 2=DAPI
    img_arr = np.asarray(img)
    
    # Basic validation
    if img_arr.ndim != 3 or img_arr.shape[2] != 3:
        return 0.0

    # Extract channels
    # Channel 0 (Actin) is best for Cytoplasm
    # Channel 2 (DAPI) is best for Nuclei
    ch_actin = img_arr[..., 0]
    ch_dapi = img_arr[..., 2]

    # Initialize masks
    nuclei_mask = None
    cell_mask = None

    # --- SEGMENTATION LOGIC ---
    
    # Check if segmentation masks are provided via arguments
    # The prompt implies masks might be passed. We need to handle 0, 1, or 2 masks.
    # If masks are provided, we assume they are label matrices (instance segmentation)
    # or binary masks.
    
    if len(segmentation_masks) > 0:
        # Heuristic to identify which mask is which if multiple are provided
        # Usually, nuclei are smaller and contained within cells.
        # If only one mask is provided, it's ambiguous, but often it's the nuclei mask in high-content screening.
        # However, for N:C ratio, we need BOTH.
        
        masks = [np.asarray(m) for m in segmentation_masks if m is not None]
        
        if len(masks) >= 2:
            # Assume the one with smaller average object size is nuclei
            # Or assume order (often nuclei, then cell) - but let's measure to be safe
            m1 = masks[0]
            m2 = masks[1]
            
            # Simple area check on non-zero pixels
            area1 = np.count_nonzero(m1)
            area2 = np.count_nonzero(m2)
            
            if area1 < area2:
                nuclei_mask = m1
                cell_mask = m2
            else:
                nuclei_mask = m2
                cell_mask = m1
        elif len(masks) == 1:
            # If only one mask, assume it's nuclei (most common) and we need to segment cytoplasm
            nuclei_mask = masks[0]
            # We will generate cell_mask below using watershed
    
    # --- FALLBACK / REFINEMENT SEGMENTATION ---
    
    # 1. Generate Nuclei Mask if missing
    if nuclei_mask is None:
        # Smooth DAPI channel
        dapi_smooth = ndimage.gaussian_filter(ch_dapi.astype(float), sigma=2)
        try:
            thresh_nuc = threshold_otsu(dapi_smooth)
        except ValueError: # Handle empty images
            thresh_nuc = 0
            
        binary_nuc = dapi_smooth > thresh_nuc
        # Clean up
        binary_nuc = remove_small_objects(binary_nuc, min_size=50)
        binary_nuc = opening(binary_nuc, disk(2))
        nuclei_mask = label(binary_nuc)

    # Ensure nuclei_mask is labeled (instance segmentation)
    if nuclei_mask.max() == 1: # If binary
        nuclei_mask = label(nuclei_mask)

    # 2. Generate Cell/Cytoplasm Mask if missing
    if cell_mask is None:
        # Use Actin channel for cell body
        actin_smooth = ndimage.gaussian_filter(ch_actin.astype(float), sigma=2)
        
        # Determine background/foreground for cells
        try:
            thresh_cell = threshold_otsu(actin_smooth)
        except ValueError:
            thresh_cell = 0
            
        # Create a binary mask of where "stuff" is
        # We can combine Actin and weak DAPI signal to be sure we cover the cell
        combined_signal = np.maximum(actin_smooth, ch_dapi.astype(float) * 0.5)
        try:
            thresh_combined = threshold_otsu(combined_signal)
        except:
            thresh_combined = 0
            
        binary_cell_foreground = combined_signal > thresh_combined
        
        # Watershed to split touching cells based on nuclei seeds
        # We use the nuclei_mask as markers
        # We use the inverse intensity as the "basin"
        
        # If no nuclei found, we can't compute N:C ratio
        if nuclei_mask.max() == 0:
            return 0.0
            
        # Markers for watershed must be the nuclei labels
        markers = nuclei_mask
        
        # Watershed
        # mask=binary_cell_foreground ensures we don't flood the background
        cell_mask = watershed(-actin_smooth, markers, mask=binary_cell_foreground)

    # --- FEATURE COMPUTATION ---
    
    # We now have nuclei_mask and cell_mask (both labeled)
    # Ideally, label 'i' in nuclei_mask corresponds to label 'i' in cell_mask
    # If they came from separate files or uncoordinated processes, we need to match them.
    # The watershed method above guarantees matching labels.
    # If masks were provided externally, we must robustly match them.
    
    # Get properties
    props_nuc = regionprops(nuclei_mask)
    props_cell = regionprops(cell_mask)
    
    # Map cell labels to areas for O(1) lookup
    cell_areas = {p.label: p.area for p in props_cell}
    
    ratios = []
    
    for pn in props_nuc:
        nuc_label = pn.label
        nuc_area = pn.area
        
        # Find corresponding cell area
        # 1. Try direct label match (works if watershed was used or masks are synchronized)
        cell_area = cell_areas.get(nuc_label, 0)
        
        # 2. If direct match fails or seems wrong (cell < nucleus), try spatial overlap
        # (This handles cases where provided masks have different label indices)
        if cell_area < nuc_area:
            # Find the cell mask value at the centroid of the nucleus
            cy, cx = map(int, pn.centroid)
            # Boundary check
            cy = min(cy, cell_mask.shape[0]-1)
            cx = min(cx, cell_mask.shape[1]-1)
            
            matched_cell_label = cell_mask[cy, cx]
            if matched_cell_label > 0:
                cell_area = cell_areas.get(matched_cell_label, 0)
        
        # Calculate areas
        # Cytoplasm Area = Whole Cell Area - Nucleus Area
        # If segmentation is imperfect and Cell < Nucleus, we clamp cyto area to a small epsilon
        cyto_area = max(cell_area - nuc_area, 1.0)
        
        # Ratio = Nucleus / Cytoplasm
        # Note: Some definitions use Nucleus / Whole Cell, but "N:C ratio" strictly implies N / C.
        # We will use N / C.
        ratio = nuc_area / cyto_area
        
        # Filter outliers (e.g., infinite ratios from naked nuclei or tiny debris)
        if 0.0 < ratio < 10.0: # Reasonable biological range for MCF-7
            ratios.append(ratio)

    if not ratios:
        return 0.0

    return float(np.mean(ratios))
