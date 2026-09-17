def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops

    # Convert to appropriate array type
    # Image is (512, 512, 3), uint8
    arr = np.asarray(img, dtype=np.float32)

    # Check dimensionality
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract Channels
    # Channel 0: Actin (Red)
    # Channel 1: Tubulin (Green) -> This is the intensity we measure
    # Channel 2: DAPI (Blue) -> Nucleus marker
    tubulin_img = arr[..., 1]
    dapi_img = arr[..., 2]
    actin_img = arr[..., 0]

    # Normalize Tubulin intensity for calculation (avoid large numbers, though ratio is scale-invariant)
    # We keep it as float32.
    
    # Define Masks
    nuclei_mask = None
    cells_mask = None

    # Strategy:
    # 1. Try to use provided segmentation masks.
    # 2. If not available, generate them on the fly using DAPI and Actin channels.

    if len(segmentation_masks) >= 2:
        # Assuming order: cells, nuclei or nuclei, cells. 
        # Usually, the smaller area is nuclei.
        m1 = segmentation_masks[0]
        m2 = segmentation_masks[1]
        
        # Simple heuristic to distinguish: Nuclei usually occupy less area than whole cells
        if np.sum(m1 > 0) < np.sum(m2 > 0):
            nuclei_mask = m1
            cells_mask = m2
        else:
            nuclei_mask = m2
            cells_mask = m1
            
    elif len(segmentation_masks) == 1:
        # If only one mask, assume it's nuclei (most common in these datasets) or cells?
        # Let's assume it's nuclei if we have to guess, and infer cytoplasm from Actin.
        # Or if it's cells, we infer nuclei from DAPI.
        # Let's try to generate the missing one.
        mask_in = segmentation_masks[0]
        
        # Generate DAPI threshold
        try:
            thresh_dapi = threshold_otsu(dapi_img)
        except:
            thresh_dapi = 0
        dapi_binary = dapi_img > thresh_dapi
        
        # Check overlap to guess what the input mask is
        # If input mask overlaps heavily with DAPI high intensity, it's likely nuclei.
        overlap = np.sum((mask_in > 0) & dapi_binary)
        total_mask = np.sum(mask_in > 0)
        
        if total_mask > 0 and (overlap / total_mask) > 0.8:
            # Input is likely nuclei
            nuclei_mask = mask_in
            # Generate cell mask from Actin
            try:
                thresh_actin = threshold_otsu(actin_img)
                cells_mask = (actin_img > thresh_actin) | (nuclei_mask > 0) # Cell includes nucleus
                cells_mask = label(cells_mask)
            except:
                cells_mask = nuclei_mask # Fallback: cell = nucleus (ratio will be 1.0 or undefined)
        else:
            # Input is likely cells
            cells_mask = mask_in
            # Use DAPI binary as nuclei, but constrained to be inside cells
            nuclei_mask = dapi_binary & (cells_mask > 0)
            nuclei_mask = label(nuclei_mask)

    else:
        # No masks provided: Generate from scratch
        try:
            # Nuclei from DAPI (Channel 2)
            t_nuc = threshold_otsu(dapi_img)
            nuclei_mask = label(dapi_img > t_nuc)
            
            # Cells from Actin (Channel 0) combined with Nuclei
            t_cell = threshold_otsu(actin_img)
            # Cell body is Actin signal OR Nucleus signal
            cells_binary = (actin_img > t_cell) | (nuclei_mask > 0)
            cells_mask = label(cells_binary)
        except:
            return 0.0

    # Ensure masks are valid
    if nuclei_mask is None or cells_mask is None:
        return 0.0
    
    if np.max(nuclei_mask) == 0:
        return 0.0

    # Compute Ratio per Cell (Instance-based)
    # This is more robust than global average because it handles cell density variations.
    
    # We iterate over nuclei, find corresponding cell body, and compute ratio.
    # To link nuclei to cells, we can look at pixel overlap.
    
    props_nuc = regionprops(nuclei_mask, intensity_image=tubulin_img)
    
    ratios = []
    
    # Create a mapping from pixel coordinate to cell label for fast lookup if needed,
    # but since we have full masks, we can just mask arrays.
    # However, masks might not be perfectly aligned (e.g. different label IDs).
    # Approach: Iterate nuclei. For each nucleus, define nuclear region.
    # Then look at the same region in cell mask to find the cell ID.
    # Then define cytoplasm = (Cell == ID) & (Nucleus != ID).
    
    # Optimization: If masks are consistent (e.g. from same pipeline), label 1 in nuclei corresponds to label 1 in cells.
    # If not, we treat them as binary masks for a global calculation or do a spatial query.
    # Given the "segmentation_masks" input, we can't guarantee ID matching.
    
    # Let's use a robust approach:
    # 1. Binarize nuclei
    binary_nuclei = nuclei_mask > 0
    # 2. Binarize cells
    binary_cells = cells_mask > 0
    # 3. Define Cytoplasm
    binary_cytoplasm = binary_cells & (~binary_nuclei)
    
    # Check if we have valid regions
    nuc_area = np.sum(binary_nuclei)
    cyto_area = np.sum(binary_cytoplasm)
    
    if nuc_area == 0 or cyto_area == 0:
        return 0.0
    
    # Calculate Mean Intensities
    # We use the global mean here as a fallback if instance matching is hard, 
    # but the prompt implies a feature that could be sensitive to individual cell states.
    # However, without guaranteed instance matching between separate mask files, 
    # global average of (Mean Nuc / Mean Cyto) is safer than trying to match potentially disjoint labels.
    
    # Let's try to do it per connected component in the *Nuclei* mask, assuming the immediate surrounding is cytoplasm.
    
    labeled_nuclei = label(binary_nuclei)
    nuc_props = regionprops(labeled_nuclei, intensity_image=tubulin_img)
    
    # We need a way to get local cytoplasm intensity for each nucleus.
    # We can dilate the nucleus mask to approximate a local cytoplasmic ring, restricted by the cell mask.
    
    # Create a ring around each nucleus
    # Dilation size: 5 pixels (arbitrary but standard for "perinuclear/cytoplasmic" sampling)
    steps = 5
    struct = ndimage.generate_binary_structure(2, 1)
    
    cell_ratios = []
    
    for prop in nuc_props:
        # 1. Get mean intensity of nucleus
        mean_nuc = prop.mean_intensity
        
        # 2. Get local cytoplasm
        # Extract bounding box for speed
        minr, minc, maxr, maxc = prop.bbox
        
        # Pad slightly to allow for dilation
        pad = steps + 2
        minr_p = max(0, minr - pad)
        minc_p = max(0, minc - pad)
        maxr_p = min(binary_nuclei.shape[0], maxr + pad)
        maxc_p = min(binary_nuclei.shape[1], maxc + pad)
        
        # Local masks
        local_nuc_mask = labeled_nuclei[minr_p:maxr_p, minc_p:maxc_p] == prop.label
        local_tubulin = tubulin_img[minr_p:maxr_p, minc_p:maxc_p]
        local_cell_mask = binary_cells[minr_p:maxr_p, minc_p:maxc_p]
        
        # Dilate nucleus to find ring
        dilated_nuc = ndimage.binary_dilation(local_nuc_mask, structure=struct, iterations=steps)
        
        # Cytoplasm ring = Dilated Nucleus AND Cell Mask AND NOT Nucleus
        local_cyto_mask = dilated_nuc & local_cell_mask & (~local_nuc_mask)
        
        if np.sum(local_cyto_mask) > 0:
            mean_cyto = np.mean(local_tubulin[local_cyto_mask])
            
            # Avoid division by zero
            if mean_cyto > 1e-6:
                ratio = mean_nuc / mean_cyto
                cell_ratios.append(ratio)
            elif mean_nuc > 0:
                # Cyto is 0, Nuc is > 0 -> Infinite ratio, cap it
                cell_ratios.append(10.0) 
            else:
                # Both 0
                cell_ratios.append(0.0)
                
    if not cell_ratios:
        return 0.0
        
    # Return the median ratio to be robust against outliers (e.g. mitotic cells in a population of interphase)
    return float(np.median(cell_ratios))
