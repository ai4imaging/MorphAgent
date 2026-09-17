def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract Channels
    # Channel 1: Tubulin (Green) - used for intensity weighting
    # Channel 2: DAPI (Blue) - used for nucleus geometric center
    tubulin_img = arr[..., 1]
    dapi_img = arr[..., 2]

    # Intensity normalization
    # Normalize tubulin for CoM calculation to avoid overflow and ensure proper weighting
    vmax_tub = np.percentile(tubulin_img, 99.5) if tubulin_img.size > 0 else 1.0
    if vmax_tub > 0:
        tubulin_img = tubulin_img / vmax_tub
    tubulin_img = np.clip(tubulin_img, 0.0, 1.0)

    # Normalize DAPI for segmentation if needed
    vmax_dapi = np.percentile(dapi_img, 99.5) if dapi_img.size > 0 else 1.0
    if vmax_dapi > 0:
        dapi_img_norm = dapi_img / vmax_dapi
    else:
        dapi_img_norm = dapi_img
    dapi_img_norm = np.clip(dapi_img_norm, 0.0, 1.0)

    # Define Masks
    nuclei_mask = None
    cell_mask = None

    # Check if segmentation masks are provided
    # We look for a nuclei mask and a cell/cytoplasm mask
    # If masks are provided, we try to identify which is which based on typical ordering or content
    # Usually, if 2 masks: [0]=cells/cyto, [1]=nuclei OR [0]=nuclei, [1]=cells
    # We will assume if multiple masks, we need to distinguish them.
    # If only one mask, we assume it's nuclei and generate cells, or it's cells and we generate nuclei.
    
    # Strategy:
    # 1. If masks exist, try to use them.
    # 2. If not, generate them.
    
    if len(segmentation_masks) > 0:
        # Simple heuristic: if we have masks, assume the one with more distinct regions or larger average area is cells, smaller is nuclei
        # Or simply rely on the provided order if known. The prompt doesn't strictly guarantee order.
        # Let's try to infer or fallback to generation if ambiguous.
        
        # For this specific task, we need accurate nuclei centers and cell boundaries.
        # Let's try to use the first mask as nuclei if it looks like nuclei (many small objects), 
        # or generate if not confident.
        
        # Given the complexity of robustly identifying mask types without metadata, 
        # and the requirement to work even if masks are None, 
        # we will implement a robust fallback pipeline that uses masks if they seem valid, 
        # but defaults to a standard watershed segmentation if not.
        
        # Let's assume the standard case where masks might be passed.
        # If we have at least one mask, let's assume it's the nuclei mask if we only get one.
        # If we get two, we assume one is nuclei and one is cells.
        
        # However, to ensure the feature is calculated correctly on the *actual* image data provided,
        # and given the high quality of the input images (MCF-7), on-the-fly segmentation is often safer 
        # than relying on potentially mismatched external masks unless guaranteed.
        
        # Let's try to use the first mask as a seed.
        candidate_mask = segmentation_masks[0]
        if candidate_mask is not None and candidate_mask.shape == dapi_img.shape:
             nuclei_mask = candidate_mask
        
        if len(segmentation_masks) > 1:
            candidate_mask_2 = segmentation_masks[1]
            if candidate_mask_2 is not None and candidate_mask_2.shape == dapi_img.shape:
                cell_mask = candidate_mask_2

    # Fallback / Refinement Segmentation Logic
    if nuclei_mask is None:
        # Generate Nuclei Mask from DAPI
        try:
            thresh = threshold_otsu(dapi_img_norm)
            binary_nuclei = dapi_img_norm > thresh
            # Fill holes and label
            binary_nuclei = ndimage.binary_fill_holes(binary_nuclei)
            nuclei_mask = label(binary_nuclei)
        except:
            # Fallback if otsu fails (e.g. empty image)
            nuclei_mask = np.zeros(dapi_img.shape, dtype=int)

    # Ensure nuclei_mask is labeled
    if nuclei_mask.max() == 1: # If binary
        nuclei_mask = label(nuclei_mask)

    # If we still don't have a cell mask, generate one using watershed
    if cell_mask is None:
        # Use nuclei as markers for watershed on the tubulin channel (or inverted distance map)
        # Tubulin signal usually defines the cell body well
        
        # Create markers from nuclei
        markers = nuclei_mask
        
        # Create a "basin" image. We want to flood from nuclei outwards.
        # We can use the inverse of the tubulin intensity (bright tubulin = valleys)
        # Or just distance transform if tubulin is weak. 
        # Tubulin is usually a good signal for cell extent.
        
        # Smooth tubulin slightly
        tubulin_smooth = ndimage.gaussian_filter(tubulin_img, sigma=2)
        
        # Define background marker if possible (low intensity regions)
        # Simple background threshold
        try:
            bg_thresh = threshold_otsu(tubulin_smooth)
            background = tubulin_smooth < bg_thresh
            # Add background as a separate label (max_label + 1) to stop watershed from filling everything
            # However, for this specific feature (displacement), we want the cell mask to capture the tubulin.
            # If we limit it too much, we miss tubulin. If we don't limit, we capture noise.
            # A standard approach is watershedding the gradient or intensity.
            
            # Let's use a mask based on tubulin signal to constrain the watershed
            mask_for_watershed = tubulin_smooth > (bg_thresh * 0.5) # Permissive mask
            mask_for_watershed = np.logical_or(mask_for_watershed, nuclei_mask > 0)
            
            cell_mask = watershed(-tubulin_smooth, markers, mask=mask_for_watershed)
        except:
            cell_mask = nuclei_mask # Fallback: cell is just the nucleus

    # Computation of Feature
    # 1. Get Geometric Center of Nuclei
    # 2. Get Intensity-Weighted Center of Mass of Tubulin for the corresponding cell
    
    # Get properties of nuclei
    nuclei_props = regionprops(nuclei_mask)
    
    displacements = []
    
    # Pre-calculate centers of mass for tubulin across the whole image using the cell mask
    # This is much faster than looping and masking
    # We need a list of indices present in the cell mask
    cell_indices = np.unique(cell_mask)
    cell_indices = cell_indices[cell_indices > 0] # Remove background
    
    if len(cell_indices) == 0:
        return 0.0
        
    # Calculate CoM for all cells at once
    # result is a list of (row, col) tuples
    try:
        coms = ndimage.center_of_mass(tubulin_img, labels=cell_mask, index=cell_indices)
    except:
        return 0.0
        
    # Map cell_label -> (r_tubulin, c_tubulin)
    # Handle case where center_of_mass returns a single tuple if only one index is passed
    if len(cell_indices) == 1:
        coms = [coms]
        
    com_map = {idx: com for idx, com in zip(cell_indices, coms)}
    
    # Iterate through nuclei to link them to cells
    for prop in nuclei_props:
        nuc_label = prop.label
        
        # Get geometric center of nucleus
        nuc_center = prop.centroid # (row, col)
        
        # Find which cell this nucleus belongs to.
        # Ideally, nuc_label matches cell_label if we used nuclei as seeds.
        # If masks were provided externally, they might not match indices.
        # We check the cell mask at the nucleus centroid.
        
        r_int, c_int = int(nuc_center[0]), int(nuc_center[1])
        
        # Boundary check
        if 0 <= r_int < cell_mask.shape[0] and 0 <= c_int < cell_mask.shape[1]:
            cell_id = cell_mask[r_int, c_int]
            
            if cell_id > 0 and cell_id in com_map:
                tub_center = com_map[cell_id]
                
                # Check for NaN (can happen if cell is empty in tubulin channel)
                if np.isnan(tub_center[0]) or np.isnan(tub_center[1]):
                    continue
                
                # Calculate Euclidean distance
                dist = np.sqrt((nuc_center[0] - tub_center[0])**2 + (nuc_center[1] - tub_center[1])**2)
                displacements.append(dist)

    if not displacements:
        return 0.0

    return float(np.mean(displacements))
