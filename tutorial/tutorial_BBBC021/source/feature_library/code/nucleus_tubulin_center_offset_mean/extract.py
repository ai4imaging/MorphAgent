def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not the expected 3-channel image, return NaN as we cannot reliably identify channels
        return float('nan')

    # Channel Mapping based on dataset description:
    # Channel 0: Actin (Red)
    # Channel 1: Tubulin (Green) - Target for intensity-weighted centroid
    # Channel 2: DAPI (Blue) - Target for Nucleus centroid
    tubulin_ch = arr[:, :, 1]
    dapi_ch = arr[:, :, 2]

    # Handle segmentation masks
    # We need at least a cell mask to define the boundaries of the cell for the tubulin signal.
    # Ideally, we have (nuclei_mask, cell_mask) or just (cell_mask).
    
    if not segmentation_masks:
        return float('nan')
    
    # Determine which masks to use
    # The dataset description suggests masks are passed. 
    # Common convention in these datasets: mask 0 is often nuclei, mask 1 is cells (cytoplasm+nuclei).
    # If only one mask is provided, we assume it's a cell mask (or nuclei mask acting as cell definition).
    
    nuc_mask = None
    cell_mask = None

    # Heuristic to assign masks based on count
    if len(segmentation_masks) >= 2:
        # Assuming order: nuclei, cells (common in CP pipelines)
        # We verify by checking containment: nuclei should be inside cells.
        m0 = segmentation_masks[0]
        m1 = segmentation_masks[1]
        
        # Simple check: usually cell masks cover more area than nuclei masks
        if np.count_nonzero(m1) > np.count_nonzero(m0):
            nuc_mask = m0
            cell_mask = m1
        else:
            nuc_mask = m1
            cell_mask = m0
    elif len(segmentation_masks) == 1:
        # Only one mask. We use it as the cell boundary.
        # For the nucleus center, we will use the intensity-weighted center of the DAPI channel 
        # within this mask, which is a robust approximation since DAPI is localized.
        cell_mask = segmentation_masks[0]
        nuc_mask = None # Will rely on DAPI intensity weighting within cell mask
    
    # Ensure cell_mask is labeled (integers > 0 for objects)
    # If it's binary, label it.
    if cell_mask.max() == 1 and cell_mask.ndim == 2:
        cell_mask, num_features = ndimage.label(cell_mask)
    
    # Get unique cell labels (excluding background 0)
    cell_labels = np.unique(cell_mask)
    cell_labels = cell_labels[cell_labels > 0]
    
    if len(cell_labels) == 0:
        return float('nan')

    # Optimization: Get bounding boxes for all cells to avoid processing full 512x512 arrays per cell
    slices = ndimage.find_objects(cell_mask)
    
    offsets = []

    for label_id in cell_labels:
        sl = slices[label_id - 1] # slices is 0-indexed, labels are 1-based usually
        if sl is None: 
            continue

        # Extract crops for the current cell
        # We use the bounding box slice 'sl'
        mask_crop = (cell_mask[sl] == label_id)
        tubulin_crop = tubulin_ch[sl]
        dapi_crop = dapi_ch[sl]
        
        # 1. Calculate Tubulin Center (Intensity Weighted)
        # We want the center of the microtubule network within the cell
        # Weight by tubulin intensity
        
        # Check if crop has signal
        if tubulin_crop[mask_crop].sum() == 0:
            continue
            
        # center_of_mass returns (y, x) relative to the crop
        # We calculate it using the mask to ensure we only use pixels from this specific cell
        # (handling overlapping bounding boxes)
        
        # Create a masked array where background is 0 for weight calculation
        tubulin_weights = tubulin_crop * mask_crop
        
        try:
            # Center of mass of Tubulin signal
            com_tub = ndimage.center_of_mass(tubulin_weights)
        except Exception:
            continue

        # 2. Calculate Nucleus Center
        # Strategy A: If we have a specific nucleus mask, use the geometric center of the nucleus part
        # Strategy B: If no nucleus mask, use DAPI intensity weighted center within the cell
        
        com_nuc = None
        
        if nuc_mask is not None:
            # Find the nucleus label corresponding to this cell
            # We look at the nuc_mask within the cell's bounding box
            nuc_crop = nuc_mask[sl]
            # Filter by the cell mask to ensure we are inside the current cell
            valid_nuc_pixels = nuc_crop * mask_crop
            
            # Find the most frequent non-zero label in this region
            nuc_labels_in_cell = valid_nuc_pixels[valid_nuc_pixels > 0]
            
            if nuc_labels_in_cell.size > 0:
                # Use bincount to find mode (most frequent label)
                # We assume one nucleus per cell for this calculation
                counts = np.bincount(nuc_labels_in_cell.flatten())
                dom_nuc_label = np.argmax(counts)
                
                # Calculate geometric center of this nucleus label
                # We can use the binary mask of the nucleus
                specific_nuc_mask = (valid_nuc_pixels == dom_nuc_label)
                try:
                    com_nuc = ndimage.center_of_mass(specific_nuc_mask)
                except Exception:
                    pass
        
        # Fallback or Strategy B: Use DAPI intensity center
        if com_nuc is None:
            dapi_weights = dapi_crop * mask_crop
            if dapi_weights.sum() > 0:
                try:
                    com_nuc = ndimage.center_of_mass(dapi_weights)
                except Exception:
                    pass
        
        if com_nuc is None or com_tub is None:
            continue
            
        # 3. Calculate Distance
        # com_nuc and com_tub are both (y, x) tuples relative to the slice 'sl'
        # Since they are in the same coordinate system (the crop), we can calculate distance directly
        dist = np.sqrt((com_nuc[0] - com_tub[0])**2 + (com_nuc[1] - com_tub[1])**2)
        offsets.append(dist)

    if not offsets:
        return float('nan')

    # Return the mean offset across all valid cells
    result = np.mean(offsets)
    
    return float(result)
