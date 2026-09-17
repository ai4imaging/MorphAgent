def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.segmentation import watershed, clear_border
    from skimage.morphology import binary_opening, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract Channels
    # Channel 1: Tubulin (Green) - used for intensity distribution
    # Channel 2: DAPI (Blue) - used for nucleus detection
    tubulin_ch = arr[..., 1]
    dapi_ch = arr[..., 2]

    # Normalize Tubulin channel for intensity weighting
    # (Avoid division by zero later)
    t_min, t_max = tubulin_ch.min(), tubulin_ch.max()
    if t_max > t_min:
        tubulin_norm = (tubulin_ch - t_min) / (t_max - t_min)
    else:
        tubulin_norm = np.zeros_like(tubulin_ch)

    # --- Segmentation Logic ---
    # We need two masks: 
    # 1. Nuclei Mask (to find geometric center of nucleus)
    # 2. Cell Mask (to define the boundary of the cytoplasm for tubulin intensity)

    nuclei_mask = None
    cells_mask = None

    if len(segmentation_masks) >= 2:
        # Assuming standard order often found in these datasets: Cell mask, then Nuclei mask
        # However, we should check sizes to be sure. Nuclei are usually smaller.
        mask1 = segmentation_masks[0]
        mask2 = segmentation_masks[1]
        
        # Simple heuristic: The mask with more total area is likely the cell mask (cytoplasm + nucleus)
        # Or check if one is contained in the other.
        # Let's assume the provided masks are labeled.
        if np.sum(mask1 > 0) > np.sum(mask2 > 0):
            cells_mask = mask1
            nuclei_mask = mask2
        else:
            cells_mask = mask2
            nuclei_mask = mask1
            
    elif len(segmentation_masks) == 1:
        # If only one mask, assume it's the cell mask if it covers a large area, 
        # or nuclei if small. But for this specific feature, we need distinct compartments.
        # Fallback: Treat the provided mask as nuclei, and watershed for cells.
        nuclei_mask = segmentation_masks[0]
        
        # Generate cell mask via watershed
        # Use nuclei as markers
        markers = nuclei_mask.astype(int)
        # Gradient of image or inverted intensity as elevation map
        # Using inverted tubulin as "basins" where cells are bright
        elevation_map = -tubulin_ch
        cells_mask = watershed(elevation_map, markers, mask=tubulin_ch > threshold_otsu(tubulin_ch))

    else:
        # No masks provided: Full fallback pipeline
        
        # 1. Detect Nuclei (DAPI)
        try:
            thresh_val = threshold_otsu(dapi_ch)
        except ValueError: # Handle empty image
            thresh_val = 0
            
        binary_nuc = dapi_ch > thresh_val
        binary_nuc = binary_opening(binary_nuc, disk(2))
        nuclei_mask = label(binary_nuc)
        
        # 2. Detect Cells (Tubulin)
        # Use watershed from nuclei seeds
        try:
            tub_thresh = threshold_otsu(tubulin_ch)
        except ValueError:
            tub_thresh = 0
            
        mask_bg = tubulin_ch > tub_thresh
        cells_mask = watershed(-tubulin_ch, nuclei_mask, mask=mask_bg)

    # Ensure masks are integer labels
    nuclei_mask = nuclei_mask.astype(int)
    cells_mask = cells_mask.astype(int)

    # Remove border cells to avoid bias in center of mass calculations due to cropping
    cells_mask = clear_border(cells_mask)
    
    # If clearing border removed a cell, we must ignore its nucleus too.
    # We can do this by iterating only through cell labels present in the cleared mask.
    
    unique_cell_ids = np.unique(cells_mask)
    unique_cell_ids = unique_cell_ids[unique_cell_ids != 0] # Remove background

    if len(unique_cell_ids) == 0:
        return 0.0

    offsets = []

    # --- Feature Calculation ---
    # We need to match nuclei to cells.
    # Strategy: For each valid cell, find the nucleus inside it.
    
    # Pre-calculate centers to optimize
    # 1. Geometric centers of nuclei
    # We can use regionprops or center_of_mass. 
    # Since we need to match specific nuclei to specific cells, iterating might be safer 
    # to handle cases where IDs don't match 1:1 perfectly.
    
    for cell_id in unique_cell_ids:
        # Create mask for current cell
        current_cell_mask = (cells_mask == cell_id)
        
        # Find the nucleus ID(s) within this cell
        # We look at the nuclei_mask where the current_cell_mask is True
        nuclei_in_cell = nuclei_mask[current_cell_mask]
        nuclei_in_cell = nuclei_in_cell[nuclei_in_cell != 0] # Exclude background
        
        if nuclei_in_cell.size == 0:
            continue # No nucleus found in this cell
            
        # Get the most frequent nucleus ID in this cell region (dominance)
        # (Handles rare cases of fragmentation)
        nuc_id = np.bincount(nuclei_in_cell).argmax()
        
        # 1. Calculate Nucleus Geometric Centroid
        # We use the nucleus mask for this
        nuc_slice = ndimage.find_objects(nuclei_mask == nuc_id)[0]
        if nuc_slice is None: 
            continue
            
        # Calculate center of mass of the binary nucleus mask (Geometric Center)
        # We compute it relative to the slice to save time, then add offset
        nuc_local_mask = (nuclei_mask[nuc_slice] == nuc_id)
        cy_n, cx_n = ndimage.center_of_mass(nuc_local_mask)
        cy_n += nuc_slice[0].start
        cx_n += nuc_slice[1].start
        
        # 2. Calculate Tubulin Intensity-Weighted Centroid
        # We use the cell mask for the region, and tubulin intensity as weights
        cell_slice = ndimage.find_objects(cells_mask == cell_id)[0]
        cell_local_mask = (cells_mask[cell_slice] == cell_id)
        cell_local_intensity = tubulin_norm[cell_slice]
        
        # If the cell is purely black (no intensity), center of mass is undefined/center of box
        if np.sum(cell_local_intensity[cell_local_mask]) == 0:
            continue

        # center_of_mass(input, labels, index)
        # input is the weights (intensity), labels is the mask
        cy_t, cx_t = ndimage.center_of_mass(cell_local_intensity, labels=cell_local_mask, index=1)
        
        # Adjust for slice offset
        cy_t += cell_slice[0].start
        cx_t += cell_slice[1].start
        
        # 3. Calculate Euclidean Distance
        dist = np.sqrt((cy_n - cy_t)**2 + (cx_n - cx_t)**2)
        offsets.append(dist)

    if not offsets:
        return 0.0

    # Return the mean offset across all valid cells in the image
    result = np.mean(offsets)
    
    return float(result)
