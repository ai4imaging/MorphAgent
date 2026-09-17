def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu, threshold_local
    from skimage.morphology import binary_opening, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If 2D (H, W), assume it's a single channel or projection, but dataset says 3 channels.
        # If it's (H, W), we can't reliably distinguish DAPI. Return 0.0.
        return 0.0

    # Extract Channels
    # Channel 0: Actin (R), Channel 1: Tubulin (G), Channel 2: DAPI (B)
    # We need DAPI for fragments, and potentially Actin/Tubulin for cell boundaries if masks are missing.
    ch_actin = arr[..., 0]
    ch_tubulin = arr[..., 1]
    ch_dapi = arr[..., 2]

    # Intensity normalization (0-1 range)
    # DAPI
    vmax_dapi = np.percentile(ch_dapi, 99.5) if ch_dapi.size > 0 else 1.0
    if vmax_dapi > 0:
        norm_dapi = ch_dapi / vmax_dapi
    else:
        norm_dapi = ch_dapi
    norm_dapi = np.clip(norm_dapi, 0.0, 1.0)

    # -------------------------------------------------------------------------
    # Step 1: Define Cell Objects (ROI)
    # -------------------------------------------------------------------------
    cell_labels = None
    
    # Check if valid segmentation masks are provided
    # We look for a mask that likely represents the whole cell (cytoplasm)
    # If multiple masks are provided, we prioritize the one that covers more area (likely cytoplasm vs nucleus)
    if len(segmentation_masks) > 0:
        # Try to find a suitable mask
        best_mask = None
        max_area = -1
        
        for mask in segmentation_masks:
            if mask is None:
                continue
            # Ensure mask is 2D
            if mask.ndim == 3:
                mask = np.max(mask, axis=2) # MIP if 3D
            
            current_area = np.sum(mask > 0)
            if current_area > max_area:
                max_area = current_area
                best_mask = mask
        
        if best_mask is not None:
            cell_labels = best_mask.astype(int)

    # Fallback: Generate cell approximation from Actin + Tubulin if no mask provided
    if cell_labels is None:
        # Combine structural channels
        structure = (ch_actin + ch_tubulin) / 2.0
        # Normalize
        vmax_struct = np.percentile(structure, 99.5) if structure.size > 0 else 1.0
        if vmax_struct > 0:
            structure = structure / vmax_struct
        structure = np.clip(structure, 0.0, 1.0)
        
        # Threshold
        try:
            thresh = threshold_otsu(structure)
            binary_cells = structure > thresh
            # Clean up
            binary_cells = binary_opening(binary_cells, disk(3))
            cell_labels = label(binary_cells)
        except Exception:
            # If thresholding fails (e.g. empty image), return 0
            return 0.0

    # -------------------------------------------------------------------------
    # Step 2: Detect Nuclear Fragments (DAPI Objects)
    # -------------------------------------------------------------------------
    # Use local thresholding to catch small, dim micronuclei as well as main nuclei
    try:
        # Block size must be odd
        local_thresh = threshold_local(norm_dapi, block_size=51, offset=0.05)
        binary_dapi = norm_dapi > local_thresh
        
        # Remove very small noise (e.g., < 5 pixels)
        binary_dapi = binary_opening(binary_dapi, disk(1))
        
        # Label DAPI objects
        dapi_labels = label(binary_dapi)
        dapi_props = regionprops(dapi_labels)
    except Exception:
        return 0.0

    # -------------------------------------------------------------------------
    # Step 3: Compute Fragmentation Index per Cell
    # -------------------------------------------------------------------------
    # Feature: Count of distinct DAPI objects within a defined radius of cell centroid, normalized by cell area.
    
    cell_props = regionprops(cell_labels)
    
    fragmentation_indices = []
    
    # Radius to search around centroid (in pixels)
    # MCF-7 cells are roughly 20-40um. At typical magnifications (e.g., 20x, 0.5um/px), 
    # a radius of ~60-80 pixels covers a reasonable cellular region.
    search_radius = 60.0 
    
    # Create a coordinate grid for distance calculations if needed, 
    # but checking bounding boxes is faster.
    
    for cell in cell_props:
        cell_area = cell.area
        if cell_area < 100: # Ignore tiny debris labeled as cells
            continue
            
        cy, cx = cell.centroid
        
        # Count DAPI objects associated with this cell
        # Criteria: DAPI object centroid is within the cell mask AND within radius
        dapi_count = 0
        
        # Optimization: Only check DAPI objects whose centroids are within the cell's bounding box
        minr, minc, maxr, maxc = cell.bbox
        
        # We iterate through DAPI objects. 
        # Note: In a highly optimized pipeline, we would mask the DAPI labels with the cell mask.
        # Here, we iterate DAPI props to check centroids against the specific cell criteria.
        # To be efficient, we can extract the DAPI label crop corresponding to the cell bbox.
        
        # Extract DAPI labels within the cell bounding box
        dapi_crop = dapi_labels[minr:maxr, minc:maxc]
        
        # Get unique DAPI labels in this region
        unique_dapi_in_bbox = np.unique(dapi_crop)
        unique_dapi_in_bbox = unique_dapi_in_bbox[unique_dapi_in_bbox != 0] # Remove background
        
        for d_id in unique_dapi_in_bbox:
            # Get properties of this specific DAPI object (using the global list is slow, better to index)
            # Since we have the ID, we can find it in dapi_props. 
            # Note: regionprops list is not indexed by label ID directly (it's 0-indexed list).
            # dapi_props[i].label returns the label.
            # Faster approach: Check geometric center of the label in the crop relative to cell centroid.
            
            # Mask for this specific DAPI object within the crop
            d_mask = (dapi_crop == d_id)
            if not np.any(d_mask):
                continue
                
            # Calculate centroid of this DAPI object in global coordinates
            # Center of mass in crop
            d_cy_local, d_cx_local = ndimage.center_of_mass(d_mask)
            d_cy = d_cy_local + minr
            d_cx = d_cx_local + minc
            
            # Check 1: Is the DAPI centroid inside the specific cell's binary mask?
            # cell.image is the binary mask of the cell in the bbox
            # Coordinates in cell.image are (d_cy - minr, d_cx - minc)
            local_y, local_x = int(d_cy - minr), int(d_cx - minc)
            
            # Boundary check for indexing
            if 0 <= local_y < cell.image.shape[0] and 0 <= local_x < cell.image.shape[1]:
                if cell.image[local_y, local_x]:
                    # Check 2: Is it within the defined radius of the cell centroid?
                    dist = np.sqrt((d_cy - cy)**2 + (d_cx - cx)**2)
                    if dist <= search_radius:
                        dapi_count += 1

        # Calculate Index
        # Normalizing by cell area makes it a density metric.
        # If a cell has 1 nucleus and area 1000, index = 0.001
        # If a cell has 10 fragments and area 1000, index = 0.010
        if cell_area > 0:
            index = dapi_count / cell_area
            fragmentation_indices.append(index)

    # -------------------------------------------------------------------------
    # Step 4: Aggregate and Return
    # -------------------------------------------------------------------------
    if not fragmentation_indices:
        return 0.0
        
    # Return the mean fragmentation index across the population
    # Multiply by a scaling factor (e.g., 1000) to make the number more readable/significant 
    # as a feature value (since counts/area is usually very small).
    result = np.mean(fragmentation_indices) * 1000.0
    
    return float(result)
