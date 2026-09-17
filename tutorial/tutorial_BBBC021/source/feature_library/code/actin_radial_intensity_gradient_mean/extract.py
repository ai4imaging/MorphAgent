def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    
    # --- 1. Data Loading and Preprocessing ---
    
    # Convert to float32 for processing
    arr = np.asarray(img, dtype=np.float32)
    
    # Check dimensionality and extract channels
    # Expected: (H, W, 3) where Ch0=Actin, Ch2=DAPI
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0
        
    actin_img = arr[..., 0]  # Channel 0: Actin (Cytoskeleton)
    dapi_img = arr[..., 2]   # Channel 2: DAPI (Nucleus)
    
    # Normalize actin image to [0, 1] globally to handle uint8 range
    # (We will re-normalize per cell later for the gradient calculation)
    actin_img = actin_img / 255.0

    # --- 2. Segmentation Handling ---
    
    # We need a cell mask and a nucleus mask.
    # Strategy:
    # 1. If segmentation_masks are provided, try to use them.
    #    - Usually, if 2 masks are provided, one is nuclei, one is cells.
    #    - If 1 mask is provided, it might be cells.
    # 2. Fallback: Generate masks from channels if not provided.
    
    cell_mask = None
    nuc_mask = None
    
    if len(segmentation_masks) >= 2:
        # Heuristic: usually smaller mask is nuclei, larger is cells, or ordered.
        # Without specific metadata, we assume mask 0 is cells and mask 1 is nuclei or vice versa.
        # Let's assume the one with more foreground pixels is the cell mask.
        m1 = segmentation_masks[0]
        m2 = segmentation_masks[1]
        if np.sum(m1 > 0) > np.sum(m2 > 0):
            cell_mask = m1
            nuc_mask = m2
        else:
            cell_mask = m2
            nuc_mask = m1
    elif len(segmentation_masks) == 1:
        cell_mask = segmentation_masks[0]
        # Generate nucleus mask from DAPI
        try:
            thresh = threshold_otsu(dapi_img)
            nuc_mask = (dapi_img > thresh).astype(int)
            nuc_mask = label(nuc_mask)
        except:
            nuc_mask = np.zeros_like(dapi_img, dtype=int)
    else:
        # No masks provided, generate both
        try:
            # Nuclei from DAPI
            t_nuc = threshold_otsu(dapi_img)
            nuc_mask = label(dapi_img > t_nuc)
            
            # Cells from Actin (rough approximation)
            # Actin often covers the whole cell body
            t_cell = threshold_otsu(actin_img) if np.max(actin_img) > 0 else 0.1
            # Combine DAPI and Actin for a better cell body estimate
            combined = np.maximum(actin_img, dapi_img/np.max(dapi_img) if np.max(dapi_img)>0 else 0)
            t_comb = threshold_otsu(combined) if np.max(combined) > 0 else 0.1
            cell_mask = label(combined > t_comb)
        except:
            return 0.0

    # Ensure masks are labeled integers
    if cell_mask is None or nuc_mask is None:
        return 0.0
        
    cell_labels = cell_mask.astype(int)
    nuc_labels = nuc_mask.astype(int)
    
    # --- 3. Feature Computation: Radial Intensity Gradient ---
    
    # We want to measure the gradient of actin intensity from nucleus to cell boundary.
    # We use a normalized distance map approach.
    # Normalized Radius (R) = Dist(pixel, nucleus) / (Dist(pixel, nucleus) + Dist(pixel, background))
    # R goes from 0 (at nucleus boundary) to 1 (at cell boundary).
    
    # Create binary masks for distance transform
    # We treat all nuclei as "sources" and all background as "sinks"
    
    # Binary total foreground
    total_foreground = (cell_labels > 0)
    # Binary total nuclei
    total_nuclei = (nuc_labels > 0)
    
    # If no cells or nuclei, return 0
    if not np.any(total_foreground) or not np.any(total_nuclei):
        return 0.0

    # Distance from nearest nucleus (0 inside nucleus, increasing outwards)
    # We invert total_nuclei because edt calculates distance to nearest 0
    dist_from_nuc = ndimage.distance_transform_edt(~total_nuclei)
    
    # Distance from nearest background (0 outside cell, increasing inwards)
    # We invert total_foreground
    dist_from_bg = ndimage.distance_transform_edt(total_foreground)
    
    # To avoid division by zero, we add a small epsilon
    denominator = dist_from_nuc + dist_from_bg
    denominator[denominator == 0] = 1.0 # Should only happen in background
    
    # Normalized Radius Map
    # Pixels inside nucleus will have dist_from_nuc=0 -> R=0
    # Pixels at cell boundary will have dist_from_bg=0 -> R=1
    normalized_radius_map = dist_from_nuc / denominator
    
    props = regionprops(cell_labels)
    slopes = []
    
    for prop in props:
        # Get the bounding box for efficiency
        minr, minc, maxr, maxc = prop.bbox
        
        # Extract local masks and maps
        local_cell_mask = (cell_labels[minr:maxr, minc:maxc] == prop.label)
        
        # Identify the nucleus corresponding to this cell
        # We look for the nucleus label that overlaps most with this cell or is contained within
        local_nuc_labels = nuc_labels[minr:maxr, minc:maxc]
        # Mask out background in nuc labels
        local_nuc_labels_masked = local_nuc_labels[local_cell_mask]
        local_nuc_labels_masked = local_nuc_labels_masked[local_nuc_labels_masked > 0]
        
        if local_nuc_labels_masked.size == 0:
            continue # No nucleus found in this cell
            
        # Use the most frequent nucleus label in this cell region
        # (Simple way to handle 1-to-1 mapping issues)
        # In a perfect segmentation, there's 1 nucleus per cell.
        
        # Define Cytoplasm Mask: Inside cell BUT Outside Nucleus
        # We use the global binary nucleus mask for exclusion to be safe
        local_is_nucleus = total_nuclei[minr:maxr, minc:maxc]
        local_cyto_mask = local_cell_mask & (~local_is_nucleus)
        
        if np.sum(local_cyto_mask) < 10:
            continue # Too small to compute gradient
            
        # Extract Intensity and Radius values for the cytoplasm
        local_actin = actin_img[minr:maxr, minc:maxc][local_cyto_mask]
        local_radius = normalized_radius_map[minr:maxr, minc:maxc][local_cyto_mask]
        
        # Normalize Intensity per cell to [0, 1]
        # This is crucial so that bright cells don't dominate the slope magnitude
        # and we measure the *relative* distribution.
        i_min = np.min(local_actin)
        i_max = np.max(local_actin)
        
        if i_max - i_min < 1e-6:
            slope = 0.0 # Flat intensity
        else:
            local_actin_norm = (local_actin - i_min) / (i_max - i_min)
            
            # Linear Regression: Intensity = slope * Radius + intercept
            # We want the slope.
            # x = local_radius, y = local_actin_norm
            # np.polyfit returns [slope, intercept] for deg=1
            try:
                m, c = np.polyfit(local_radius, local_actin_norm, 1)
                slope = m
            except:
                slope = 0.0
        
        slopes.append(slope)
        
    # --- 4. Aggregation ---
    
    if not slopes:
        return 0.0
        
    # Return the mean slope across all valid cells
    result = np.mean(slopes)
    
    return float(result)
