def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import dilation, disk

    # --- 1. Data Loading and Preprocessing ---
    # Convert to float32 for calculations
    arr = np.asarray(img, dtype=np.float32)

    # Check dimensionality and extract channels
    # Expected shape: (512, 512, 3)
    # Channel 0: Actin (Red) - Cytoskeleton
    # Channel 1: Tubulin (Green) - Target Signal
    # Channel 2: DAPI (Blue) - Nucleus (Center reference)
    
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    tubulin_ch = arr[..., 1]
    dapi_ch = arr[..., 2]
    
    # Normalize Tubulin channel for intensity weighting
    # Robust min-max normalization
    p_min, p_max = np.percentile(tubulin_ch, (1, 99))
    if p_max > p_min:
        tubulin_norm = (tubulin_ch - p_min) / (p_max - p_min)
    else:
        tubulin_norm = tubulin_ch # Fallback if flat
    tubulin_norm = np.clip(tubulin_norm, 0.0, 1.0)

    # --- 2. Segmentation Handling ---
    # We need a cell mask (defining the boundary) and a nucleus mask (defining the center)
    # If provided, use them. If not, generate them.
    
    cell_labels = None
    nuclei_labels = None

    if len(segmentation_masks) >= 2:
        # Assuming standard order often seen: [cell_mask, nucleus_mask] or similar
        # We need to be careful. Usually, if 2 masks are provided, one is likely nuclei and one is cells.
        # Heuristic: Nuclei are usually smaller and contained within cells.
        mask1 = segmentation_masks[0]
        mask2 = segmentation_masks[1]
        
        # Simple heuristic: The mask with more total area is likely the cell mask (or cytoplasm)
        # However, let's assume the caller provides meaningful masks. 
        # If we can't distinguish, we'll treat the first as cell and rely on DAPI channel for centers.
        # A safer fallback is to regenerate nuclei labels from the DAPI channel to ensure centers are accurate
        # relative to the image signal, and use the provided mask for cell boundaries.
        
        if np.sum(mask1 > 0) > np.sum(mask2 > 0):
            cell_labels = mask1
            # We will re-derive nuclei centers from DAPI to be safe, or use mask2 if it aligns well.
            # Let's use mask2 as nuclei labels.
            nuclei_labels = mask2
        else:
            cell_labels = mask2
            nuclei_labels = mask1
            
    elif len(segmentation_masks) == 1:
        # Only one mask provided. Assume it's the cell mask.
        cell_labels = segmentation_masks[0]
        # We must generate nuclei labels
        try:
            thresh = threshold_otsu(dapi_ch)
            nuclei_mask = dapi_ch > thresh
            nuclei_labels = label(nuclei_mask)
        except:
            return 0.0
            
    else:
        # No masks provided. Generate both.
        try:
            # Nuclei
            thresh_nuc = threshold_otsu(dapi_ch)
            nuclei_mask = dapi_ch > thresh_nuc
            nuclei_labels = label(nuclei_mask)
            
            # Cells (Coarse approximation using Tubulin + Actin)
            # We need a region to measure the "periphery"
            combined_signal = arr[..., 0] + arr[..., 1] # Actin + Tubulin
            thresh_cell = threshold_otsu(combined_signal)
            cell_mask = combined_signal > thresh_cell
            # Clean up cell mask
            cell_mask = dilation(cell_mask, disk(3))
            
            # Separate cells using watershed is ideal, but simple labeling is safer for a robust fallback
            # without complex imports. We will just label the blobs.
            # Note: This might merge touching cells, but it's better than nothing.
            cell_labels = label(cell_mask)
        except:
            return 0.0

    # Ensure labels are integers
    cell_labels = cell_labels.astype(int)
    nuclei_labels = nuclei_labels.astype(int)

    # --- 3. Feature Computation: Radial Distribution Peak ---
    # Logic:
    # For each cell:
    # 1. Find the centroid of the nucleus (Center).
    # 2. Calculate distance of every pixel in the cell from this center.
    # 3. Normalize distances by the max distance in that cell (0=center, 1=edge).
    # 4. Compute the intensity-weighted average distance.
    #    - If Tubulin is collapsed (peak at center), weighted avg distance is LOW.
    #    - If Tubulin is uniform, weighted avg distance is MEDIUM/HIGH.
    # 5. Feature "Peak" = 1.0 - (Weighted Avg Distance).
    #    - High value -> Mass is concentrated at center.
    
    cell_props = regionprops(cell_labels)
    nuclei_props = regionprops(nuclei_labels)
    
    # Map nuclei to cells to find the center of each cell
    # We create a map of {cell_label: (center_r, center_c)}
    cell_centers = {}
    
    # Strategy: Iterate over nuclei, find which cell label is at the nucleus centroid
    for n_prop in nuclei_props:
        yc, xc = n_prop.centroid
        yc, xc = int(yc), int(xc)
        
        # Check bounds
        if 0 <= yc < cell_labels.shape[0] and 0 <= xc < cell_labels.shape[1]:
            c_lbl = cell_labels[yc, xc]
            if c_lbl > 0:
                # If multiple nuclei in one cell, average them or take first. Taking first is simple.
                if c_lbl not in cell_centers:
                    cell_centers[c_lbl] = (yc, xc)

    peak_scores = []

    for c_prop in cell_props:
        c_lbl = c_prop.label
        
        # Skip if we didn't find a nucleus for this cell
        if c_lbl not in cell_centers:
            continue
            
        # Get bounding box to reduce computation
        minr, minc, maxr, maxc = c_prop.bbox
        
        # Extract cell mask and intensity within bbox
        local_mask = (cell_labels[minr:maxr, minc:maxc] == c_lbl)
        local_intensity = tubulin_norm[minr:maxr, minc:maxc]
        
        if np.sum(local_mask) < 50: # Skip tiny fragments
            continue
            
        # Get center coordinates relative to bbox
        global_cy, global_cx = cell_centers[c_lbl]
        local_cy = global_cy - minr
        local_cx = global_cx - minc
        
        # Create coordinate grids
        yy, xx = np.indices(local_mask.shape)
        
        # Calculate Euclidean distance from nucleus center
        # We only care about pixels inside the cell mask
        valid_pixels = local_mask
        
        if not np.any(valid_pixels):
            continue
            
        # Distances for valid pixels
        dy = yy[valid_pixels] - local_cy
        dx = xx[valid_pixels] - local_cx
        dists = np.sqrt(dy**2 + dx**2)
        
        # Intensities for valid pixels
        ints = local_intensity[valid_pixels]
        
        # Normalize distances (0 to 1)
        # Max distance is the furthest point in the cell from the nucleus
        max_dist = np.max(dists)
        if max_dist == 0:
            continue
            
        norm_dists = dists / max_dist
        
        # Calculate Intensity-Weighted Mean Distance (IWMD)
        # Sum(Intensity * Distance) / Sum(Intensity)
        total_intensity = np.sum(ints)
        
        if total_intensity == 0:
            iwmd = 0.5 # Neutral if no signal
        else:
            iwmd = np.sum(ints * norm_dists) / total_intensity
            
        # The feature is "Peak". 
        # IWMD is low (near 0) if mass is at center.
        # IWMD is high (near 0.5-0.6) if mass is distributed.
        # We invert it so High Value = High Peak (Collapsed).
        # Theoretical max IWMD for a ring at edge is 1.0.
        # Theoretical min IWMD for a point at center is 0.0.
        # Uniform disk distribution IWMD is approx 2/3 (0.66) for linear weighting in 2D area?
        # Actually for a uniform disk, integral(r * r dr dtheta) / integral(r dr dtheta) -> r^3/3 / r^2/2 = 2/3 R.
        # So uniform is ~0.66. Collapsed is < 0.3.
        
        peak_score = 1.0 - iwmd
        peak_scores.append(peak_score)

    # --- 4. Aggregation ---
    if not peak_scores:
        return 0.0
        
    # Return the median score across the population of cells
    result = np.median(peak_scores)
    
    return float(result)
