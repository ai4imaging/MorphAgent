def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import regionprops, label
    from skimage.filters import threshold_otsu
    from skimage.segmentation import watershed, clear_border
    from skimage.morphology import binary_closing, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3) -> (Height, Width, Channels)
    # Channel 0: Actin (Red) - Target for intensity distribution
    # Channel 1: Tubulin (Green) - Helper for cell body
    # Channel 2: DAPI (Blue) - Reference for nucleus centroid
    
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract channels
    actin = arr[..., 0]
    tubulin = arr[..., 1]
    dapi = arr[..., 2]

    # Normalize intensities to [0, 1] for calculation
    def normalize(ch):
        vmax = np.percentile(ch, 99.5) if ch.size > 0 else 1.0
        if vmax <= 0: return ch
        return np.clip(ch / vmax, 0.0, 1.0)

    actin_norm = normalize(actin)
    dapi_norm = normalize(dapi)
    tubulin_norm = normalize(tubulin)

    # --- Segmentation Logic ---
    # Goal: Obtain a labeled mask where each integer represents a distinct cell.
    # We need both cell boundaries and nuclear centroids.
    
    cell_labels = None
    nuclei_labels = None

    # Check if valid segmentation masks are provided
    # The system might pass masks. We look for a mask that resembles cell bodies or nuclei.
    # If multiple masks, we try to identify which is which, or default to the first one as cells.
    valid_masks = [m for m in segmentation_masks if m is not None and m.shape == actin.shape]
    
    if len(valid_masks) > 0:
        # Heuristic: If we have masks, assume the one with larger average area per object is the cell mask,
        # and the smaller one is nuclei. If only one, treat it as cell mask if it covers significant area,
        # or nuclei if very sparse. For simplicity in this robust implementation:
        # If 2 masks: assume [0] is cell, [1] is nuclei (common convention) or check overlap.
        # If 1 mask: assume it's the cell mask (or nuclei mask that we expand).
        
        # Let's try to use the first mask as the primary object definition
        cell_labels = valid_masks[0].astype(int)
        
        # If we have a second mask, use it for nuclei
        if len(valid_masks) > 1:
            nuclei_labels = valid_masks[1].astype(int)
    
    # Fallback: Automatic Segmentation if no masks provided or they are empty
    if cell_labels is None or np.max(cell_labels) == 0:
        # 1. Detect Nuclei (Seeds)
        try:
            thresh_nuc = threshold_otsu(dapi_norm)
        except ValueError: # Handle empty images
            thresh_nuc = 0.1
            
        mask_nuc = dapi_norm > thresh_nuc
        mask_nuc = binary_closing(mask_nuc, disk(2))
        nuclei_labels = label(mask_nuc)

        # 2. Detect Cell Body (Mask)
        # Combine Actin and Tubulin for better cell body definition
        combined_cyto = (actin_norm + tubulin_norm) / 2.0
        try:
            thresh_cell = threshold_otsu(combined_cyto)
        except ValueError:
            thresh_cell = 0.1
            
        mask_cell = combined_cyto > thresh_cell
        mask_cell = binary_closing(mask_cell, disk(3))

        # 3. Watershed to separate cells
        # Use nuclei as markers, intensity as basin
        if np.max(nuclei_labels) > 0:
            # Invert intensity for watershed (basins are dark/low values in inverted image)
            # We want "high intensity" to be the basin bottom.
            basin = -combined_cyto
            cell_labels = watershed(basin, nuclei_labels, mask=mask_cell)
        else:
            cell_labels = label(mask_cell) # Fallback if no nuclei found

    # If we still don't have nuclei labels but have cell labels (e.g. from provided mask),
    # we need to estimate nucleus position.
    # We can re-derive nuclei from DAPI inside the cell labels.
    if nuclei_labels is None:
        try:
            thresh_nuc = threshold_otsu(dapi_norm)
        except:
            thresh_nuc = 0.1
        nuclei_labels = label(dapi_norm > thresh_nuc)

    # Clean border cells to avoid edge artifacts in radial distribution
    cell_labels = clear_border(cell_labels)

    # --- Feature Computation: Actin Radial Distribution ---
    
    props_cells = regionprops(cell_labels)
    
    # We need to map cell labels to their corresponding nucleus centroid.
    # Since nuclei_labels might not perfectly match cell_labels 1-to-1 in all cases (e.g. segmentation errors),
    # we compute the centroid of the DAPI signal *within* each cell label region.
    
    radial_scores = []

    # Create coordinate grids once
    h, w = actin.shape
    y_indices, x_indices = np.indices((h, w))

    for prop in props_cells:
        # Get the bounding box to slice the arrays (optimization)
        minr, minc, maxr, maxc = prop.bbox
        
        # Extract local masks and signals
        cell_mask_local = cell_labels[minr:maxr, minc:maxc] == prop.label
        
        # If the cell is too small, skip
        if np.sum(cell_mask_local) < 50:
            continue

        # 1. Determine the "Center" of the cell.
        # Ideally, this is the centroid of the nucleus.
        # We look at the DAPI signal inside this specific cell mask.
        dapi_local = dapi_norm[minr:maxr, minc:maxc]
        # Mask DAPI by the cell boundary to ensure we only look inside this cell
        dapi_masked = dapi_local * cell_mask_local
        
        # Calculate intensity-weighted centroid of DAPI within this cell
        # If DAPI signal is too weak, fall back to geometric centroid of the cell
        dapi_sum = np.sum(dapi_masked)
        if dapi_sum > 0:
            # ndimage.center_of_mass returns (y, x) relative to the slice
            cy_local, cx_local = ndimage.center_of_mass(dapi_masked)
        else:
            cy_local, cx_local = prop.centroid_local

        # 2. Calculate Radial Distances for all pixels in the cell
        # Coordinates relative to the slice
        y_local, x_local = np.indices((maxr-minr, maxc-minc))
        
        # Distance from the determined center
        distances = np.sqrt((y_local - cy_local)**2 + (x_local - cx_local)**2)
        
        # 3. Extract Actin Intensity
        actin_local = actin_norm[minr:maxr, minc:maxc]
        
        # Filter for pixels inside the cell mask
        valid_pixels = cell_mask_local
        
        if not np.any(valid_pixels):
            continue
            
        r_vals = distances[valid_pixels]
        i_vals = actin_local[valid_pixels]
        
        # 4. Compute Distribution Metric
        # Metric: Intensity-weighted mean radius normalized by max radius
        # This tells us: "On average, how far from the nucleus is the actin signal?"
        # Normalized 0.0 (center) to 1.0 (edge)
        
        total_intensity = np.sum(i_vals)
        max_r = np.max(r_vals)
        
        if total_intensity > 0 and max_r > 0:
            # Weighted mean radius
            weighted_mean_r = np.sum(r_vals * i_vals) / total_intensity
            
            # Normalize by the maximum radius of the cell (to be scale invariant)
            # This distinguishes "cortical" (near edge, high score) from "diffuse/perinuclear" (low score)
            normalized_score = weighted_mean_r / max_r
            radial_scores.append(normalized_score)

    # Aggregate results
    if not radial_scores:
        return 0.0
        
    result = np.mean(radial_scores)

    return float(result)
