def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import stats
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.segmentation import watershed

    # Convert to appropriate array type
    # Image shape is (512, 512, 3), dtype uint8
    arr = np.asarray(img, dtype=np.float32)
    
    # Check dimensionality
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract Channels
    # Channel 1: Tubulin (Green) - Target for intensity measurement
    # Channel 2: DAPI (Blue) - Target for nucleus identification
    tubulin = arr[:, :, 1]
    dapi = arr[:, :, 2]

    # Normalize Tubulin channel globally for safety (though we do per-cell norm later)
    # This just ensures we are working in a reasonable float range
    tubulin = tubulin / 255.0
    dapi = dapi / 255.0

    # --- Segmentation Logic ---
    nuclei_mask = None
    cell_mask = None

    # Helper to check if a mask is valid
    def is_valid_mask(m):
        return m is not None and m.size > 0 and m.max() > 0

    # Strategy: Determine which mask is which based on size, or generate if missing
    if len(segmentation_masks) >= 2:
        # If 2 masks are provided, we assume one is Nuclei and one is Cells.
        # Heuristic: Nuclei are generally smaller than Cells.
        m1 = segmentation_masks[0]
        m2 = segmentation_masks[1]
        
        # Ensure labeled
        if m1.max() == 1: m1 = label(m1)
        if m2.max() == 1: m2 = label(m2)
        
        # Calculate mean area to distinguish
        props1 = regionprops(m1)
        props2 = regionprops(m2)
        area1 = np.mean([p.area for p in props1]) if props1 else 0
        area2 = np.mean([p.area for p in props2]) if props2 else 0
        
        if area1 < area2:
            nuclei_mask, cell_mask = m1, m2
        else:
            nuclei_mask, cell_mask = m2, m1
            
    elif len(segmentation_masks) == 1:
        # If 1 mask, assume it's the Cell mask (or whatever object we are measuring)
        cell_mask = segmentation_masks[0]
        if cell_mask.max() == 1: cell_mask = label(cell_mask)
        # We will use cell centroids as the "center" since we lack a specific nuclei mask
        
    else:
        # Fallback: Generate masks on the fly
        try:
            # 1. Segment Nuclei (DAPI)
            thresh_dapi = threshold_otsu(dapi)
            nuclei_mask = label(dapi > thresh_dapi)
            
            # 2. Segment Cells (Tubulin)
            # Use watershed seeded by nuclei to partition the tubulin signal
            thresh_tub = threshold_otsu(tubulin)
            binary_tub = tubulin > thresh_tub
            # Combine signals to ensure cell covers nucleus
            mask_combined = np.logical_or(binary_tub, dapi > thresh_dapi)
            
            if nuclei_mask.max() > 0:
                cell_mask = watershed(~tubulin, nuclei_mask, mask=mask_combined)
            else:
                cell_mask = label(mask_combined)
        except Exception:
            # If Otsu fails (e.g. empty image), return 0
            return 0.0

    if not is_valid_mask(cell_mask):
        return 0.0

    # --- Feature Computation: Radial Intensity Gradient ---
    
    # 1. Map Nuclei to Cells (if nuclei mask exists) to find the "biological center"
    cell_centers = {}
    
    # Get cell properties
    props_cells = regionprops(cell_mask)
    
    # If we have a nuclei mask, map each cell label to its nucleus centroid
    if is_valid_mask(nuclei_mask):
        props_nuclei = regionprops(nuclei_mask)
        for p in props_nuclei:
            # Get centroid of nucleus
            yc, xc = p.centroid
            yi, xi = int(yc), int(xc)
            
            # Check which cell this nucleus falls into
            if 0 <= yi < cell_mask.shape[0] and 0 <= xi < cell_mask.shape[1]:
                cell_lbl = cell_mask[yi, xi]
                if cell_lbl > 0:
                    # Store the nucleus centroid as the center for this cell
                    cell_centers[cell_lbl] = (yc, xc)

    # 2. Iterate over cells and compute gradient
    slopes = []
    
    for cell in props_cells:
        # Filter small artifacts
        if cell.area < 50:
            continue
            
        # Determine the center point for radial calculation
        if cell.label in cell_centers:
            center = cell_centers[cell.label]
        else:
            # Fallback to geometric centroid of the cell itself
            center = cell.centroid
            
        # Get coordinates of all pixels in the cell
        # coords is (N, 2) array of (row, col)
        coords = cell.coords
        
        # Extract Intensity values for these pixels
        pixel_intensities = tubulin[coords[:, 0], coords[:, 1]]
        
        # Calculate Euclidean distance from center for each pixel
        # center is (row, col)
        dy = coords[:, 0] - center[0]
        dx = coords[:, 1] - center[1]
        distances = np.sqrt(dy**2 + dx**2)
        
        # --- Normalization ---
        # We normalize both distance and intensity to [0, 1] to make the slope 
        # comparable across cells of different sizes and brightness levels.
        
        # Normalize Distance
        max_dist = np.max(distances)
        if max_dist == 0: continue # Single pixel cell
        norm_dist = distances / max_dist
        
        # Normalize Intensity (Min-Max scaling per cell)
        # This focuses on the *pattern* of distribution rather than absolute amount
        min_int = np.min(pixel_intensities)
        max_int = np.max(pixel_intensities)
        
        if max_int - min_int < 1e-6:
            # Uniform intensity -> slope is 0
            slope = 0.0
        else:
            norm_int = (pixel_intensities - min_int) / (max_int - min_int)
            
            # Linear Regression: Intensity = slope * Distance + intercept
            # x = normalized distance (0=center, 1=edge)
            # y = normalized intensity
            res = stats.linregress(norm_dist, norm_int)
            slope = res.slope
            
            # Handle NaN results from regression
            if np.isnan(slope):
                slope = 0.0
                
        slopes.append(slope)

    # 3. Aggregate
    if not slopes:
        return 0.0
        
    # Return the mean slope across all cells
    # Negative slope: Intensity drops as we go out (Perinuclear accumulation)
    # Positive slope: Intensity increases as we go out (Cortical accumulation)
    result = np.mean(slopes)

    return float(result)
