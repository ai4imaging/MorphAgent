def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max

    # 1. Data Loading and Preprocessing
    # Ensure image is float32 for calculations
    arr = np.asarray(img, dtype=np.float32)

    # Check dimensionality
    # Expected: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract Channels
    # Channel 0: Actin (Red) - Cytoskeleton
    # Channel 1: Tubulin (Green) - Microtubules (Target for this feature)
    # Channel 2: DAPI (Blue) - Nucleus (Center reference)
    actin = arr[..., 0]
    tubulin = arr[..., 1]
    dapi = arr[..., 2]

    # Normalize Tubulin channel for intensity measurements [0, 1]
    # Using robust max to avoid hot pixels skewing normalization
    vmax = np.percentile(tubulin, 99.5) if tubulin.size > 0 else 1.0
    if vmax > 0:
        tubulin_norm = tubulin / vmax
    else:
        tubulin_norm = tubulin
    tubulin_norm = np.clip(tubulin_norm, 0.0, 1.0)

    # 2. Segmentation Strategy
    # We need both nuclei (for centers) and cell bodies (for boundaries)
    
    nuclei_mask = None
    cell_mask = None

    # Check if segmentation masks are provided
    # Standard expectation: mask[0] is usually nuclei, mask[1] is cells (if available)
    # However, we must be robust.
    if len(segmentation_masks) >= 1:
        # Assume first mask is nuclei or cells. 
        # If only one mask is provided, we might have to infer the other or use it for both.
        # Let's try to identify if we have distinct masks.
        
        # If we have at least 2 masks, we assume mask[0]=nuclei, mask[1]=cells (common convention)
        if len(segmentation_masks) >= 2:
            nuclei_mask = segmentation_masks[0]
            cell_mask = segmentation_masks[1]
        else:
            # Only one mask. If it covers the whole cell, we need to find nuclei inside.
            # If it covers only nuclei, we need to find cell boundaries.
            # Fallback: Use the provided mask as the primary object definition, 
            # but we still need the specific nucleus center.
            # Let's assume the provided mask is the cell mask for safety, and re-segment nuclei,
            # OR assume it's nuclei and estimate cell boundaries.
            # Given the feature relies on "radial from nucleus", accurate nucleus center is key.
            # Let's treat the provided mask as the 'objects' and refine from there.
            provided_mask = segmentation_masks[0]
            # We will use the provided mask as the cell mask and re-segment nuclei to be safe,
            # or use the provided mask as nuclei and expand for cells.
            # Heuristic: Nuclei are usually smaller. 
            # Let's stick to a robust internal segmentation if we aren't sure, 
            # but the prompt says "masks are automatically loaded".
            # Let's assume mask 0 is nuclei if it's the only one (common for simple datasets),
            # but actually, usually segmentation/ contains 'nuclei.tif' and 'cells.tif'.
            # Let's try to use the provided mask as nuclei and generate cell mask via watershed.
            nuclei_mask = provided_mask
            # Generate cell mask below...

    # Fallback / Refinement Segmentation
    if nuclei_mask is None:
        # Segment Nuclei from DAPI
        try:
            thresh_nuc = threshold_otsu(dapi)
        except ValueError: # Empty image
            thresh_nuc = 0
        binary_nuc = dapi > thresh_nuc
        # Remove small noise
        binary_nuc = ndimage.binary_opening(binary_nuc, structure=np.ones((3,3)))
        nuclei_mask = label(binary_nuc)

    if cell_mask is None:
        # Segment Cells using Watershed
        # Combine Actin and Tubulin for cell body signal
        cell_signal = actin + tubulin
        try:
            thresh_cell = threshold_otsu(cell_signal)
        except ValueError:
            thresh_cell = 0
        binary_cell = cell_signal > thresh_cell
        
        # Seeds from nuclei
        # Ensure nuclei_mask is int
        nuclei_mask = nuclei_mask.astype(int)
        
        # Watershed
        # We need a distance map or gradient. Simple approach:
        # Invert intensity for basins
        basins = -cell_signal
        # Run watershed
        cell_mask = watershed(basins, nuclei_mask, mask=binary_cell)

    # Ensure masks are consistent integers
    nuclei_mask = nuclei_mask.astype(np.int32)
    cell_mask = cell_mask.astype(np.int32)

    # 3. Feature Computation: Radial CV
    # Iterate over each cell
    
    # Get properties to find bounding boxes (optimization)
    cell_props = regionprops(cell_mask)
    
    # We need to map cell labels to nuclei centroids.
    # Since we used nuclei as seeds for watershed (or assumed correspondence),
    # label 'L' in cell_mask should correspond to label 'L' in nuclei_mask.
    # However, if masks came from different files, this might not hold.
    # Let's compute centroids of nuclei and match them to cell labels.
    
    nuclei_props = regionprops(nuclei_mask)
    nuclei_centers = {prop.label: prop.centroid for prop in nuclei_props}
    
    cv_values = []
    
    num_bins = 8  # Number of concentric rings
    
    for prop in cell_props:
        label_id = prop.label
        
        # Check if we have a corresponding nucleus
        if label_id not in nuclei_centers:
            # If labels don't match (e.g. different segmentation files), 
            # find the nucleus that overlaps most or is contained.
            # Simplified: skip if no direct label match, assuming watershed logic.
            # If masks were provided externally and mismatch, this might skip cells,
            # but matching overlapping masks is complex without more libraries.
            # Fallback: Calculate centroid of the cell part that overlaps with ANY nucleus?
            # Let's stick to the ID match assumption which is standard for matched mask sets.
            continue
            
        nuc_center = nuclei_centers[label_id] # (row, col)
        
        # Extract bounding box for the cell to reduce computation
        min_row, min_col, max_row, max_col = prop.bbox
        
        # Crop images and masks
        cell_roi_mask = cell_mask[min_row:max_row, min_col:max_col] == label_id
        tubulin_roi = tubulin_norm[min_row:max_row, min_col:max_col]
        
        # Adjust nucleus center to ROI coordinates
        center_r = nuc_center[0] - min_row
        center_c = nuc_center[1] - min_col
        
        # Create coordinate grid for ROI
        rows, cols = cell_roi_mask.shape
        rr, cc = np.indices((rows, cols))
        
        # Calculate distance from nucleus center
        dist_map = np.sqrt((rr - center_r)**2 + (cc - center_c)**2)
        
        # Mask the distance map with the cell shape
        # We only care about pixels inside the cell
        valid_pixels = cell_roi_mask
        
        if not np.any(valid_pixels):
            continue
            
        # Get distances and intensities for valid pixels
        pixel_dists = dist_map[valid_pixels]
        pixel_intensities = tubulin_roi[valid_pixels]
        
        if pixel_dists.size == 0:
            continue
            
        max_dist = np.max(pixel_dists)
        if max_dist == 0:
            continue
            
        # Define bins
        # We use relative bins (0-100% of radius) to be scale invariant across different cell sizes
        bins = np.linspace(0, max_dist, num_bins + 1)
        
        # Digitize distances to find which bin each pixel belongs to
        # indices will be 1 to num_bins
        bin_indices = np.digitize(pixel_dists, bins)
        
        bin_means = []
        
        for i in range(1, num_bins + 1):
            # Select intensities in the current bin
            in_bin = pixel_intensities[bin_indices == i]
            
            if in_bin.size > 0:
                bin_means.append(np.mean(in_bin))
            else:
                # If a bin is empty (rare with sufficient pixels), we can interpolate or ignore.
                # Ignoring is safer for statistics.
                pass
        
        if len(bin_means) < 2:
            # Not enough bins to calculate variation
            continue
            
        bin_means = np.array(bin_means)
        
        # Calculate Coefficient of Variation (CV) of the radial means
        # CV = std / mean
        mu = np.mean(bin_means)
        sigma = np.std(bin_means)
        
        if mu > 1e-6: # Avoid division by zero
            cv = sigma / mu
            cv_values.append(cv)
        else:
            cv_values.append(0.0)

    # 4. Aggregation
    if not cv_values:
        return 0.0
        
    result = np.mean(cv_values)
    
    return float(result)
