def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from scipy import stats
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type and normalize
    # Image shape is (512, 512, 3), dtype uint8
    arr = np.asarray(img, dtype=np.float32)
    
    # Check dimensionality
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract relevant channels
    # Channel 1: Tubulin (Green) - Target for intensity measurement
    # Channel 2: DAPI (Blue) - Target for finding centers (Nuclei)
    tubulin = arr[:, :, 1]
    dapi = arr[:, :, 2]

    # Normalize Tubulin channel to [0, 1] for consistent slope calculation
    # Robust normalization using percentiles to handle outliers
    p_min, p_max = np.percentile(tubulin, (1, 99))
    if p_max > p_min:
        tubulin = (tubulin - p_min) / (p_max - p_min)
    else:
        tubulin = tubulin - p_min # Should be 0
    tubulin = np.clip(tubulin, 0.0, 1.0)

    # Determine Nuclei/Cell Objects
    # Strategy: Use segmentation masks if available, otherwise fallback to DAPI thresholding
    
    labeled_nuclei = None
    labeled_cells = None
    
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Assuming the first mask is nuclei or cells. 
        # We need labeled objects.
        mask = segmentation_masks[0]
        if mask.ndim == 2:
            if mask.max() > 1: # Already labeled
                labeled_nuclei = mask
            else: # Binary mask
                labeled_nuclei = label(mask)
    
    # Fallback if no valid mask provided
    if labeled_nuclei is None:
        # Simple segmentation on DAPI
        try:
            thresh = threshold_otsu(dapi)
            binary_dapi = dapi > thresh
            # Remove small noise and label
            # Using binary opening to clean up
            binary_dapi = ndimage.binary_opening(binary_dapi, structure=np.ones((3,3)))
            labeled_nuclei = label(binary_dapi)
        except Exception:
            return 0.0

    # If we have a second mask (often cytoplasm/cell body), use it to constrain the radius
    if len(segmentation_masks) > 1 and segmentation_masks[1] is not None:
        mask2 = segmentation_masks[1]
        if mask2.ndim == 2:
             if mask2.max() > 1:
                 labeled_cells = mask2
             else:
                 labeled_cells = label(mask2)
    
    # If no cell mask, we will use a fixed radius or the nuclei labels expanded
    
    # Get properties of nuclei to find centers
    props = regionprops(labeled_nuclei)
    
    if len(props) == 0:
        return 0.0

    slopes = []
    
    # Parameters for radial profiling
    max_radius = 60  # Pixels. Reasonable for 512x512 image of cells.
    
    # Iterate over each cell/nucleus
    for prop in props:
        # Centroid coordinates
        cy, cx = prop.centroid
        
        # Define a bounding box for efficiency (clip to image boundaries)
        r_int = int(max_radius)
        y_min = max(0, int(cy) - r_int)
        y_max = min(tubulin.shape[0], int(cy) + r_int + 1)
        x_min = max(0, int(cx) - r_int)
        x_max = min(tubulin.shape[1], int(cx) + r_int + 1)
        
        # Extract local patch
        patch_tubulin = tubulin[y_min:y_max, x_min:x_max]
        
        # Create local coordinate grid relative to centroid
        y_indices, x_indices = np.indices(patch_tubulin.shape)
        # Adjust indices to be relative to the patch top-left
        # Centroid relative to patch:
        cy_local = cy - y_min
        cx_local = cx - x_min
        
        # Calculate distances
        distances = np.sqrt((y_indices - cy_local)**2 + (x_indices - cx_local)**2)
        
        # Create a mask for valid pixels
        # 1. Must be within max_radius
        valid_mask = distances <= max_radius
        
        # 2. If we have a cell mask, pixel must belong to the same cell ID (or background if we assume isolation)
        # However, matching nuclei labels to cell labels can be complex without a mapping.
        # Simplified approach: If cell mask exists, check if the centroid's cell label matches the pixel's cell label.
        if labeled_cells is not None:
            # Get cell label at centroid
            try:
                cell_id = labeled_cells[int(cy), int(cx)]
                if cell_id > 0:
                    patch_cells = labeled_cells[y_min:y_max, x_min:x_max]
                    valid_mask = valid_mask & (patch_cells == cell_id)
            except IndexError:
                pass
        
        # Flatten for binning
        valid_dists = distances[valid_mask]
        valid_intensities = patch_tubulin[valid_mask]
        
        if len(valid_dists) < 10:
            continue
            
        # Binning: Calculate mean intensity per integer radius
        # We use integer bins from 0 to max_radius
        bins = np.arange(0, max_radius + 1)
        
        # Use histogram to sum intensities and count pixels per bin
        counts, _ = np.histogram(valid_dists, bins=bins)
        sums, _ = np.histogram(valid_dists, bins=bins, weights=valid_intensities)
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_intensities = sums / counts
        
        # Filter out empty bins (NaNs)
        valid_bins_mask = counts > 0
        r_values = bins[:-1][valid_bins_mask]
        i_values = mean_intensities[valid_bins_mask]
        
        # We need at least a few points to fit a slope
        if len(r_values) < 5:
            continue
            
        # Calculate Slope (Linear Regression)
        # We expect Intensity to decrease with Radius, so slope should be negative.
        # We measure the "decay rate", so we look at the magnitude of the negative slope.
        # Alternatively, just return the raw slope.
        # A steeper negative slope (e.g. -0.05 vs -0.01) indicates more central concentration (Monoastral).
        
        # Simple linear fit: I = m*r + c
        slope, intercept, r_value, p_value, std_err = stats.linregress(r_values, i_values)
        
        # We are interested in the decay. 
        # If the slope is positive (intensity increases outwards), it's not a standard decay, 
        # but we still record it.
        slopes.append(slope)

    # Aggregation
    if not slopes:
        return 0.0
        
    # We return the median absolute decay. 
    # Since we defined the feature as "decay", a positive value usually implies the magnitude of the drop.
    # However, raw slope is more mathematically precise. 
    # If the feature is "decay rate", we usually invert the sign of the slope (so a drop is positive decay).
    # Let's return the median of the negative slopes (inverted), so higher value = faster decay.
    # We filter for negative slopes first to focus on the decay phenomenon? 
    # No, let's take the median of all slopes and invert it.
    # Normal phenotype: gradual decay (small negative slope).
    # Monoastral: sharp decay (large negative slope).
    # Result: -1 * slope.
    
    result = -1.0 * np.median(slopes)
    
    return float(result)
