def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type (float32 for precision)
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), uint8
    # Channel 0: Actin (Red)
    # Channel 1: Tubulin (Green) -> Target for intensity measurement
    # Channel 2: DAPI (Blue) -> Target for centroid definition
    
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract channels
    tubulin_ch = arr[..., 1]
    dapi_ch = arr[..., 2]

    # Handle segmentation masks
    # We need a nuclei mask to define centroids.
    # We ideally want a cell mask to define the boundary of the radial bands.
    
    nuclei_mask = None
    cell_mask = None

    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Assume first mask is nuclei or cells. 
        # Given the context of "radial bands from nuclear centroid", we treat the first available mask as the primary object definition (Nuclei).
        nuclei_mask = segmentation_masks[0]
        
        # If a second mask is available, it might be the cytoplasm/cell mask
        if len(segmentation_masks) > 1 and segmentation_masks[1] is not None:
            cell_mask = segmentation_masks[1]
    
    # Fallback if no masks provided
    if nuclei_mask is None:
        # Simple Otsu on DAPI
        try:
            thresh = threshold_otsu(dapi_ch)
            nuclei_mask = (dapi_ch > thresh).astype(int)
            nuclei_mask = label(nuclei_mask)
        except Exception:
            return 0.0

    # If cell mask is missing, we can approximate it or just use the nuclei mask 
    # However, for "radial bands extending from centroid", we usually want to measure into the cytoplasm.
    # If no cell mask, we will define a maximum radius or use a background threshold on Tubulin.
    # Let's create a simple background mask from Tubulin to avoid measuring empty space.
    if cell_mask is None:
        try:
            tub_thresh = np.percentile(tubulin_ch, 20) # Conservative background threshold
            cell_mask = (tubulin_ch > tub_thresh).astype(int)
            # We don't necessarily need instance labels for the cell mask if we just want to exclude background,
            # but to associate with specific nuclei, Voronoi would be better. 
            # For simplicity and robustness without heavy deps, we will just mask valid pixels.
        except:
            cell_mask = np.ones_like(tubulin_ch, dtype=int)

    # Get properties of nuclei to find centroids
    # Ensure nuclei_mask is labeled
    if nuclei_mask.max() == 0:
        return 0.0
        
    if nuclei_mask.ndim == 3: # Handle if mask was passed as 3D
        nuclei_mask = nuclei_mask.squeeze()
        
    labeled_nuclei = nuclei_mask if nuclei_mask.max() > 1 else label(nuclei_mask)
    props = regionprops(labeled_nuclei)

    # Parameters for radial analysis
    band_width = 4  # Width of each radial ring in pixels
    max_radius = 60 # Maximum radius to check (approx cell radius)
    epsilon = 1e-6

    cell_cv_scores = []

    for prop in props:
        # Centroid coordinates
        cy, cx = prop.centroid
        
        # Define a bounding box for analysis to speed up distance calculation
        # We look at a region around the centroid defined by max_radius
        r_int = int(max_radius)
        min_row, max_row = max(0, int(cy) - r_int), min(arr.shape[0], int(cy) + r_int)
        min_col, max_col = max(0, int(cx) - r_int), min(arr.shape[1], int(cx) + r_int)
        
        # Extract local patches
        local_tubulin = tubulin_ch[min_row:max_row, min_col:max_col]
        
        # If we have a specific cell mask (instance segmentation), we should use it to mask the local patch.
        # If cell_mask is just a binary foreground mask, we use that.
        # Here we check if cell_mask is labeled (instance) or binary.
        if cell_mask.ndim == 2:
            local_cell_mask = cell_mask[min_row:max_row, min_col:max_col]
            
            # If cell_mask is instance segmentation (matching nuclei labels), filter for current label
            # Note: This assumes label consistency which isn't guaranteed if masks come from different sources.
            # A safer heuristic: If cell_mask is labeled, pick the label at the centroid.
            if np.max(cell_mask) > 1:
                center_label = cell_mask[int(cy), int(cx)]
                if center_label > 0:
                    valid_pixels = (local_cell_mask == center_label)
                else:
                    # Fallback: just use non-zero pixels if centroid is background (misalignment)
                    valid_pixels = (local_cell_mask > 0)
            else:
                # Binary mask
                valid_pixels = (local_cell_mask > 0)
        else:
            valid_pixels = np.ones_like(local_tubulin, dtype=bool)

        # Create local coordinate grid for distance calculation
        h_local, w_local = local_tubulin.shape
        y_indices, x_indices = np.indices((h_local, w_local))
        
        # Adjust indices to be relative to the centroid
        # The patch starts at (min_row, min_col). Centroid is at (cy, cx).
        # Relative centroid in patch:
        rel_cy = cy - min_row
        rel_cx = cx - min_col
        
        distances = np.sqrt((y_indices - rel_cy)**2 + (x_indices - rel_cx)**2)
        
        # Calculate CV in bands
        band_cvs = []
        
        # Iterate through bands
        for r in range(0, max_radius, band_width):
            # Define band mask: within radius range AND within valid cell area
            mask_band = (distances >= r) & (distances < (r + band_width)) & valid_pixels
            
            # Extract intensities
            intensities = local_tubulin[mask_band]
            
            # We need enough pixels to calculate a meaningful variance
            if intensities.size > 5:
                mean_val = np.mean(intensities)
                std_val = np.std(intensities)
                
                # Coefficient of Variation
                if mean_val > epsilon:
                    cv = std_val / mean_val
                    band_cvs.append(cv)
        
        # If we computed CVs for this cell, average them to get the cell's score
        if len(band_cvs) > 0:
            cell_cv_scores.append(np.mean(band_cvs))

    # Aggregate across all cells in the image
    if len(cell_cv_scores) == 0:
        return 0.0
    
    result = np.mean(cell_cv_scores)

    return float(result)
