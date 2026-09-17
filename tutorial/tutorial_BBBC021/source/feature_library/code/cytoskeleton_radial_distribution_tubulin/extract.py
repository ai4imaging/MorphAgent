def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type and handle dimensions
    # Dataset: (512, 512, 3), uint8. Channel 1 is Tubulin, Channel 2 is Nuclei.
    img_arr = np.asarray(img, dtype=np.float32)
    
    if img_arr.ndim != 3 or img_arr.shape[2] != 3:
        return 0.0

    # Extract relevant channels
    # Channel 1: Tubulin (Green) - The signal to measure
    # Channel 2: DAPI (Blue) - The reference for the center (nucleus)
    tubulin_ch = img_arr[:, :, 1]
    dapi_ch = img_arr[:, :, 2]

    # Normalize Tubulin channel for intensity weighting
    # Robust max to avoid hot pixels skewing normalization
    vmax = np.percentile(tubulin_ch, 99.5) if tubulin_ch.size > 0 else 1.0
    if vmax > 0:
        tubulin_ch = tubulin_ch / vmax
    tubulin_ch = np.clip(tubulin_ch, 0.0, 1.0)

    # Determine Masks (Nuclei and Cells)
    # Strategy: Use provided masks if available, otherwise generate basic ones
    nuclei_mask = None
    cell_mask = None

    if len(segmentation_masks) >= 2:
        # Assuming order might be cell, nuclei or vice versa. 
        # Usually, nuclei are smaller and contained within cells.
        # We will try to identify which is which based on containment or size, 
        # or strictly follow a convention if provided. 
        # Given the prompt doesn't specify strict order for *args, we'll assume:
        # If 2 masks: likely [cell_mask, nuclei_mask] or [nuclei, cell].
        # We can heuristically assign them or just use simple thresholding if ambiguous.
        # Let's stick to a robust fallback: Generate our own from the channels to be safe and consistent,
        # UNLESS the masks are clearly labeled. Since we can't see labels, 
        # we will use the image channels to generate fresh, consistent masks for this specific feature calculation.
        # This ensures the "center" (nucleus) matches the DAPI channel exactly.
        pass

    # Robust internal segmentation logic (fallback or primary)
    # 1. Segment Nuclei (from DAPI)
    try:
        thresh_nuc = threshold_otsu(dapi_ch)
        nuclei_mask = dapi_ch > thresh_nuc
        nuclei_labels = label(nuclei_mask)
    except Exception:
        return 0.0

    # 2. Segment Cells (from Tubulin, usually dimmer and larger)
    # If Tubulin is too faint, this might fail, but it's the best proxy for cell boundary.
    try:
        # Lower threshold for cell body
        thresh_cell = threshold_otsu(tubulin_ch) * 0.5 
        cell_mask = tubulin_ch > thresh_cell
        # Clean up: cell mask must contain nuclei
        cell_mask = np.logical_or(cell_mask, nuclei_mask)
        cell_labels = label(cell_mask)
    except Exception:
        return 0.0

    # Compute Radial Distribution per Cell
    # We need to link nuclei to cells.
    # Approach: Iterate over nuclei, find the enclosing cell, compute profile.
    
    nuclei_props = regionprops(nuclei_labels)
    
    radial_scores = []

    # Create coordinate grids once
    h, w = tubulin_ch.shape
    y_indices, x_indices = np.indices((h, w))

    for n_prop in nuclei_props:
        # Get centroid of the nucleus
        yc, xc = n_prop.centroid
        
        # Identify which cell this nucleus belongs to
        # Sample the cell_labels at the nucleus centroid
        cell_id = cell_labels[int(yc), int(xc)]
        
        if cell_id == 0:
            continue # Nucleus is in background of cell mask (shouldn't happen with logical_or)

        # Extract the specific cell mask
        # Optimization: Use bounding box of the cell to avoid processing full 512x512 arrays
        # We need to find the bbox of the cell_id
        # This can be slow if we do `cell_labels == cell_id` on the whole image every time.
        # Instead, we can approximate by using a large crop around the nucleus or just masking.
        # Given 512x512 is small, full mask operation is acceptable for speed vs complexity trade-off.
        
        current_cell_mask = (cell_labels == cell_id)
        
        # Get pixels belonging to this cell
        cell_pixels_y = y_indices[current_cell_mask]
        cell_pixels_x = x_indices[current_cell_mask]
        cell_intensities = tubulin_ch[current_cell_mask]
        
        if len(cell_intensities) == 0:
            continue

        # Calculate Euclidean distance from nucleus center for every pixel in the cell
        distances = np.sqrt((cell_pixels_x - xc)**2 + (cell_pixels_y - yc)**2)
        
        # Metric: Intensity-Weighted Mean Radius (normalized by max radius)
        # This describes "how far out" the mass of the tubulin is.
        # Low value -> Perinuclear accumulation
        # High value -> Spread to periphery
        
        total_intensity = np.sum(cell_intensities)
        if total_intensity == 0:
            continue
            
        weighted_mean_distance = np.sum(distances * cell_intensities) / total_intensity
        max_distance = np.max(distances)
        
        if max_distance == 0:
            continue
            
        normalized_score = weighted_mean_distance / max_distance
        radial_scores.append(normalized_score)

    # Aggregation
    # If no valid cells found, return 0.0
    if not radial_scores:
        return 0.0
        
    # Return the median score for the image to be robust against outliers
    result = np.median(radial_scores)

    return float(result)
