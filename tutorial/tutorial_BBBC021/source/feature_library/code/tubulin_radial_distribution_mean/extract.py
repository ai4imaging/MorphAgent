def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import regionprops
    
    # 1. Input Validation and Preparation
    # Convert image to float32 for precision
    img_arr = np.asarray(img, dtype=np.float32)
    
    # Check dimensionality
    if img_arr.ndim != 3 or img_arr.shape[2] != 3:
        # If not standard 3-channel image, return NaN as we cannot reliably identify tubulin channel
        return float('nan')

    # Extract Tubulin Channel (Channel 1: Green)
    # Channel 0: Actin, Channel 1: Tubulin, Channel 2: DAPI
    tubulin_channel = img_arr[:, :, 1]
    
    # Handle Segmentation Masks
    # We need at least one mask to define cells. Ideally, we have two: nuclei and cells.
    # Standard convention often passes nuclei first, then cells, or vice versa.
    # We will try to identify them or use fallbacks.
    
    if not segmentation_masks:
        # Without segmentation, we cannot compute per-cell radial distribution relative to nucleus
        return float('nan')
    
    # Heuristic to assign masks:
    # If 2 masks: assume one is nuclei, one is cell. Usually nuclei are smaller.
    # If 1 mask: treat as cell mask, use cell centroid as proxy for nuclear centroid.
    
    nuclei_mask = None
    cell_mask = None
    
    masks = [np.asarray(m, dtype=np.int32) for m in segmentation_masks]
    
    if len(masks) >= 2:
        # Simple heuristic: Nuclei mask usually has smaller total area than cell mask
        area0 = np.count_nonzero(masks[0])
        area1 = np.count_nonzero(masks[1])
        if area0 < area1:
            nuclei_mask = masks[0]
            cell_mask = masks[1]
        else:
            nuclei_mask = masks[1]
            cell_mask = masks[0]
    elif len(masks) == 1:
        # Only one mask, treat as cell mask. Centroid will be cell centroid.
        cell_mask = masks[0]
        nuclei_mask = masks[0] # Use cell mask as proxy for nuclei location
    
    # Get unique cell IDs (excluding background 0)
    cell_ids = np.unique(cell_mask)
    cell_ids = cell_ids[cell_ids > 0]
    
    if len(cell_ids) == 0:
        return float('nan')

    # Pre-calculate coordinate grid for distance computation
    h, w = tubulin_channel.shape
    y_indices, x_indices = np.indices((h, w))
    
    # Calculate centroids for all nuclei at once for efficiency
    # ndimage.center_of_mass returns list of tuples [(y, x), ...]
    # The index argument allows us to get centroids for specific labels
    # Note: center_of_mass returns coordinates in order of index provided
    nuclei_centroids_list = ndimage.center_of_mass(np.ones_like(nuclei_mask), nuclei_mask, cell_ids)
    
    # Map cell_id -> (cy, cx)
    # Handle case where a label might be missing in nuclei_mask but present in cell_mask
    nuclei_centroids = {}
    for i, cid in enumerate(cell_ids):
        # center_of_mass returns tuple if found, or None/NaN if label missing
        # However, scipy returns a tuple of NaNs if label is missing in older versions, or raises error?
        # Actually, if label is missing, it might return NaN. Let's handle safely.
        centroid = nuclei_centroids_list[i]
        if centroid is not None and not np.isnan(centroid[0]):
            nuclei_centroids[cid] = centroid
        else:
            # Fallback: compute centroid of the cell mask itself for this ID
            # This is slow inside loop but handles mismatched masks
            slice_y, slice_x = ndimage.find_objects(cell_mask == cid)[0]
            roi = (cell_mask[slice_y, slice_x] == cid)
            cy, cx = ndimage.center_of_mass(roi)
            nuclei_centroids[cid] = (cy + slice_y.start, cx + slice_x.start)

    radial_means = []

    # Iterate over each cell to compute the feature
    # Optimization: Iterate over regionprops of cell_mask
    props = regionprops(cell_mask, intensity_image=tubulin_channel)
    
    for prop in props:
        cid = prop.label
        
        # Skip if we don't have a centroid for this cell
        if cid not in nuclei_centroids:
            continue
            
        cy, cx = nuclei_centroids[cid]
        
        # Get pixel coordinates within the cell bounding box
        # prop.coords returns (row, col) coordinates of pixels in the region
        coords = prop.coords
        pixel_y = coords[:, 0]
        pixel_x = coords[:, 1]
        
        # Get intensities
        # prop.image_intensity is the intensity image cropped to bbox
        # But prop.coords gives global coordinates. 
        # Easier to index directly into the global intensity image using coords
        intensities = tubulin_channel[pixel_y, pixel_x]
        
        # Filter out zero or negative intensities to avoid issues, though unlikely with uint8 input
        valid_mask = intensities > 0
        if not np.any(valid_mask):
            continue
            
        intensities = intensities[valid_mask]
        pixel_y = pixel_y[valid_mask]
        pixel_x = pixel_x[valid_mask]
        
        # Calculate Euclidean distance from nuclear centroid for each pixel
        distances = np.sqrt((pixel_y - cy)**2 + (pixel_x - cx)**2)
        
        # Calculate Intensity-Weighted Mean Distance
        # Formula: Sum(Distance * Intensity) / Sum(Intensity)
        weighted_sum = np.sum(distances * intensities)
        total_intensity = np.sum(intensities)
        
        if total_intensity > 0:
            mean_dist = weighted_sum / total_intensity
            radial_means.append(mean_dist)

    # Aggregate results
    if not radial_means:
        return float('nan')
        
    result = np.mean(radial_means)
    
    return float(result)
