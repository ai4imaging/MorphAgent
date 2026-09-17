def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import disk, binary_opening
    from scipy.stats import linregress

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract Channels
    # Channel 1: Tubulin (Green) - Signal to measure
    # Channel 2: DAPI (Blue) - Reference for center
    tubulin = arr[..., 1]
    dapi = arr[..., 2]

    # Intensity normalization (0-1 range)
    # Normalize Tubulin
    vmax_tub = np.percentile(tubulin, 99.5) if tubulin.size > 0 else 1.0
    if vmax_tub > 0:
        tubulin = tubulin / vmax_tub
    tubulin = np.clip(tubulin, 0.0, 1.0)

    # Normalize DAPI for segmentation
    vmax_dapi = np.percentile(dapi, 99.5) if dapi.size > 0 else 1.0
    if vmax_dapi > 0:
        dapi = dapi / vmax_dapi
    dapi = np.clip(dapi, 0.0, 1.0)

    # --- Segmentation Logic ---
    # We need to identify individual cells to calculate radial gradients per cell.
    # Strategy:
    # 1. Detect Nuclei (Centers) using DAPI.
    # 2. Detect Cytoplasm (Boundaries) using Tubulin.
    # 3. Partition the image into cell territories (Voronoi-like based on distance to nuclei).

    # 1. Nuclei Segmentation
    try:
        thresh_dapi = threshold_otsu(dapi)
    except ValueError:  # Handle empty images
        return 0.0
    
    mask_nuclei = dapi > thresh_dapi
    mask_nuclei = binary_opening(mask_nuclei, footprint=disk(2))
    labeled_nuclei = label(mask_nuclei)
    nuclei_props = regionprops(labeled_nuclei)

    if not nuclei_props:
        return 0.0

    # 2. Cytoplasm Segmentation (Coarse mask for valid pixels)
    try:
        thresh_tub = threshold_otsu(tubulin)
    except ValueError:
        thresh_tub = 0.1
    mask_cyto = tubulin > thresh_tub

    # 3. Create Cell Partitions
    # We create a "labeled_cells" map where each pixel belongs to the nearest nucleus
    # This acts as a Voronoi partition restricted to the cytoplasm mask
    
    # Get centroids
    centroids = np.array([p.centroid for p in nuclei_props]) # (N, 2) array of (row, col)
    num_cells = len(centroids)

    # Create coordinate grids
    h, w = dapi.shape
    y_indices, x_indices = np.indices((h, w))
    
    # Flatten for vectorized distance calculation
    # Note: Doing full distance transform for many cells can be heavy.
    # Optimization: Use scipy.ndimage.distance_transform_edt with markers
    
    # Create marker image for watershed-like expansion
    markers = np.zeros_like(labeled_nuclei)
    for i, prop in enumerate(nuclei_props):
        # Mark the centroid pixel with the label ID
        rr, cc = int(prop.centroid[0]), int(prop.centroid[1])
        if 0 <= rr < h and 0 <= cc < w:
            markers[rr, cc] = prop.label

    # Distance transform to find nearest marker (Voronoi partition)
    # distance_transform_edt can return indices of the nearest feature
    # indices[0] is row index of nearest feature, indices[1] is col index
    # We use this to map every pixel to a nucleus ID
    if np.sum(markers) == 0:
        return 0.0
        
    # Calculate distance to nearest non-zero marker
    # We actually want the label of the nearest marker. 
    # ndimage.distance_transform_edt with return_indices=True gives the index of the nearest background point.
    # So we invert the logic: markers are background (0), we want distance to them? No.
    # Standard approach: use indices from EDT on inverted markers? No, markers are sparse points.
    
    # Alternative: Brute force distance for < 100 cells is fast enough on 512x512 usually, 
    # but let's use the indices method on the inverted marker map for speed.
    # Actually, simpler: use ndimage.distance_transform_edt on (markers==0) with return_indices=True.
    # The indices point to the nearest pixel where markers!=0.
    
    _, (idx_r, idx_c) = ndimage.distance_transform_edt(markers == 0, return_indices=True)
    cell_labels_map = markers[idx_r, idx_c]
    
    # Mask out background
    cell_labels_map[~mask_cyto] = 0

    # --- Feature Computation ---
    # For each cell, calculate the gradient of intensity vs distance from centroid
    
    slopes = []

    for i in range(1, num_cells + 1): # Labels are 1-based
        # Get mask for this specific cell
        cell_mask = (cell_labels_map == i)
        
        if np.sum(cell_mask) < 50: # Ignore tiny fragments
            continue

        # Get pixels for this cell
        cell_pixels_y = y_indices[cell_mask]
        cell_pixels_x = x_indices[cell_mask]
        intensities = tubulin[cell_mask]

        # Get centroid for this cell
        # (We rely on the order of regionprops matching the labels 1..N)
        # regionprops list is not guaranteed sorted by label, but usually is.
        # Safer to find the prop with the matching label.
        # Optimization: assume i-1 index if sorted, but let's be safe:
        centroid = None
        # Since we iterated 1..N, and regionprops usually returns sorted by label,
        # let's check:
        if i <= len(nuclei_props) and nuclei_props[i-1].label == i:
            centroid = nuclei_props[i-1].centroid
        else:
            # Fallback search
            for prop in nuclei_props:
                if prop.label == i:
                    centroid = prop.centroid
                    break
        
        if centroid is None:
            continue

        cy, cx = centroid

        # Calculate distances from centroid
        distances = np.sqrt((cell_pixels_y - cy)**2 + (cell_pixels_x - cx)**2)

        # We want to measure if intensity drops as distance increases.
        # Linear regression: Intensity = slope * distance + intercept
        # A Monoastral phenotype (Monastrol) has high intensity at center, low at edge.
        # This implies a NEGATIVE slope.
        
        # Handle constant intensity or single pixel cases
        if len(intensities) > 10 and np.std(distances) > 0:
            slope, intercept, r_value, p_value, std_err = linregress(distances, intensities)
            
            # If slope is NaN (e.g. constant input), treat as 0
            if np.isnan(slope):
                slope = 0.0
            
            slopes.append(slope)

    # --- Aggregation ---
    if not slopes:
        return 0.0

    # We return the negative mean slope.
    # Why?
    # Normal cells (meshwork): Slope is close to 0 or slightly negative.
    # Monoastral cells: Intensity drops sharply from center -> Slope is strongly negative (e.g., -0.05).
    # By returning -1 * slope, a "Monoastral" phenotype yields a higher positive value.
    # This aligns with the concept of "Radial Intensity Gradient Magnitude".
    
    avg_slope = np.mean(slopes)
    
    # To make the feature intuitive: 
    # Positive value = Intensity decreases radially (Center bright, edge dark)
    # Negative value = Intensity increases radially (Center dark, edge bright - e.g. cortical ring)
    result = -1.0 * avg_slope * 1000.0 # Scale up for readability (slopes are usually small per pixel)

    return float(result)
