def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import regionprops, label
    from skimage.filters import threshold_otsu

    # 1. Prepare Image Data
    # Convert to float32 for precision
    arr = np.asarray(img, dtype=np.float32)

    # Check dimensionality and extract channels
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] < 2:
        return 0.0
    
    # Channel 1: Tubulin (Green) - Target for intensity distribution
    tubulin_img = arr[..., 1]
    # Channel 2: DAPI (Blue) - Used for fallback center detection
    dapi_img = arr[..., 2]

    # Normalize Tubulin channel to [0, 1]
    # Robust normalization using percentiles to handle outliers
    p_min, p_max = np.percentile(tubulin_img, (1, 99))
    if p_max > p_min:
        tubulin_img = (tubulin_img - p_min) / (p_max - p_min)
    else:
        tubulin_img = tubulin_img - p_min
    tubulin_img = np.clip(tubulin_img, 0.0, 1.0)

    # 2. Handle Segmentation Masks
    cell_labels = None
    nuclei_labels = None

    # Heuristic to identify masks based on typical ordering or content
    # If masks are provided:
    if len(segmentation_masks) > 0:
        # Assume the first mask is cells if only one is provided, or if it's larger
        # Often in these datasets: mask 0 = cells/cytoplasm, mask 1 = nuclei
        # We will try to use the first mask as the primary ROI mask.
        mask_0 = segmentation_masks[0]
        if mask_0.shape == tubulin_img.shape:
            cell_labels = mask_0.astype(int)
        
        # If a second mask exists, it might be nuclei. We can use it for better centers.
        if len(segmentation_masks) > 1:
            mask_1 = segmentation_masks[1]
            if mask_1.shape == tubulin_img.shape:
                nuclei_labels = mask_1.astype(int)

    # Fallback: If no masks provided, generate them from DAPI
    if cell_labels is None:
        try:
            # Simple Otsu on DAPI to find nuclei
            thresh = threshold_otsu(dapi_img)
            nuclei_mask = dapi_img > thresh
            # Label nuclei
            nuclei_labels = label(nuclei_mask)
            
            # Expand nuclei to approximate cells (Voronoi-like or simple dilation)
            # Here we use a simple dilation for the ROI, or just use the nuclei as the ROI
            # For radial distribution, using the nucleus + surrounding area is better.
            # Let's dilate the nuclei mask to approximate cytoplasm
            cell_mask = ndimage.binary_dilation(nuclei_mask, iterations=15)
            cell_labels = label(cell_mask)
            
            # Re-assign nuclei labels to match cell labels if they merged/split differently
            # (Simplified: just use the dilated labels as the cell definition)
        except Exception:
            return 0.0

    # If we still have no valid labels, return 0
    if cell_labels is None or cell_labels.max() == 0:
        return 0.0

    # 3. Compute Feature: Radial Distribution Kurtosis
    
    # Get properties for cells
    cell_props = regionprops(cell_labels, intensity_image=tubulin_img)
    
    # If we have separate nuclei labels, get their centroids to use as "true centers"
    # Mapping cell_label_id -> (center_y, center_x)
    centers_map = {}
    if nuclei_labels is not None:
        nuclei_props = regionprops(nuclei_labels)
        # We need to match nuclei to cells. 
        # Simple approach: iterate nuclei, find which cell label is at the nucleus centroid
        for npop in nuclei_props:
            cy, cx = int(npop.centroid[0]), int(npop.centroid[1])
            # Ensure coordinates are within bounds
            if 0 <= cy < cell_labels.shape[0] and 0 <= cx < cell_labels.shape[1]:
                target_cell_id = cell_labels[cy, cx]
                if target_cell_id > 0:
                    centers_map[target_cell_id] = npop.centroid

    kurtosis_values = []

    for prop in cell_props:
        # Skip small artifacts
        if prop.area < 100:
            continue

        # Determine Center
        if prop.label in centers_map:
            center_y, center_x = centers_map[prop.label]
        else:
            # Fallback to cell centroid (weighted centroid is better if available, else geometric)
            center_y, center_x = prop.centroid

        # Get pixel coordinates of the cell
        # regionprops.coords returns (row, col)
        coords = prop.coords
        rows = coords[:, 0]
        cols = coords[:, 1]

        # Get intensities
        intensities = prop.image_intensity_original if hasattr(prop, 'image_intensity_original') else tubulin_img[rows, cols]
        
        # Ensure intensities are positive for weighting
        # Add a small epsilon to avoid division by zero if sum is 0
        weights = intensities + 1e-6
        sum_weights = np.sum(weights)

        if sum_weights == 0:
            continue

        # Calculate Euclidean distance from center for each pixel
        # r = sqrt((y - cy)^2 + (x - cx)^2)
        distances = np.sqrt((rows - center_y)**2 + (cols - center_x)**2)

        # Calculate Weighted Statistics for Radial Distribution
        
        # 1. Weighted Mean Radius
        mean_r = np.average(distances, weights=weights)

        # 2. Weighted Variance
        # variance = sum(w * (x - mean)^2) / sum(w)
        variance_r = np.average((distances - mean_r)**2, weights=weights)
        std_r = np.sqrt(variance_r)

        if std_r < 1e-6:
            # Variance is effectively zero (e.g., single pixel or perfect ring at exact distance)
            # Kurtosis is undefined or extreme. Skip.
            continue

        # 3. Weighted Fourth Moment
        # m4 = sum(w * (x - mean)^4) / sum(w)
        m4_r = np.average((distances - mean_r)**4, weights=weights)

        # 4. Fisher Kurtosis (Excess Kurtosis)
        # K = m4 / sigma^4 - 3
        kurtosis = (m4_r / (std_r**4)) - 3.0

        kurtosis_values.append(kurtosis)

    # 4. Aggregate Results
    if not kurtosis_values:
        return 0.0

    # Return the median kurtosis across all cells in the image
    # Median is robust to segmentation errors or debris
    result = np.median(kurtosis_values)

    # Handle NaN or Inf
    if not np.isfinite(result):
        return 0.0

    return float(result)
