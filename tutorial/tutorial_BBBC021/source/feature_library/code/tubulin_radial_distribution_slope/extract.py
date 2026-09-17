def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from scipy import stats
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    
    # Convert to appropriate array type and normalize
    # Image is (512, 512, 3), uint8
    arr = np.asarray(img, dtype=np.float32)
    
    # Check dimensions
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Normalize intensity to [0, 1]
    # We use a robust max to avoid hot pixels affecting the scale too much
    vmax = np.percentile(arr, 99.5) if arr.size > 0 else 1.0
    if vmax > 0:
        arr = arr / vmax
    arr = np.clip(arr, 0.0, 1.0)

    # Extract channels
    # Channel 1: Tubulin (Green) - The signal we measure
    # Channel 2: DAPI (Blue) - The reference for the center
    tubulin_ch = arr[..., 1]
    dapi_ch = arr[..., 2]
    
    # --- Segmentation Logic ---
    # We need labeled cells to compute this feature per cell.
    # Ideally, we want the whole cell mask to define the boundary and the nucleus to define the center.
    
    labeled_cells = None
    labeled_nuclei = None

    if len(segmentation_masks) > 0:
        # Heuristic: If masks are provided, assume the first one is cells or nuclei.
        # If multiple, usually the order is something like (nuclei, cells) or (cells, nuclei).
        # Without explicit metadata on mask order, we check sizes. Nuclei are usually smaller/contained.
        
        mask1 = segmentation_masks[0]
        if len(segmentation_masks) > 1:
            mask2 = segmentation_masks[1]
            # Simple heuristic: the mask with larger total area is likely the cell body
            if np.sum(mask1 > 0) > np.sum(mask2 > 0):
                labeled_cells = mask1
                labeled_nuclei = mask2
            else:
                labeled_cells = mask2
                labeled_nuclei = mask1
        else:
            # Only one mask. If it covers a large area, treat as cell.
            # If we only have one mask, we will use it as the cell boundary and 
            # derive the nucleus center from the DAPI intensity within that mask.
            labeled_cells = mask1
            labeled_nuclei = None # Will infer centroids from DAPI
            
    else:
        # Fallback: Generate segmentation on the fly
        # 1. Nuclei from DAPI
        try:
            thresh_nuc = threshold_otsu(dapi_ch)
            mask_nuc = dapi_ch > thresh_nuc
            # Clean up noise
            mask_nuc = ndimage.binary_opening(mask_nuc, structure=np.ones((3,3)))
            labeled_nuclei = label(mask_nuc)
            
            # 2. Cells from combined intensity (Tubulin + Actin usually defines cell shape better)
            # Channel 0 is Actin.
            actin_ch = arr[..., 0]
            combined = tubulin_ch + actin_ch
            thresh_cell = threshold_otsu(combined)
            mask_cell = combined > thresh_cell
            mask_cell = ndimage.binary_closing(mask_cell, structure=np.ones((3,3)))
            labeled_cells = label(mask_cell)
        except Exception:
            # If Otsu fails (e.g. empty image), return 0
            return 0.0

    # Ensure we have a valid cell mask
    if labeled_cells is None or labeled_cells.max() == 0:
        return 0.0

    # --- Feature Computation ---
    # Goal: Calculate slope of Tubulin intensity vs Normalized Radial Distance
    
    props_cells = regionprops(labeled_cells, intensity_image=tubulin_ch)
    
    slopes = []

    for prop in props_cells:
        # Skip small artifacts
        if prop.area < 100:
            continue

        # 1. Determine the Center (Origin)
        # If we have a nuclei mask, find the nucleus centroid inside this cell.
        # Otherwise, use the weighted centroid of the DAPI channel within the cell bounding box.
        
        minr, minc, maxr, maxc = prop.bbox
        
        # Extract DAPI crop for centroid calculation
        dapi_crop = dapi_ch[minr:maxr, minc:maxc]
        cell_mask_crop = prop.image  # Binary mask of the cell in the bounding box
        
        # Calculate center of mass of DAPI *inside* the cell mask
        # This is robust: it finds the brightest DAPI spot (nucleus) within the cell
        if np.sum(dapi_crop * cell_mask_crop) == 0:
            continue
            
        cy_local, cx_local = ndimage.center_of_mass(dapi_crop * cell_mask_crop)
        
        # 2. Calculate Radial Distances for all pixels in the cell
        # Create coordinate grids relative to the local centroid
        yy, xx = np.indices(cell_mask_crop.shape)
        distances = np.sqrt((yy - cy_local)**2 + (xx - cx_local)**2)
        
        # Filter: only consider pixels inside the cell mask
        valid_pixels = cell_mask_crop
        if not np.any(valid_pixels):
            continue
            
        pixel_dists = distances[valid_pixels]
        pixel_intensities = prop.intensity_image[valid_pixels] # Tubulin intensity
        
        # 3. Normalize Distances
        # We normalize by the maximum distance in this cell to make the slope comparable across cell sizes.
        # This maps the center to 0.0 and the furthest edge to 1.0.
        max_dist = np.max(pixel_dists)
        if max_dist == 0:
            continue
            
        norm_dists = pixel_dists / max_dist
        
        # 4. Binning (Optional but good for stability)
        # Instead of regressing on thousands of pixels, we bin them into radial shells.
        # This reduces noise and gives equal weight to the gradient trend.
        num_bins = 20
        bins = np.linspace(0, 1.0, num_bins + 1)
        
        # Compute mean intensity per bin
        bin_means = []
        bin_centers = []
        
        # Digitizing is faster than looping
        digitized = np.digitize(norm_dists, bins)
        
        for i in range(1, len(bins)):
            mask_bin = (digitized == i)
            if np.any(mask_bin):
                bin_mean = np.mean(pixel_intensities[mask_bin])
                bin_means.append(bin_mean)
                bin_centers.append((bins[i-1] + bins[i]) / 2)
        
        if len(bin_means) < 3:
            continue
            
        # 5. Linear Regression
        # We want the slope.
        # If tubulin is concentrated at the nucleus (collapsed), intensity drops as r increases -> Negative Slope.
        # If tubulin extends to edges, intensity is flatter -> Slope closer to 0.
        slope, intercept, r_value, p_value, std_err = stats.linregress(bin_centers, bin_means)
        
        if not np.isnan(slope):
            slopes.append(slope)

    # Aggregate results
    if not slopes:
        return 0.0
        
    # We return the median slope across the population.
    # A very negative median indicates collapsed microtubules (e.g. Nocodazole/Vinca alkaloids).
    # A less negative median indicates extended microtubules (Control/Taxol).
    return float(np.median(slopes))
