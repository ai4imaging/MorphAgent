def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract Channels
    # Channel 1: Tubulin (Green) - Target for intensity analysis
    # Channel 2: DAPI (Blue) - Target for finding centers (Nuclei)
    tubulin = arr[..., 1]
    dapi = arr[..., 2]

    # Intensity normalization for Tubulin
    # Normalize to [0, 1] to ensure consistent CV calculation scale
    # Using robust min/max to handle outliers
    p1, p99 = np.percentile(tubulin, [1, 99])
    if p99 > p1:
        tubulin = (tubulin - p1) / (p99 - p1)
        tubulin = np.clip(tubulin, 0.0, 1.0)
    else:
        if p99 > 0:
            tubulin = tubulin / p99
        tubulin = np.clip(tubulin, 0.0, 1.0)

    # ----------------------------------------------------------------
    # 1. Identify Nuclei (Centers)
    # ----------------------------------------------------------------
    # We always derive centers from the DAPI channel as it's the biological definition of the nucleus.
    # Even if masks are provided, re-computing centroids from DAPI ensures we have the centers 
    # corresponding to the intensity data.
    
    # Smooth DAPI to reduce noise
    dapi_smooth = ndimage.gaussian_filter(dapi, sigma=2.0)
    
    try:
        thresh_val = threshold_otsu(dapi_smooth)
        nuclei_mask = dapi_smooth > thresh_val
    except Exception:
        # Fallback if image is empty or constant
        return 0.0

    # Fill holes and label
    nuclei_mask = ndimage.binary_fill_holes(nuclei_mask)
    labeled_nuclei, num_nuclei = label(nuclei_mask, return_num=True)

    if num_nuclei == 0:
        return 0.0

    nuclei_props = regionprops(labeled_nuclei)

    # ----------------------------------------------------------------
    # 2. Define Foreground / Cell Mask
    # ----------------------------------------------------------------
    # We need a mask to define where the "cell" is, so we don't calculate CV on background noise.
    # If segmentation masks are provided, we use them as the valid area.
    # If not, we generate a fallback mask from the Tubulin channel.
    
    foreground_mask = None
    
    if len(segmentation_masks) > 0:
        # Combine all provided masks into a single binary valid mask
        # We assume any labeled region in any mask is valid cellular material
        combined_mask = np.zeros(dapi.shape, dtype=bool)
        for m in segmentation_masks:
            m_arr = np.asarray(m)
            # Handle potential extra dimensions in masks
            if m_arr.ndim == 3:
                m_arr = np.max(m_arr, axis=2) # Project if 3D/multichannel mask
            if m_arr.shape == dapi.shape:
                combined_mask = combined_mask | (m_arr > 0)
        
        if np.any(combined_mask):
            foreground_mask = combined_mask

    if foreground_mask is None:
        # Fallback: Threshold Tubulin to find general cell area
        try:
            t_thresh = threshold_otsu(tubulin)
            foreground_mask = tubulin > t_thresh
        except Exception:
            foreground_mask = np.ones_like(tubulin, dtype=bool)

    # ----------------------------------------------------------------
    # 3. Compute Radial Intensity CV
    # ----------------------------------------------------------------
    # Feature: tubulin_radial_intensity_cv
    # Logic: For each cell, iterate radial bins. Calculate CV (std/mean) in each bin.
    # Average across bins, then average across cells.
    
    cell_cv_scores = []
    
    # Parameters
    bin_width = 4.0  # Width of radial rings in pixels
    max_radius = 80.0 # Max distance to check (approx radius of MCF7 cell)
    min_pixels_per_bin = 10 # Minimum pixels to compute stats
    
    for prop in nuclei_props:
        # Skip very small objects (debris)
        if prop.area < 50:
            continue
            
        cy, cx = prop.centroid
        
        # Define a crop around the nucleus to speed up distance calculation
        # instead of computing distance map for full 512x512 image
        r_min = int(max(0, cy - max_radius))
        r_max = int(min(arr.shape[0], cy + max_radius + 1))
        c_min = int(max(0, cx - max_radius))
        c_max = int(min(arr.shape[1], cx + max_radius + 1))
        
        # Crop data
        crop_tubulin = tubulin[r_min:r_max, c_min:c_max]
        crop_mask = foreground_mask[r_min:r_max, c_min:c_max]
        
        if crop_tubulin.size == 0:
            continue

        # Create local coordinate grid
        y_indices, x_indices = np.indices(crop_tubulin.shape)
        
        # Adjust centroid to local crop coordinates
        cy_local = cy - r_min
        cx_local = cx - c_min
        
        # Compute Euclidean distance from centroid
        distances = np.sqrt((y_indices - cy_local)**2 + (x_indices - cx_local)**2)
        
        # Define bins
        bins = np.arange(0, max_radius, bin_width)
        
        bin_cvs = []
        
        for i in range(len(bins) - 1):
            r_inner = bins[i]
            r_outer = bins[i+1]
            
            # Select pixels in this ring that are also part of the foreground
            ring_mask = (distances >= r_inner) & (distances < r_outer) & crop_mask
            
            # Extract intensities
            vals = crop_tubulin[ring_mask]
            
            if len(vals) < min_pixels_per_bin:
                continue
            
            mu = np.mean(vals)
            sigma = np.std(vals)
            
            # Calculate Coefficient of Variation (CV)
            # Add epsilon to avoid division by zero
            if mu > 1e-6:
                cv = sigma / mu
                bin_cvs.append(cv)
        
        # If we computed valid CVs for this cell, add the average to the list
        if len(bin_cvs) > 0:
            cell_cv_scores.append(np.mean(bin_cvs))

    # ----------------------------------------------------------------
    # 4. Aggregate Results
    # ----------------------------------------------------------------
    if not cell_cv_scores:
        result = 0.0
    else:
        result = np.mean(cell_cv_scores)

    return float(result)
