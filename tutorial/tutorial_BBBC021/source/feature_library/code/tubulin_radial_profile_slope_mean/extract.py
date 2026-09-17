def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage, stats
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract channels
    # Channel 1: Tubulin (Green) - Target for intensity profile
    # Channel 2: DAPI (Blue) - Reference for centroids
    tubulin_ch = arr[..., 1]
    dapi_ch = arr[..., 2]

    # Intensity normalization (per channel)
    def normalize_channel(ch):
        vmax = np.percentile(ch, 99.5) if ch.size > 0 else 1.0
        if vmax > 0:
            return np.clip(ch / vmax, 0.0, 1.0)
        return np.zeros_like(ch)

    tubulin_norm = normalize_channel(tubulin_ch)
    dapi_norm = normalize_channel(dapi_ch)

    # Handle segmentation masks
    nuclei_mask = None
    cell_mask = None

    # Check if masks are provided via *segmentation_masks
    # Based on typical biological segmentation outputs, if masks are provided:
    # usually mask 0 is cells/cytoplasm, mask 1 is nuclei, or vice versa.
    # However, without strict guarantees, we often rely on computing them if not explicitly clear.
    # Given the prompt says "masks are automatically loaded", let's try to use them if valid.
    
    if len(segmentation_masks) > 0:
        # Heuristic: usually the smaller objects are nuclei.
        # But to be safe and robust given the prompt instructions to handle "optional" masks,
        # we will prioritize self-computed masks from the DAPI channel for the centroids 
        # to ensure the "center" matches the DAPI signal, unless the provided mask is clearly labeled.
        # For this specific feature, calculating centroids from the DAPI channel is the most robust method
        # to ensure we are measuring "radial from nucleus".
        pass

    # 1. Generate Nuclei Mask (for centroids)
    # Use DAPI channel
    try:
        thresh_dapi = threshold_otsu(dapi_norm)
        nuclei_binary = dapi_norm > thresh_dapi
        # Remove small noise
        nuclei_binary = ndimage.binary_opening(nuclei_binary, structure=np.ones((3,3)))
        nuclei_labels = label(nuclei_binary)
    except Exception:
        return 0.0

    # 2. Generate Foreground Mask (to limit radial profile to cell area)
    # Use Tubulin channel
    try:
        thresh_tubulin = threshold_otsu(tubulin_norm)
        foreground_mask = tubulin_norm > thresh_tubulin
    except Exception:
        # Fallback if thresholding fails (e.g. empty image)
        foreground_mask = np.ones_like(tubulin_norm, dtype=bool)

    # Get properties of nuclei
    props = regionprops(nuclei_labels)
    
    if not props:
        return 0.0

    slopes = []
    
    # Parameters for radial profile
    max_radius = 60  # Pixels. Reasonable for 512x512 image of cells.
    
    # Create a grid of coordinates once
    y_indices, x_indices = np.indices(tubulin_norm.shape)

    for prop in props:
        # Centroid
        cy, cx = prop.centroid
        
        # Define a Region of Interest (ROI) around the nucleus to speed up distance calculation
        # We only care about pixels within max_radius
        r_int = int(max_radius)
        y_min = max(0, int(cy) - r_int)
        y_max = min(tubulin_norm.shape[0], int(cy) + r_int + 1)
        x_min = max(0, int(cx) - r_int)
        x_max = min(tubulin_norm.shape[1], int(cx) + r_int + 1)
        
        # Extract ROI crops
        roi_tubulin = tubulin_norm[y_min:y_max, x_min:x_max]
        roi_foreground = foreground_mask[y_min:y_max, x_min:x_max]
        
        # Local coordinates
        roi_y = y_indices[y_min:y_max, x_min:x_max]
        roi_x = x_indices[y_min:y_max, x_min:x_max]
        
        # Calculate distance from centroid for every pixel in ROI
        distances = np.sqrt((roi_y - cy)**2 + (roi_x - cx)**2)
        
        # Flatten for processing
        flat_dist = distances.ravel()
        flat_int = roi_tubulin.ravel()
        flat_mask = roi_foreground.ravel()
        
        # Filter: 
        # 1. Must be within max_radius
        # 2. Must be in foreground (part of a cell, not black background)
        valid_indices = (flat_dist <= max_radius) & (flat_mask)
        
        if np.sum(valid_indices) < 10:
            continue
            
        valid_dist = flat_dist[valid_indices]
        valid_int = flat_int[valid_indices]
        
        # Bin the distances to compute mean intensity profile
        # We use integer bins: 0, 1, 2, ... max_radius
        bins = np.arange(0, max_radius + 1)
        
        # Compute mean intensity per radial bin
        # digitize returns indices 1..len(bins)
        bin_indices = np.digitize(valid_dist, bins)
        
        radial_means = []
        radial_r = []
        
        for i in range(1, len(bins)):
            mask_bin = (bin_indices == i)
            if np.any(mask_bin):
                mean_val = np.mean(valid_int[mask_bin])
                radial_means.append(mean_val)
                radial_r.append(bins[i-1]) # Use the bin edge as r value
        
        if len(radial_means) < 5:
            # Not enough points to fit a slope reliably
            continue
            
        # Linear regression: Intensity = slope * radius + intercept
        # We expect slope to be negative (intensity drops as we go out)
        # or near zero (diffuse).
        slope, intercept, r_value, p_value, std_err = stats.linregress(radial_r, radial_means)
        
        if not np.isnan(slope):
            slopes.append(slope)

    if not slopes:
        return 0.0

    # Return the mean slope across all valid cells
    result = np.mean(slopes)
    
    return float(result)
