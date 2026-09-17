def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu, threshold_local
    
    # --- 1. Data Loading and Preprocessing ---
    # Convert to float32 for processing
    arr = np.asarray(img, dtype=np.float32)
    
    # Check dimensionality
    # Expected: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # Fallback for potential 2D or other formats, though dataset spec says (H, W, 3)
        if arr.ndim == 2:
            # If 2D, assume it's a single channel image, treat as actin if we have no other info
            actin_channel = arr
        else:
            return 0.0
    else:
        # Extract Actin Channel (Channel 0 based on dataset description)
        actin_channel = arr[:, :, 0]

    # Normalize actin channel to 0-1
    vmax = np.percentile(actin_channel, 99.5) if actin_channel.size > 0 else 1.0
    if vmax <= 0: vmax = 1.0
    actin_norm = np.clip(actin_channel / vmax, 0.0, 1.0)

    # --- 2. Define Region of Interest (ROI) ---
    # We need a mask to define where the cells are.
    # Lacunarity is sensitive to the background (large empty spaces), so we want to calculate it
    # primarily within the cellular regions to measure cytoskeletal texture, not cell density.
    
    roi_mask = None
    
    # Try to use provided segmentation masks first
    if len(segmentation_masks) > 0:
        # Combine all masks to get a general "cellular area" mask
        combined_mask = np.zeros(actin_channel.shape, dtype=bool)
        for mask in segmentation_masks:
            if mask is not None and mask.shape == actin_channel.shape:
                combined_mask = np.logical_or(combined_mask, mask > 0)
        
        if np.any(combined_mask):
            roi_mask = combined_mask

    # Fallback: Generate mask from actin channel if no valid segmentation provided
    if roi_mask is None:
        # Simple background segmentation
        try:
            # Gaussian blur to smooth noise
            blurred = ndimage.gaussian_filter(actin_norm, sigma=2.0)
            thresh = threshold_otsu(blurred)
            roi_mask = blurred > thresh
        except Exception:
            # If thresholding fails (e.g., uniform image), use whole image
            roi_mask = np.ones(actin_channel.shape, dtype=bool)

    # If ROI is empty or too small, return 0
    if np.sum(roi_mask) < 100:
        return 0.0

    # --- 3. Binarize the Texture ---
    # Lacunarity is typically calculated on a binary set (foreground pixels vs gaps).
    # We want to binarize the actin filaments *within* the ROI.
    # Adaptive thresholding is good for capturing local texture (filaments) regardless of global intensity.
    
    # Apply local thresholding within the ROI
    # We use a block size relative to image size (e.g., 31 for 512x512)
    try:
        local_thresh = threshold_local(actin_norm, block_size=31, offset=0.0)
        binary_texture = actin_norm > local_thresh
    except:
        binary_texture = actin_norm > 0.5

    # Intersect texture with ROI (we only care about texture inside cells)
    # Pixels that are 1 are "mass" (filaments), 0 are "lacunae" (gaps)
    # Note: In standard lacunarity, the background outside the ROI shouldn't count as "gaps" 
    # in the same way. However, the standard gliding box algorithm runs over a rectangular domain.
    # To handle irregular ROIs, we mask the calculation or crop.
    # A robust approach for irregular shapes is to only consider boxes that are fully contained 
    # within the ROI, but that's computationally heavy.
    # A common approximation in bioimage analysis is to treat the ROI as the domain.
    # Here, we will simply zero out the background in the binary texture.
    binary_texture = np.logical_and(binary_texture, roi_mask)

    # --- 4. Gliding Box Lacunarity Calculation using Integral Image ---
    # Lacunarity L(r) = (sigma(r)^2 / mu(r)^2) + 1
    # where mu(r) and sigma(r)^2 are mean and variance of pixel sums in boxes of size r.
    
    # Compute Integral Image (Summed Area Table) for O(1) box sum calculation
    # Use float64 to prevent overflow
    int_img = np.cumsum(np.cumsum(binary_texture.astype(np.float64), axis=0), axis=1)
    
    def get_box_sum(r, integral_img):
        # Calculate sums for all r x r boxes using vectorization
        # S(x, y) = I(x, y) - I(x-r, y) - I(x, y-r) + I(x-r, y-r)
        # We slice the arrays to perform this operation for all boxes at once
        
        # The valid range for top-left corner (i, j) is 0 to H-r, 0 to W-r
        # I[r:, r:] corresponds to I(x, y)
        # I[:-r, r:] corresponds to I(x-r, y)
        # I[r:, :-r] corresponds to I(x, y-r)
        # I[:-r, :-r] corresponds to I(x-r, y-r)
        
        i_xy = integral_img[r:, r:]
        i_xr_y = integral_img[:-r, r:]
        i_x_yr = integral_img[r:, :-r]
        i_xr_yr = integral_img[:-r, :-r]
        
        box_sums = i_xy - i_xr_y - i_x_yr + i_xr_yr
        return box_sums

    # Also compute integral image of the ROI mask to count how many valid pixels are in each box
    # This allows us to filter out boxes that are mostly background
    int_mask = np.cumsum(np.cumsum(roi_mask.astype(np.float64), axis=0), axis=1)

    # Define box sizes (r)
    # Geometric progression is standard for fractal analysis
    box_sizes = [2, 4, 8, 16, 32, 64]
    lacunarity_values = []

    for r in box_sizes:
        if r >= min(actin_channel.shape):
            continue
            
        # Get sums of texture mass in boxes
        mass_counts = get_box_sum(r, int_img)
        
        # Get sums of valid ROI pixels in boxes
        valid_counts = get_box_sum(r, int_mask)
        
        # Filter boxes: We only want to calculate statistics on boxes that significantly overlap with the cell.
        # If a box is mostly in the background (outside ROI), it will have mass=0 and valid=0, 
        # which distorts the "gappiness" measure of the *cytoskeleton*.
        # Criterion: Box must be at least 50% inside the ROI
        valid_boxes_mask = valid_counts >= (0.5 * r * r)
        
        if np.sum(valid_boxes_mask) < 2:
            # Not enough valid boxes for this size
            continue
            
        relevant_masses = mass_counts[valid_boxes_mask]
        
        mean_mass = np.mean(relevant_masses)
        var_mass = np.var(relevant_masses)
        
        if mean_mass == 0:
            # If mean is 0, it means all boxes are empty. 
            # This implies extreme homogeneity (all empty), so Lacunarity -> 1 (conceptually) 
            # or undefined. We handle as 1.0 (minimum possible value).
            lacunarity = 1.0
        else:
            # L = (sigma^2 / mu^2) + 1
            # Coefficient of Variation squared + 1
            lacunarity = (var_mass / (mean_mass ** 2)) + 1
            
        lacunarity_values.append(lacunarity)

    # --- 5. Aggregation ---
    if not lacunarity_values:
        return 0.0
        
    # Return the mean lacunarity across the calculated scales
    result = np.mean(lacunarity_values)
    
    return float(result)
