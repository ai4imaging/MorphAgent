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
        return 0.0
        
    # Extract Tubulin Channel (Channel 1 - Green)
    tubulin = arr[:, :, 1]
    
    # Normalize intensity to [0, 1]
    # Robust normalization using percentiles to handle outliers
    p_min, p_max = np.percentile(tubulin, (1, 99))
    if p_max > p_min:
        tubulin = (tubulin - p_min) / (p_max - p_min)
    else:
        tubulin = (tubulin - p_min) # Should result in zeros if range is 0
    tubulin = np.clip(tubulin, 0.0, 1.0)
    
    # --- 2. Define Region of Interest (ROI) ---
    
    # We need to measure lacunarity *inside* the cells.
    # If we include the black background between cells, lacunarity will be dominated
    # by cell confluence rather than intracellular texture.
    
    roi_mask = None
    
    if len(segmentation_masks) > 0:
        # Combine all available segmentation masks
        # Masks are typically labeled integers (0=bg, 1..N=cells)
        combined_mask = np.zeros(tubulin.shape, dtype=bool)
        for mask in segmentation_masks:
            if mask is not None and mask.shape == tubulin.shape:
                combined_mask = np.logical_or(combined_mask, mask > 0)
        
        if np.any(combined_mask):
            roi_mask = combined_mask

    # Fallback if no masks provided or masks are empty: Generate a tissue mask
    if roi_mask is None:
        # Use a smoothed version of the image to find general cell areas
        # We use the max projection of all channels to ensure we catch the whole cell area
        # (nucleus + tubulin + actin)
        composite_intensity = np.max(arr, axis=2)
        smoothed = ndimage.gaussian_filter(composite_intensity, sigma=5)
        try:
            thresh = threshold_otsu(smoothed)
            roi_mask = smoothed > thresh
        except Exception:
            # Fallback for completely flat images
            roi_mask = np.ones(tubulin.shape, dtype=bool)

    # If mask is still empty (e.g. blank image), return 0
    if not np.any(roi_mask):
        return 0.0

    # --- 3. Structure Extraction (Binarization) ---
    
    # Lacunarity is typically calculated on binary sets (the "mass").
    # We want to binarize the tubulin filaments.
    # Adaptive thresholding is better than global for preserving local filament structure
    # amidst varying cell brightness.
    
    # Block size for local thresholding (odd number)
    block_size = 35 
    try:
        local_thresh = threshold_local(tubulin, block_size, offset=0.05, method='gaussian')
        binary_tubulin = tubulin > local_thresh
    except Exception:
        # Fallback if image is too small or other error
        binary_tubulin = tubulin > 0.2

    # Apply ROI mask to the binary structure
    # Pixels outside the ROI should not contribute to "mass"
    binary_tubulin = np.logical_and(binary_tubulin, roi_mask)
    
    # Convert to float for convolution
    binary_tubulin_float = binary_tubulin.astype(np.float32)
    roi_mask_float = roi_mask.astype(np.float32)

    # --- 4. Gliding Box Lacunarity Calculation ---
    
    # Lacunarity L(r) = 1 + (variance(mass) / mean(mass)^2)
    # We calculate this for multiple box sizes r.
    
    box_sizes = [4, 8, 16, 32]
    lacunarity_values = []
    
    for r in box_sizes:
        # We use a uniform filter (convolution) to count pixels in a sliding window of size r
        # ndimage.uniform_filter computes the mean. To get the sum (mass), we multiply by size.
        # However, uniform_filter handles boundaries by padding/reflecting, which we don't want.
        # A cleaner way for "valid" boxes inside the ROI is manual convolution logic or careful masking.
        
        # Kernel for sum
        kernel_size = r
        kernel_area = r * r
        
        # Compute sum of tubulin pixels in every r x r box
        # uniform_filter calculates mean, so multiply by area to get sum
        # mode='constant', cval=0 ensures we don't pick up artifacts from outside
        local_mass_mean = ndimage.uniform_filter(binary_tubulin_float, size=kernel_size, mode='constant', cval=0.0)
        local_mass = local_mass_mean * kernel_area
        
        # Compute how much of the box is inside the ROI
        local_roi_mean = ndimage.uniform_filter(roi_mask_float, size=kernel_size, mode='constant', cval=0.0)
        local_roi_coverage = local_roi_mean * kernel_area
        
        # Define valid boxes:
        # 1. The box must be fully (or mostly) inside the ROI mask.
        #    Let's be strict: at least 95% of the box must be inside the cell mask.
        #    This prevents measuring the "gap" between the cell edge and the image border.
        valid_boxes_mask = local_roi_coverage >= (0.95 * kernel_area)
        
        if not np.any(valid_boxes_mask):
            continue
            
        # Extract masses for valid boxes
        valid_masses = local_mass[valid_boxes_mask]
        
        # Calculate statistics
        mean_mass = np.mean(valid_masses)
        var_mass = np.var(valid_masses)
        
        # Avoid division by zero
        if mean_mass == 0:
            # If mean is 0, it means all valid boxes are empty.
            # This implies maximum "gappiness" relative to the signal (or no signal).
            # We assign a high value or 0 depending on interpretation. 
            # Usually, if there is no structure, lacunarity is undefined or 1 (if var is also 0).
            # If var is 0 and mean is 0, L = 1 + 0/0 -> undefined.
            # Let's assume L=1 for completely empty regions (uniform emptiness).
            lacunarity = 1.0
        else:
            lacunarity = 1.0 + (var_mass / (mean_mass ** 2))
            
        lacunarity_values.append(lacunarity)
        
    # --- 5. Aggregation ---
    
    if not lacunarity_values:
        return 0.0
        
    # Return the mean lacunarity across the tested scales.
    # This provides a scalar summary of heterogeneity.
    result = np.mean(lacunarity_values)
    
    return float(result)
