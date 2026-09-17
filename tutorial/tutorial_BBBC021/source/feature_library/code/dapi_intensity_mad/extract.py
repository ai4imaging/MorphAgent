def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage

    # 1. Handle Image Format and Channel Extraction
    # The dataset description specifies (512, 512, 3) with DAPI in Channel 2 (Blue).
    # We need to be robust to potential variations, but prioritize the known structure.
    
    img_arr = np.asarray(img)
    
    # Check for empty image
    if img_arr.size == 0:
        return 0.0

    # Extract DAPI channel
    # Case 1: Standard (H, W, C) format
    if img_arr.ndim == 3 and img_arr.shape[2] == 3:
        dapi_channel = img_arr[:, :, 2]
    # Case 2: (C, H, W) format (less likely but possible in some loaders)
    elif img_arr.ndim == 3 and img_arr.shape[0] == 3:
        dapi_channel = img_arr[2, :, :]
    # Case 3: 2D image (assume it's already a single channel or grayscale projection)
    elif img_arr.ndim == 2:
        dapi_channel = img_arr
    # Fallback: Return 0 if dimensions are unexpected
    else:
        return 0.0

    # 2. Normalization
    # Convert to float32 and normalize to [0, 1] for consistent statistics
    # The dataset is uint8 (0-255)
    dapi_float = dapi_channel.astype(np.float32)
    
    # Robust normalization using max value in the type or data range
    # Using 255.0 is standard for uint8, but let's be safe with data max if it exceeds 255 (unlikely for uint8)
    max_val = 255.0
    if dapi_float.max() > 255.0:
        max_val = dapi_float.max()
    
    if max_val > 0:
        dapi_float = dapi_float / max_val
    
    # 3. Masking / Region of Interest Selection
    # We want to calculate MAD only on the nuclear pixels, not the background.
    # Including the large black background would skew the median towards 0 and the MAD towards 0.
    
    pixels_of_interest = None

    # Strategy A: Use provided segmentation masks
    if len(segmentation_masks) > 0:
        # Iterate through masks to find a suitable one. 
        # Usually, if multiple masks exist, one might be nuclei and another cells.
        # We prefer the one that covers less area (likely nuclei) for DAPI analysis, 
        # or simply combine them.
        
        # Let's try to use the first mask available, assuming it labels relevant objects.
        # If multiple masks are passed, we can check if any specifically aligns with nuclei logic,
        # but without specific metadata, using the union of all masks is a safe bet to capture foreground.
        
        combined_mask = np.zeros_like(dapi_channel, dtype=bool)
        
        for mask in segmentation_masks:
            if mask is not None and mask.shape == dapi_channel.shape:
                combined_mask = np.logical_or(combined_mask, mask > 0)
        
        if np.any(combined_mask):
            pixels_of_interest = dapi_float[combined_mask]

    # Strategy B: Fallback if no masks provided or masks were empty
    if pixels_of_interest is None or pixels_of_interest.size == 0:
        # Create a simple intensity-based mask to separate foreground (nuclei) from background.
        # Otsu's method is standard, but a simple percentile or mean threshold works for robust stats.
        # Here we use a low threshold to exclude pure background noise.
        
        # Estimate background level
        threshold = np.percentile(dapi_float, 50) # Median is often background in sparse images
        if threshold < 0.05: # If median is very dark (sparse cells), use a fixed small epsilon
             threshold = 0.05
        
        # Select pixels above threshold
        pixels_of_interest = dapi_float[dapi_float > threshold]

    # 4. Compute Median Absolute Deviation (MAD)
    # If we still have no pixels (e.g., completely black image), return 0.0
    if pixels_of_interest.size == 0:
        return 0.0

    # MAD = Median( | X - Median(X) | )
    # 1. Calculate Median of the intensity distribution
    median_intensity = np.median(pixels_of_interest)
    
    # 2. Calculate Absolute Deviations from the median
    absolute_deviations = np.abs(pixels_of_interest - median_intensity)
    
    # 3. Calculate the Median of those deviations
    mad_value = np.median(absolute_deviations)

    return float(mad_value)
