def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops
    from skimage.morphology import remove_small_objects
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3)
    # Channel 2 is DAPI (Nuclei)
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Extract DAPI channel (Blue, index 2)
        dapi_channel = arr[:, :, 2]
    elif arr.ndim == 2:
        # Fallback if passed a single channel image, assume it's the relevant one
        dapi_channel = arr
    else:
        return 0.0

    # Check for empty image or very low signal (background only)
    if np.max(dapi_channel) < 5:  # Arbitrary low threshold for empty image
        return 0.0

    # --- STRATEGY 1: Use Segmentation Masks if Available ---
    # The system might pass pre-computed masks. If so, use them for consistency.
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask = segmentation_masks[0]
        # Ensure mask is integer type
        mask = mask.astype(int)
        
        # If the mask is labeled (values > 1), count unique labels
        # If the mask is binary (0 and 1), we need to label it first
        unique_vals = np.unique(mask)
        
        if len(unique_vals) <= 1: # Only background
            return 0.0
            
        # Check if it's a binary mask (max value is 1) or instance mask (max value > 1)
        if np.max(mask) == 1:
            labeled_mask = label(mask)
            return float(np.max(labeled_mask))
        else:
            # It's likely an instance segmentation mask
            # Count unique non-zero labels
            # Subtract 1 for background (0) if present
            count = len(unique_vals) - (1 if 0 in unique_vals else 0)
            return float(count)

    # --- STRATEGY 2: Compute from Raw Image (Fallback) ---
    # If no mask is provided, we implement a robust nuclei counting pipeline.
    
    # 1. Normalization
    # Normalize to [0, 1] for consistent thresholding
    p_low, p_high = np.percentile(dapi_channel, (1, 99))
    if p_high - p_low > 0:
        norm_img = (dapi_channel - p_low) / (p_high - p_low)
    else:
        norm_img = dapi_channel
    norm_img = np.clip(norm_img, 0, 1)

    # 2. Smoothing
    # Gaussian blur to reduce noise and smooth texture within nuclei
    smoothed = ndimage.gaussian_filter(norm_img, sigma=2)

    # 3. Thresholding
    try:
        thresh = threshold_otsu(smoothed)
        binary = smoothed > thresh
    except Exception:
        # Fallback if otsu fails (e.g. uniform image)
        return 0.0

    # 4. Cleanup
    # Remove small artifacts (noise)
    # Area threshold: 50 pixels (approximate for 512x512 image of cells)
    cleaned_binary = remove_small_objects(binary, min_size=50)
    
    # Fill holes to ensure nuclei are solid
    cleaned_binary = ndimage.binary_fill_holes(cleaned_binary)

    # 5. Separation (Watershed)
    # MCF-7 cells often cluster. Simple labeling will undercount.
    # We use distance transform + watershed to separate touching nuclei.
    
    # Compute distance map
    distance = ndimage.distance_transform_edt(cleaned_binary)
    
    # Find peaks in the distance map (centers of nuclei)
    # min_distance ensures we don't over-segment a single nucleus
    local_maxi = peak_local_max(distance, min_distance=7, labels=cleaned_binary)
    
    # Create markers for watershed
    markers = np.zeros_like(cleaned_binary, dtype=int)
    markers[tuple(local_maxi.T)] = np.arange(len(local_maxi)) + 1
    
    # Dilate markers slightly to ensure they are not single pixels (helps watershed stability)
    markers = ndimage.maximum_filter(markers, size=3)

    # Apply watershed
    # We use -distance as the "elevation" map so peaks become basins
    labels = watershed(-distance, markers, mask=cleaned_binary)

    # 6. Final Count
    # The max label value corresponds to the number of detected objects
    cell_count = labels.max()

    return float(cell_count)
