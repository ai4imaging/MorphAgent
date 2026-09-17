def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type (float32 for calculations)
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) -> (Height, Width, Channels)
    # Channel 0: Actin (Red)
    # Channel 1: Tubulin (Green)
    # Channel 2: DAPI (Blue)
    
    if arr.ndim != 3 or arr.shape[2] != 3:
        # Unexpected format, return 0.0
        return 0.0

    # Extract relevant channels
    actin_channel = arr[:, :, 0]
    dapi_channel = arr[:, :, 2]

    # Determine the mask to use for calculation
    # Strategy:
    # 1. If segmentation masks are provided, use the first one (assuming it labels cells/nuclei).
    #    Calculate correlation per object and average them.
    # 2. If no segmentation masks, generate a global foreground mask to avoid background bias.
    #    Calculate correlation over all foreground pixels.

    correlations = []
    
    has_segmentation = False
    label_mask = None

    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use provided segmentation
        candidate_mask = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if candidate_mask.shape == arr.shape[:2]:
            label_mask = candidate_mask
            has_segmentation = True

    if has_segmentation:
        # Per-object calculation
        # Get unique labels (excluding background 0)
        unique_labels = np.unique(label_mask)
        unique_labels = unique_labels[unique_labels > 0]

        if len(unique_labels) == 0:
            return 0.0

        for label_id in unique_labels:
            # Extract pixels for this object
            mask_bool = (label_mask == label_id)
            
            # Optimization: Get indices to avoid full array boolean indexing if sparse
            # But for 512x512, boolean masking is fast enough
            
            actin_vals = actin_channel[mask_bool]
            dapi_vals = dapi_channel[mask_bool]

            # Need at least 2 pixels to calculate correlation
            if len(actin_vals) < 2:
                continue

            # Check for variance to avoid division by zero
            actin_std = np.std(actin_vals)
            dapi_std = np.std(dapi_vals)

            if actin_std > 1e-6 and dapi_std > 1e-6:
                # Calculate Pearson correlation
                # np.corrcoef returns correlation matrix [[1, r], [r, 1]]
                r = np.corrcoef(actin_vals, dapi_vals)[0, 1]
                if not np.isnan(r):
                    correlations.append(r)
    
    else:
        # Fallback: Global foreground calculation
        # Create a foreground mask based on intensity to exclude empty background
        # Combine channels for thresholding
        combined_intensity = actin_channel + dapi_channel
        
        # Check if image is essentially empty
        if np.max(combined_intensity) < 1e-6:
            return 0.0
            
        try:
            thresh = threshold_otsu(combined_intensity)
            fg_mask = combined_intensity > thresh
        except Exception:
            # Fallback for extremely low contrast images
            fg_mask = combined_intensity > np.mean(combined_intensity)

        actin_vals = actin_channel[fg_mask]
        dapi_vals = dapi_channel[fg_mask]

        if len(actin_vals) < 2:
            return 0.0

        actin_std = np.std(actin_vals)
        dapi_std = np.std(dapi_vals)

        if actin_std > 1e-6 and dapi_std > 1e-6:
            r = np.corrcoef(actin_vals, dapi_vals)[0, 1]
            if not np.isnan(r):
                correlations.append(r)

    # Aggregate results
    if len(correlations) == 0:
        return 0.0
    
    result = np.mean(correlations)
    
    # Ensure result is within [-1, 1] (numerical errors might push it slightly outside)
    result = np.clip(result, -1.0, 1.0)

    return float(result)
