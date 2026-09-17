def extract(img, *segmentation_masks):
    import numpy as np
    from skimage.measure import label
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects
    from scipy import ndimage

    # Convert to appropriate array type
    arr = np.asarray(img)
    
    # Initialize count
    num_cells = 0
    
    # 1. Check for provided segmentation masks
    # If masks are provided by the system, use them as they are likely quality-controlled.
    # We iterate through all provided masks and pick the one with the most objects,
    # assuming it represents the cell/nuclei count.
    if len(segmentation_masks) > 0:
        max_objects = -1
        for mask in segmentation_masks:
            if mask is None:
                continue
            
            # Handle potential 3D masks (z-stacks) by projecting
            if mask.ndim == 3:
                curr_mask = np.max(mask, axis=0)
            else:
                curr_mask = mask
                
            # Count objects
            if np.issubdtype(curr_mask.dtype, np.integer):
                # Labeled mask: max label is usually the count
                count = curr_mask.max()
            else:
                # Binary mask: label connected components
                _, count = label(curr_mask > 0, return_num=True)
            
            if count > max_objects:
                max_objects = count
                num_cells = count

    # 2. Fallback: Compute from raw image if no valid masks found
    if num_cells == 0:
        # Validate dimensions for (H, W, C) = (512, 512, 3)
        # Channel 2 is DAPI (Nuclei)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            dapi_channel = arr[:, :, 2]
        elif arr.ndim == 2:
            dapi_channel = arr
        else:
            # Return 0 if dimensions are completely unexpected
            return 0.0

        # Normalize intensity to [0, 1]
        dapi_channel = dapi_channel.astype(np.float32)
        p99 = np.percentile(dapi_channel, 99.9)
        if p99 > 0:
            dapi_channel = dapi_channel / p99
        dapi_channel = np.clip(dapi_channel, 0, 1)

        # Preprocessing: Gaussian blur to reduce noise
        # Sigma=1.0 is a good balance for 512x512 images to keep nuclei distinct
        dapi_smooth = ndimage.gaussian_filter(dapi_channel, sigma=1.0)

        # Thresholding
        # Otsu is generally robust for DAPI. We wrap in try/except for empty images.
        try:
            thresh = threshold_otsu(dapi_smooth)
            binary_mask = dapi_smooth > thresh
        except Exception:
            binary_mask = np.zeros_like(dapi_smooth, dtype=bool)

        # Morphological cleanup
        # Remove small artifacts (min_size=20 pixels)
        # This filters noise without removing small nuclei (typical nucleus > 20px area)
        binary_mask = remove_small_objects(binary_mask, min_size=20)

        # Label connected components
        labeled_mask, num_cells = label(binary_mask, return_num=True)

    # 3. Compute Normalized Count
    # Area = Height * Width
    if arr.ndim >= 2:
        area = float(arr.shape[0] * arr.shape[1])
    else:
        area = 1.0

    if area <= 0:
        return 0.0

    return float(num_cells / area)
