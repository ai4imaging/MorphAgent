def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import blob_log
    from skimage.filters import threshold_otsu
    from skimage.morphology import closing, square
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not the expected shape, return 0.0
        return 0.0

    # Extract Actin Channel (Channel 0)
    # Channel 0: Red (Actin/Cytoskeleton)
    actin_channel = arr[..., 0]

    # Normalize intensity to [0, 1]
    # The input is uint8 (0-255), so we divide by 255.0
    # This is crucial for the 'threshold' parameter in blob_log to be consistent
    actin_norm = actin_channel / 255.0
    
    # Clip to ensure range [0, 1]
    actin_norm = np.clip(actin_norm, 0.0, 1.0)

    # Define Region of Interest (ROI) Mask
    # We only want to count puncta inside cells, not background noise.
    roi_mask = None
    
    # Check if valid segmentation masks are provided
    valid_masks = [m for m in segmentation_masks if m is not None and isinstance(m, np.ndarray)]
    
    if len(valid_masks) > 0:
        # Combine all masks to create a comprehensive cellular mask
        # Masks typically have labels > 0 for cells
        combined_mask = np.zeros(actin_channel.shape, dtype=bool)
        for mask in valid_masks:
            # Handle potential dimension mismatch if mask is 3D (e.g. 1, H, W) vs 2D image
            if mask.ndim == 3:
                mask = np.max(mask, axis=0) # Project max if 3D
            if mask.shape == actin_channel.shape:
                combined_mask = np.logical_or(combined_mask, mask > 0)
        
        if np.any(combined_mask):
            roi_mask = combined_mask

    # Fallback: If no masks provided or mask generation failed, create one using Otsu
    if roi_mask is None:
        # Check if image is not empty/black
        if np.max(actin_norm) > 0:
            try:
                thresh = threshold_otsu(actin_norm)
                # Create binary mask
                binary = actin_norm > thresh
                # Close small holes to make the mask more solid
                roi_mask = closing(binary, square(3))
            except Exception:
                # Fallback if otsu fails (e.g. uniform image)
                roi_mask = np.ones(actin_channel.shape, dtype=bool)
        else:
            return 0.0

    # Detect Blobs using Laplacian of Gaussian (LoG)
    # Parameters tuned for small actin puncta (focal adhesions/aggregates)
    # min_sigma=1.0: Detects small spots (radius ~1.4px)
    # max_sigma=5.0: Avoids detecting large structures like nuclei
    # threshold=0.05: Relative intensity threshold (since image is 0-1)
    
    # blob_log returns array of [y, x, sigma]
    blobs = blob_log(actin_norm, min_sigma=1.0, max_sigma=5.0, num_sigma=10, threshold=0.05)

    if len(blobs) == 0:
        return 0.0

    # Filter blobs based on ROI mask
    count = 0
    H, W = actin_norm.shape
    
    for blob in blobs:
        y, x, sigma = blob
        y_int, x_int = int(y), int(x)
        
        # Ensure coordinates are within bounds
        if 0 <= y_int < H and 0 <= x_int < W:
            # Only count if the center of the blob is within the cellular ROI
            if roi_mask[y_int, x_int]:
                count += 1

    return float(count)
