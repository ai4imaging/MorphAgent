def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), Channel 0 is Actin (Red)
    if arr.ndim != 3 or arr.shape[2] < 1:
        return 0.0

    # Extract Actin channel (Channel 0)
    actin_channel = arr[:, :, 0]

    # Intensity normalization
    # Data is uint8 (0-255). Normalize to [0, 1] for consistent feature calculation.
    # This makes the std dev comparable across images regardless of bit-depth scaling.
    actin_normalized = actin_channel / 255.0
    actin_normalized = np.clip(actin_normalized, 0.0, 1.0)

    # Handle segmentation masks (ROI definition)
    # We want to calculate std dev ONLY within the cells, not the background.
    mask = None

    if len(segmentation_masks) > 0:
        # If masks are provided, combine them to form a comprehensive foreground mask.
        # We assume any non-zero label in any mask indicates a cell region.
        combined_mask = np.zeros(actin_channel.shape, dtype=bool)
        for m in segmentation_masks:
            if m is not None and m.shape == actin_channel.shape:
                combined_mask = np.logical_or(combined_mask, m > 0)
        
        if np.any(combined_mask):
            mask = combined_mask

    # Fallback: If no valid masks provided, generate one using Otsu thresholding on the actin channel
    if mask is None:
        # Check if image is not empty/black
        if np.max(actin_normalized) > 0:
            try:
                thresh = threshold_otsu(actin_normalized)
                mask = actin_normalized > thresh
            except Exception:
                # Fallback for extremely low contrast or uniform images
                mask = actin_normalized > 0.05
        else:
            return 0.0

    # Feature Computation: Standard Deviation of Actin Intensity within Cells
    # Select pixels belonging to the mask
    foreground_pixels = actin_normalized[mask]

    # If no foreground pixels found, return 0.0
    if foreground_pixels.size == 0:
        return 0.0

    # Calculate standard deviation
    # High std dev -> high texture/contrast (e.g., stress fibers)
    # Low std dev -> diffuse signal
    actin_std = np.std(foreground_pixels)

    return float(actin_std)
