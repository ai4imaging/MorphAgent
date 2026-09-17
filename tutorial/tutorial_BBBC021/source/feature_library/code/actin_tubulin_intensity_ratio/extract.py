def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    
    # 1. Data Validation and Preparation
    # Convert to float32 to prevent overflow during summation and allow for accurate division
    # The input is expected to be (H, W, C) = (512, 512, 3)
    arr = np.asarray(img, dtype=np.float32)
    
    # Check dimensions: Must be at least 3D with 3 channels for this specific feature
    if arr.ndim != 3 or arr.shape[2] < 2:
        return 0.0

    # 2. Channel Extraction
    # Based on dataset description:
    # Channel 0 = Actin (Red)
    # Channel 1 = Tubulin (Green)
    actin_channel = arr[:, :, 0]
    tubulin_channel = arr[:, :, 1]

    # 3. Mask Handling (Region of Interest)
    # If segmentation masks are provided, we use them to restrict the calculation to cellular regions.
    # This reduces the impact of background noise on the ratio.
    mask = None
    if len(segmentation_masks) > 0:
        # Combine all available masks (logical OR) to capture all cellular material
        # Masks are typically labeled integers (0=bg, 1..N=cells)
        combined_mask = np.zeros(actin_channel.shape, dtype=bool)
        for m in segmentation_masks:
            if m is not None and m.shape == actin_channel.shape:
                combined_mask = np.logical_or(combined_mask, m > 0)
        
        # If we successfully created a valid mask with at least one pixel
        if np.any(combined_mask):
            mask = combined_mask

    # 4. Intensity Computation
    if mask is not None:
        # Compute sum only within the mask
        total_actin = np.sum(actin_channel[mask])
        total_tubulin = np.sum(tubulin_channel[mask])
    else:
        # Compute global sum if no mask is available
        total_actin = np.sum(actin_channel)
        total_tubulin = np.sum(tubulin_channel)

    # 5. Ratio Calculation
    # Handle division by zero (e.g., if tubulin channel is empty or black)
    if total_tubulin <= 1e-6:
        # If there is actin but no tubulin, the ratio is effectively infinite.
        # However, for feature stability, we return 0.0 or a very large number?
        # Standard practice in this context: if image is empty (both 0), return 0.
        # If only denominator is 0, it's an edge case.
        if total_actin <= 1e-6:
            return 0.0
        else:
            # Return a high value to indicate extreme imbalance, but cap it to avoid Inf
            # Alternatively, just return 0.0 to be safe against artifacts.
            # Let's return 0.0 to be safe, assuming a bad image.
            return 0.0

    ratio = total_actin / total_tubulin

    return float(ratio)
