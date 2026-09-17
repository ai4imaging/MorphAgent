def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import stats

    # 1. Input Validation and Format Handling
    # Ensure input is a numpy array
    img = np.asarray(img)
    
    # Check dimensionality. We expect (H, W, C) = (512, 512, 3) based on dataset info.
    # If the image is 2D (H, W), it might be a single channel projection or error.
    # If it is 3D (H, W, C), we need to select the correct channel.
    
    tubulin_pixels = None

    if img.ndim == 3 and img.shape[2] == 3:
        # Standard format: (512, 512, 3)
        # Channel 1 is Tubulin (Green) according to dataset description
        tubulin_channel = img[:, :, 1]
    elif img.ndim == 2:
        # Fallback: if single channel, assume it is the relevant one (though unlikely given description)
        tubulin_channel = img
    else:
        # Unexpected format
        return 0.0

    # Convert to float64 for statistical precision
    tubulin_channel = tubulin_channel.astype(np.float64)

    # 2. Region of Interest (ROI) Selection
    # If segmentation masks are provided, we should calculate kurtosis ONLY within the cellular regions.
    # Including the large black background (zeros) would artificially inflate kurtosis (heavy tail at 0).
    
    mask_combined = None
    
    if len(segmentation_masks) > 0:
        # Combine all available masks to get a "total foreground" mask
        # Masks are typically labeled integers (0=bg, 1..N=cells)
        for mask in segmentation_masks:
            if mask is not None:
                # Ensure mask shape matches image shape (handle potential 3D vs 2D mismatch if any)
                # Assuming masks are 2D (H, W) matching image spatial dims
                if mask.shape == tubulin_channel.shape:
                    binary_mask = mask > 0
                    if mask_combined is None:
                        mask_combined = binary_mask
                    else:
                        mask_combined = np.logical_or(mask_combined, binary_mask)
    
    # 3. Pixel Extraction
    if mask_combined is not None and np.any(mask_combined):
        # Extract pixels within the mask
        tubulin_pixels = tubulin_channel[mask_combined]
    else:
        # Fallback: Use the whole image if no masks are valid or provided
        # Note: This might result in higher kurtosis due to background, but is the standard "global" fallback.
        tubulin_pixels = tubulin_channel.flatten()

    # 4. Feature Computation: Kurtosis
    # Kurtosis requires variance. If variance is 0 (flat image), kurtosis is undefined/problematic.
    
    if tubulin_pixels.size < 4:
        # Not enough data points for stable 4th moment
        return 0.0

    # Check for zero variance (all pixels identical)
    if np.std(tubulin_pixels) == 0:
        return 0.0

    # Calculate Fisher's Kurtosis (normal distribution = 0.0)
    # bias=False corrects for statistical bias in sample kurtosis
    k_val = stats.kurtosis(tubulin_pixels, fisher=True, bias=False)

    # Handle potential NaNs from calculation
    if np.isnan(k_val) or np.isinf(k_val):
        return 0.0

    return float(k_val)
