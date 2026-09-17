def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import stats

    # 1. Input Validation and Channel Selection
    # The dataset description specifies:
    # Channel 0: Actin (Red)
    # Channel 1: Tubulin (Green)
    # Channel 2: DAPI (Blue) - Nucleus
    # The feature request asks for DAPI intensity skewness.
    
    # Check if image is valid
    if img is None:
        return 0.0
        
    img_arr = np.asarray(img)
    
    # Handle dimensionality
    # Expected shape: (512, 512, 3)
    if img_arr.ndim == 3 and img_arr.shape[2] >= 3:
        # Select Channel 2 (DAPI) based on dataset specs
        dapi_channel = img_arr[:, :, 2]
    elif img_arr.ndim == 2:
        # Fallback for single channel images (assume it's the relevant one)
        dapi_channel = img_arr
    else:
        # Unexpected format
        return 0.0

    # 2. Pixel Selection Strategy
    # Skewness is highly sensitive to the massive peak of background pixels (usually 0 or near 0).
    # To get a biologically relevant measure of nuclear texture/condensation, we should
    # ideally calculate this only on nuclear pixels.

    pixels_to_analyze = None

    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first available mask (assuming it segments cells/nuclei)
        mask = np.asarray(segmentation_masks[0])
        
        # Ensure mask shape matches image shape (handling potential 2D vs 3D mismatch)
        if mask.shape == dapi_channel.shape:
            # Select pixels where mask indicates an object (label > 0)
            pixels_to_analyze = dapi_channel[mask > 0]
        else:
            # If shapes don't match, fallback to image-only processing
            pass

    # Fallback if no mask or mask processing failed
    if pixels_to_analyze is None:
        # Strategy: Exclude pure background (0) to avoid skewing the distribution 
        # with the large background peak.
        # This focuses the statistic on the actual signal.
        pixels_to_analyze = dapi_channel[dapi_channel > 0]

    # 3. Compute Skewness
    # Edge case: No pixels selected (e.g., completely black image)
    if pixels_to_analyze.size < 3:
        return 0.0

    # Convert to float for statistical precision
    pixel_values = pixels_to_analyze.astype(np.float64)

    # Calculate skewness
    # Skewness = 0 for normal distribution
    # Positive skewness = tail on the right (subset of very bright pixels)
    # Negative skewness = tail on the left
    # bias=False calculates the sample skewness (unbiased estimator)
    skewness_val = stats.skew(pixel_values, bias=False, nan_policy='omit')

    # 4. Sanitize Output
    if np.isnan(skewness_val) or np.isinf(skewness_val):
        return 0.0

    return float(skewness_val)
