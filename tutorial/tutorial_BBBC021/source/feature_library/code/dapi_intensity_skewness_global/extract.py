def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy.stats import skew

    # Convert to appropriate array type (float64 for statistical precision)
    arr = np.asarray(img, dtype=np.float64)

    # Handle dimensionality according to dataset format
    # Dataset Description: (Height, Width, Channels) = (512, 512, 3)
    # Channel Mapping: 0=Actin, 1=Tubulin, 2=DAPI
    # We specifically target the DAPI channel (index 2) for this feature.
    if arr.ndim == 3 and arr.shape[2] >= 3:
        dapi_channel = arr[:, :, 2]
    elif arr.ndim == 2:
        # Fallback: if image is 2D, assume it is the relevant channel or a projection
        dapi_channel = arr
    else:
        # Return 0.0 for unexpected formats (e.g., 1D or 4D)
        return 0.0

    # Feature: dapi_intensity_skewness_global
    # Logic: Calculate the skewness of the pixel intensity distribution.
    # Interpretation: 
    #   - High positive skewness (>0): Indicates a distribution with a long tail towards high values.
    #     In fluorescence microscopy, this corresponds to sparse, bright signals (nuclei) on a 
    #     predominantly dark background.
    #   - Lower skewness: Indicates a more symmetric distribution, suggesting either high background 
    #     noise (raising the floor) or nuclear swelling/diffusion (spreading the signal), reducing 
    #     the contrast between the "peaks" and the "floor".

    # Flatten the 2D channel to a 1D array of pixel intensities
    pixels = dapi_channel.flatten()

    # Check for zero variance (uniform image)
    # Skewness is undefined if standard deviation is 0 (division by zero in the formula)
    if np.std(pixels) < 1e-9:
        return 0.0

    # Calculate skewness
    # bias=False calculates the unbiased estimator (Fisher-Pearson coefficient of skewness)
    result = skew(pixels, bias=False)

    # Safety check for NaN/Inf results
    if not np.isfinite(result):
        return 0.0

    return float(result)
