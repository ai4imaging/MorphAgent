def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import stats

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: BBBC021, (512, 512, 3), Channel 1 is Tubulin
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Extract Tubulin channel (Channel 1 - Green)
        tubulin_channel = arr[:, :, 1]
    elif arr.ndim == 2:
        # Fallback if passed a single channel image (though unlikely given description)
        tubulin_channel = arr
    else:
        # Unexpected format
        return 0.0

    # Feature Description:
    # "Calculates the kurtosis of the pixel intensity distribution in the Tubulin channel for the whole image."
    # High kurtosis indicates the presence of bright bundles (outliers) against a dark background.
    
    # Flatten the image to a 1D array of pixel intensities
    pixels = tubulin_channel.flatten()

    # Check for empty or zero-variance arrays to avoid division by zero
    if pixels.size == 0:
        return 0.0
    
    # Calculate standard deviation to check for constant images
    std_dev = np.std(pixels)
    if std_dev == 0:
        return 0.0

    # Calculate Fisher's Kurtosis (excess kurtosis)
    # Fisher's definition subtracts 3 from the result to make the normal distribution 0.0
    # This is the default behavior of scipy.stats.kurtosis
    kurtosis_val = stats.kurtosis(pixels, fisher=True, bias=False)

    # Handle potential NaN/Inf results
    if not np.isfinite(kurtosis_val):
        return 0.0

    return float(kurtosis_val)
