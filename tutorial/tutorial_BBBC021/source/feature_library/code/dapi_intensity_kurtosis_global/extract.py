def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import stats
    
    # Convert to appropriate array type
    # The input is (512, 512, 3) uint8
    arr = np.asarray(img, dtype=np.float64)

    # Handle dimensionality according to dataset format
    # Dataset Description: (Height, Width, Channels) = (512, 512, 3)
    # Channel 2 is DAPI (Blue)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # Fallback for unexpected shapes, though dataset guarantees (512, 512, 3)
        return 0.0
        
    # Extract DAPI channel (Channel 2)
    dapi_channel = arr[:, :, 2]

    # Determine pixels of interest
    pixels = None
    
    # Check if segmentation masks are available
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first mask (typically nuclei or cells) to define the region of interest
        mask = segmentation_masks[0]
        
        # Ensure mask matches image spatial dimensions
        if mask.shape == dapi_channel.shape:
            # Select pixels belonging to foreground objects (label > 0)
            pixels = dapi_channel[mask > 0]
        else:
            # If shapes mismatch, fallback to full image (or simple thresholding)
            pixels = dapi_channel.flatten()
    else:
        # If no mask is provided, we analyze the global histogram.
        # However, including the vast background (zeros) will heavily skew kurtosis.
        # A simple threshold is better than including all background pixels for this specific biological feature
        # (condensed chromatin detection).
        # We filter out very low intensity background pixels to focus on the signal distribution.
        flat_dapi = dapi_channel.flatten()
        pixels = flat_dapi[flat_dapi > 5] # Basic background rejection

    # Edge case: No pixels selected (empty image or empty mask)
    if pixels is None or pixels.size < 4:
        return 0.0

    # Check for zero variance (uniform image), which causes division by zero in kurtosis calculation
    if np.std(pixels) == 0:
        return 0.0

    # Calculate Fisher's Kurtosis (Excess Kurtosis)
    # Fisher=True subtracts 3, so a normal distribution has kurtosis 0.0.
    # Bias=False uses the unbiased estimator for sample statistics.
    # High kurtosis indicates heavy tails (outliers), corresponding to very bright condensed chromatin.
    kurt_val = stats.kurtosis(pixels, fisher=True, bias=False)

    # Handle potential NaN results from scipy
    if np.isnan(kurt_val):
        return 0.0

    return float(kurt_val)
