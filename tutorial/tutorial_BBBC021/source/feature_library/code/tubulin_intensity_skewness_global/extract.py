def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import stats
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type
    # The input is expected to be uint8, convert to float for calculations
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset Description: (Height, Width, Channels) = (512, 512, 3)
    # Channel 1 is Tubulin (Green)
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Extract Tubulin channel (Index 1)
        tubulin_channel = arr[:, :, 1]
    elif arr.ndim == 2:
        # Fallback: if single channel, assume it is the relevant one or a projection
        tubulin_channel = arr
    else:
        # Unexpected format
        return 0.0

    # Define Region of Interest (ROI)
    # We want to calculate skewness on the cellular foreground, not the black background.
    # Including the massive peak at 0 (background) would artificially inflate skewness.
    
    roi_mask = None

    # 1. Try to use provided segmentation masks
    if len(segmentation_masks) > 0:
        # Combine all available masks to define the cellular area
        combined_mask = np.zeros(tubulin_channel.shape, dtype=bool)
        for mask in segmentation_masks:
            if mask is not None:
                # Ensure mask matches image dimensions (handle potential 2D vs 3D mismatch if any)
                if mask.shape == tubulin_channel.shape:
                    combined_mask = np.logical_or(combined_mask, mask > 0)
        
        if np.any(combined_mask):
            roi_mask = combined_mask

    # 2. Fallback: If no masks provided or masks were empty, generate a foreground mask
    if roi_mask is None:
        # Use Otsu's thresholding to separate foreground from background
        # Check if image has variance
        if np.min(tubulin_channel) == np.max(tubulin_channel):
            return 0.0 # Flat image, skewness undefined/zero
        
        thresh = threshold_otsu(tubulin_channel)
        roi_mask = tubulin_channel > thresh

    # Extract pixels within the ROI
    pixels = tubulin_channel[roi_mask]

    # Check for empty ROI or insufficient pixels
    if pixels.size < 3:
        return 0.0

    # Compute Skewness
    # Skewness is the third standardized moment.
    # High positive skewness indicates a distribution with a tail of high values 
    # (e.g., bright fibers on a darker background).
    # Low skewness indicates a more symmetric or diffuse distribution.
    
    # Check for zero variance to avoid division by zero in skewness calculation
    if np.std(pixels) == 0:
        return 0.0

    skewness_val = stats.skew(pixels, bias=False)

    # Handle NaN result (can happen if variance is extremely close to zero)
    if np.isnan(skewness_val):
        return 0.0

    return float(skewness_val)
