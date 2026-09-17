def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    
    # Check for valid input
    if img is None:
        return 0.0

    # Convert to appropriate array type for calculation
    # The input is uint8 (0-255), we convert to float32 for precision
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected shape: (512, 512, 3)
    # Channel 0: Actin (Red)
    # Channel 1: Tubulin (Green)
    # Channel 2: DAPI (Blue)
    
    actin_channel = None

    if arr.ndim == 3:
        if arr.shape[2] >= 1:
            # Select Channel 0 (Actin)
            actin_channel = arr[:, :, 0]
        else:
            # Fallback if channels are missing but dim exists
            return 0.0
    elif arr.ndim == 2:
        # If image is unexpectedly 2D (grayscale), treat the whole image as the channel
        # This is a fallback for robustness
        actin_channel = arr
    else:
        # Unexpected dimensions
        return 0.0

    # Normalization
    # The dataset description states data is uint8.
    # We normalize to [0, 1] by dividing by 255.0.
    # This makes the feature comparable across different bit-depths if data changes.
    actin_channel = actin_channel / 255.0
    
    # Clip to ensure range [0, 1] in case of any preprocessing artifacts
    actin_channel = np.clip(actin_channel, 0.0, 1.0)

    # Computation: Global Mean Intensity
    # We calculate the mean over the entire image (FOV).
    # This includes background pixels, making it a proxy for confluence/total abundance.
    # We do NOT use segmentation masks here because we want the global average.
    result = np.mean(actin_channel)

    return float(result)
