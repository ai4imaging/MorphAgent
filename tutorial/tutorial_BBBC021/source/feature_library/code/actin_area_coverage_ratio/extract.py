def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.filters import threshold_otsu, gaussian
    
    # Convert to appropriate array type if needed, but keep original values for thresholding logic initially
    # The input is expected to be (512, 512, 3) uint8
    
    # Handle dimensionality and validate input
    if img is None:
        return 0.0
    
    # Check dimensions
    # Expected: (H, W, C) = (512, 512, 3)
    if img.ndim != 3:
        # If it's 2D (H, W), we can't reliably distinguish channels, return 0.0
        # If it's (C, H, W), we need to transpose, but the dataset says (H, W, C)
        if img.ndim == 2:
            # Fallback: treat the whole image as the signal if it's a single channel projection
            actin_channel = img
        else:
            return 0.0
    elif img.shape[-1] == 3:
        # Standard case: Channel 0 is Actin (Red)
        actin_channel = img[:, :, 0]
    else:
        # Unexpected channel count, return 0.0
        return 0.0

    # Convert to float for processing to avoid overflow/underflow
    actin_channel = actin_channel.astype(np.float32)
    
    # Preprocessing: Gaussian blur to reduce noise
    # This prevents single bright noise pixels from skewing the threshold or count
    # Sigma=1.0 is a gentle smoothing for 512x512 images
    actin_blurred = gaussian(actin_channel, sigma=1.0)

    # Check for empty or near-empty images to avoid Otsu errors
    # If the dynamic range is too low, Otsu's method is unstable/meaningless
    data_min = np.min(actin_blurred)
    data_max = np.max(actin_blurred)
    
    if data_max - data_min < 0.01: # E.g., a completely black image
        return 0.0

    # Determine Threshold
    # We use Otsu's method to find the optimal separation between background and foreground
    try:
        thresh = threshold_otsu(actin_blurred)
    except Exception:
        # Fallback if Otsu fails (e.g. uniform image)
        return 0.0

    # Calculate Coverage
    # Create binary mask where pixels exceed the threshold
    binary_mask = actin_blurred > thresh
    
    # Count foreground pixels
    foreground_pixels = np.sum(binary_mask)
    
    # Total pixels
    total_pixels = binary_mask.size
    
    if total_pixels == 0:
        return 0.0

    # Calculate ratio
    ratio = foreground_pixels / total_pixels

    return float(ratio)
