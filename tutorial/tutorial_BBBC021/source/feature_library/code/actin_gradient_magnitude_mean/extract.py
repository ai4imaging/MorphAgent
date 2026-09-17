def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    
    # 1. Data Preprocessing and Validation
    # Convert to float32 for precision
    arr = np.asarray(img, dtype=np.float32)
    
    # Handle dimensionality
    # Expected shape is (512, 512, 3) for BBBC021
    # Channel 0 is Actin (Red)
    if arr.ndim == 3 and arr.shape[2] >= 1:
        # Extract Actin channel (Channel 0)
        actin_channel = arr[..., 0]
    elif arr.ndim == 2:
        # Fallback if single channel passed
        actin_channel = arr
    else:
        return 0.0

    # Normalize intensity to [0, 1] range for consistent gradient scale
    # This is crucial because gradient magnitude depends on the absolute intensity values
    vmax = 255.0  # Since input is uint8
    if actin_channel.max() > 0:
        actin_channel = actin_channel / vmax
    
    # 2. Compute Gradient Magnitude
    # We use Sobel operators to compute gradients in X and Y directions
    # Sobel provides some smoothing which helps reduce noise sensitivity compared to simple differences
    sx = ndimage.sobel(actin_channel, axis=0, mode='reflect')
    sy = ndimage.sobel(actin_channel, axis=1, mode='reflect')
    
    # Calculate Euclidean magnitude of the gradient vector at each pixel
    gradient_magnitude = np.hypot(sx, sy)
    
    # 3. Define Region of Interest (ROI)
    # We want to measure the gradient *within cells*, not in the background.
    # Background noise can dilute the mean gradient signal.
    
    mask = None
    
    # Strategy A: Use provided segmentation masks
    if len(segmentation_masks) > 0:
        # Combine all available masks to get a general "cellular foreground" mask
        combined_mask = np.zeros(actin_channel.shape, dtype=bool)
        for m in segmentation_masks:
            if m is not None:
                # Ensure mask matches image dimensions (handle potential 2D vs 3D mismatch)
                if m.shape == actin_channel.shape:
                    combined_mask = np.logical_or(combined_mask, m > 0)
                elif m.ndim == 3 and m.shape[:2] == actin_channel.shape:
                     # If mask is 3D (e.g. labeled), flatten it
                    combined_mask = np.logical_or(combined_mask, np.max(m, axis=2) > 0)
        
        if np.any(combined_mask):
            mask = combined_mask

    # Strategy B: Fallback to intensity-based thresholding if no valid mask found
    if mask is None:
        # Check if image has content
        if np.max(actin_channel) > np.min(actin_channel):
            try:
                thresh = threshold_otsu(actin_channel)
                mask = actin_channel > thresh
            except Exception:
                # Fallback for extremely low contrast or empty images
                mask = actin_channel > np.mean(actin_channel)
        else:
            # Image is constant
            return 0.0

    # 4. Compute Feature
    # Extract gradient values only from the cellular regions
    if np.sum(mask) == 0:
        return 0.0
        
    roi_gradients = gradient_magnitude[mask]
    
    # Calculate the mean
    result = np.mean(roi_gradients)
    
    return float(result)
