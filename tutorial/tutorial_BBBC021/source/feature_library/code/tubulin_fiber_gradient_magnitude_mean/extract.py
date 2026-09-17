def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), uint8
    # Channel 1 is Tubulin (Green)
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Extract Tubulin channel (index 1)
        tubulin_channel = arr[..., 1]
    elif arr.ndim == 2:
        # Fallback if single channel passed (unlikely based on spec but safe)
        tubulin_channel = arr
    else:
        return 0.0

    # Intensity normalization
    # Normalize to [0, 1] range for consistent gradient magnitude calculation
    # The input is uint8 (0-255), so we divide by 255.0
    tubulin_normalized = tubulin_channel / 255.0
    
    # Determine the Region of Interest (ROI) / Mask
    # We need to calculate the mean gradient *within cells* to avoid the background
    # dragging down the mean (which would make it a confluence metric, not a texture metric).
    
    mask = None
    
    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0:
        # Iterate through masks to find a suitable one.
        # We prefer a mask that covers the cytoplasm/whole cell.
        # Often, if multiple masks are present, one might be nuclei and another cells.
        # Without specific metadata on which is which in the *args, we look for the largest coverage
        # assuming cell masks cover more area than nuclei masks.
        
        best_mask = None
        max_area = -1
        
        for m in segmentation_masks:
            if m is None:
                continue
            
            # Ensure mask is boolean or binary
            current_mask = m > 0
            current_area = np.sum(current_mask)
            
            if current_area > max_area:
                max_area = current_area
                best_mask = current_mask
        
        if best_mask is not None and max_area > 0:
            mask = best_mask

    # Fallback: If no valid mask provided, generate a foreground mask from the image itself
    if mask is None:
        # Use Otsu's thresholding to separate foreground (cells) from background
        # Check if image is not empty/constant
        if np.min(tubulin_normalized) == np.max(tubulin_normalized):
            return 0.0
            
        thresh = threshold_otsu(tubulin_normalized)
        mask = tubulin_normalized > thresh

    # Ensure mask is boolean and matches image shape
    if mask.shape != tubulin_normalized.shape:
        # If shapes mismatch (e.g. mask is 2D but image was somehow processed differently),
        # try to resize or just fallback to full image if impossible
        if mask.ndim == 2 and tubulin_normalized.ndim == 2:
             # Simple shape mismatch check
             if mask.shape != tubulin_normalized.shape:
                 # Resize not allowed by rules (no cv2.resize), so we fallback to threshold
                 try:
                    thresh = threshold_otsu(tubulin_normalized)
                    mask = tubulin_normalized > thresh
                 except:
                    mask = np.ones_like(tubulin_normalized, dtype=bool)
        else:
             mask = np.ones_like(tubulin_normalized, dtype=bool)

    # Compute Gradients
    # We use Sobel operator for edge detection (gradient approximation)
    # Sobel provides a bit of smoothing which is good for biological images
    sx = ndimage.sobel(tubulin_normalized, axis=0, mode='reflect')
    sy = ndimage.sobel(tubulin_normalized, axis=1, mode='reflect')
    
    # Compute Gradient Magnitude
    grad_mag = np.hypot(sx, sy)
    
    # Extract values within the mask
    # We only care about the texture *inside* the cells
    valid_pixels = grad_mag[mask]
    
    # Compute the mean
    if valid_pixels.size == 0:
        return 0.0
        
    result = np.mean(valid_pixels)

    return float(result)
