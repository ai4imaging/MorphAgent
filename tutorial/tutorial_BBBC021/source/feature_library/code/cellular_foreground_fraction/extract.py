def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu, threshold_triangle
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected: (512, 512, 3)
    if arr.ndim == 2:
        # If 2D (grayscale), treat as single channel intensity
        pass
    elif arr.ndim == 3:
        # If 3D (H, W, C) or (Z, H, W), we need to aggregate to get a 2D intensity map
        # Dataset is (512, 512, 3) where channels are Actin, Tubulin, DAPI.
        # To get the full cellular footprint, we should take the maximum intensity across channels.
        # This ensures we capture the cytoskeleton (Actin/Tubulin) which defines the cell boundary,
        # not just the nucleus.
        if arr.shape[-1] == 3: # (H, W, C)
            arr = np.max(arr, axis=-1)
        elif arr.shape[0] < 10: # Likely (C, H, W) or (Z, H, W)
            arr = np.max(arr, axis=0)
        else:
            # Fallback for unexpected 3D shape, assume last dim is channel
            arr = np.max(arr, axis=-1)
    else:
        return 0.0

    # Check for empty image
    if arr.size == 0 or np.max(arr) == 0:
        return 0.0

    # Intensity normalization
    # Normalize to [0, 1] for consistent thresholding behavior
    vmin, vmax = np.min(arr), np.max(arr)
    if vmax - vmin > 1e-6:
        arr = (arr - vmin) / (vmax - vmin)
    else:
        arr = np.zeros_like(arr)

    # --- Feature Calculation Strategy ---
    
    foreground_mask = None

    # Strategy 1: Use Segmentation Masks if available
    # The system passes masks as *segmentation_masks tuple
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first available mask. 
        # Usually, if multiple masks exist, they might be [cells, nuclei]. 
        # We prefer 'cells' if labeled as such, but here we take the first one provided.
        # If the mask is labeled (0=bg, 1,2...=cells), >0 gives the foreground.
        mask = np.asarray(segmentation_masks[0])
        
        # Ensure mask matches image spatial dimensions
        if mask.shape == arr.shape:
            foreground_mask = mask > 0
        elif mask.ndim == 3 and mask.shape[:2] == arr.shape:
             # If mask is 3D (e.g. one-hot or stacked), flatten it
             foreground_mask = np.max(mask, axis=-1) > 0
    
    # Strategy 2: Intensity-based Thresholding (Fallback)
    if foreground_mask is None:
        # 1. Smooth the image to reduce noise and close small gaps in cytoskeleton
        # Sigma=2.0 is generally good for 512x512 microscopy images to merge texture
        blurred = ndimage.gaussian_filter(arr, sigma=2.0)
        
        # 2. Determine Threshold
        # For fluorescence images with dark backgrounds, Triangle often works better than Otsu
        # if the histogram is unimodal with a long tail. However, Otsu is robust for bimodal.
        # We'll use a simple heuristic: if the image is very sparse, Triangle is safer.
        # Here, we default to Otsu as a standard for "foreground vs background".
        try:
            thresh = threshold_otsu(blurred)
            foreground_mask = blurred > thresh
        except Exception:
            # Fallback if Otsu fails (e.g., constant image)
            foreground_mask = np.zeros_like(blurred, dtype=bool)

    # --- Compute Fraction ---
    
    # Count foreground pixels
    foreground_pixels = np.sum(foreground_mask)
    total_pixels = foreground_mask.size
    
    if total_pixels == 0:
        return 0.0
        
    result = foreground_pixels / total_pixels

    return float(result)
