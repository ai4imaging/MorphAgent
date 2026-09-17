def extract(img, *segmentation_masks):
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu

    # 1. Input Validation and Preprocessing
    # Convert to float32 for processing to avoid overflow/underflow
    arr = np.asarray(img, dtype=np.float32)

    # Check for empty or invalid images
    if arr.size == 0 or np.max(arr) == 0:
        return 0.0

    # 2. Channel Selection
    # Dataset Description:
    # Channel 0: Actin (Cytoskeleton) - Primary indicator of cell shape/area
    # Channel 1: Tubulin (Microtubules) - Secondary indicator of cell body
    # Channel 2: DAPI (Nucleus) - Less relevant for total area occupancy
    
    # We combine Actin and Tubulin to get the full cellular footprint.
    # Using maximum projection ensures we capture the union of the signals.
    if arr.ndim == 3 and arr.shape[-1] >= 2:
        # Use Actin (0) and Tubulin (1)
        cytoskeleton = np.maximum(arr[..., 0], arr[..., 1])
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        # Fallback for single channel
        cytoskeleton = arr[..., 0]
    elif arr.ndim == 2:
        # Fallback for 2D image
        cytoskeleton = arr
    else:
        # Unexpected dimensionality
        return 0.0

    # 3. Signal Smoothing
    # Apply a Gaussian blur to smooth out noise and connect fragmented cytoskeletal structures
    # before thresholding. This helps create a more contiguous mask.
    # Sigma=2.0 is appropriate for 512x512 images to bridge small gaps.
    cytoskeleton_smooth = ndimage.gaussian_filter(cytoskeleton, sigma=2.0)

    # 4. Foreground Segmentation (Thresholding)
    # Use Otsu's method to automatically find the separation between background and cell signal.
    try:
        # Check if the image has enough variance for Otsu
        if np.min(cytoskeleton_smooth) == np.max(cytoskeleton_smooth):
            return 0.0
            
        thresh = threshold_otsu(cytoskeleton_smooth)
        binary_mask = cytoskeleton_smooth > thresh
    except Exception:
        # Fallback if thresholding fails (e.g., extremely uniform image)
        return 0.0

    # 5. Calculate Occupancy Fraction
    # Count foreground pixels
    occupied_pixels = np.sum(binary_mask)
    total_pixels = binary_mask.size

    if total_pixels == 0:
        return 0.0

    fraction = occupied_pixels / total_pixels

    return float(fraction)
