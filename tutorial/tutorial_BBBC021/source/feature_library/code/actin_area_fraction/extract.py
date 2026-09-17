def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.filters import threshold_otsu
    from scipy import ndimage

    # Convert to appropriate array type
    arr = np.asarray(img)

    # Handle dimensionality according to dataset format
    # Dataset Description: (512, 512, 3), Channel 0 = Actin (Red)
    if arr.ndim == 3 and arr.shape[2] >= 1:
        # Extract Channel 0 (Actin)
        actin_channel = arr[..., 0]
    elif arr.ndim == 2:
        # Fallback if a single channel 2D image is passed
        actin_channel = arr
    else:
        # Unexpected format
        return 0.0

    # Convert to float32 for processing to prevent overflow/underflow
    actin_float = actin_channel.astype(np.float32)

    # Safety check: Empty or extremely low signal images
    # If the dynamic range is negligible (e.g., < 5 intensity levels in uint8), 
    # Otsu will threshold on sensor noise. Return 0.0 in this case.
    if np.ptp(actin_float) < 5.0:
        return 0.0

    # Preprocessing: Gaussian Blur
    # Actin filaments are fine, high-frequency structures. To calculate a robust "area fraction"
    # that serves as a proxy for cell spreading/confluence, we apply a Gaussian blur.
    # This connects individual filaments into a cohesive "cytoplasmic footprint".
    # Sigma=2.0 is chosen based on the 512x512 resolution to bridge gaps between fibers.
    blurred = ndimage.gaussian_filter(actin_float, sigma=2.0)

    # Thresholding: Otsu's Method
    # We use Otsu's method to automatically determine the separation between 
    # background (substrate) and foreground (cells).
    try:
        thresh = threshold_otsu(blurred)
        
        # Create binary mask (True where signal > threshold)
        binary_mask = blurred > thresh
        
        # Calculate fraction: (Number of foreground pixels) / (Total pixels)
        # np.mean on a boolean array calculates this ratio efficiently.
        result = np.mean(binary_mask)
        
        return float(result)

    except Exception:
        # Fallback for cases where Otsu fails (e.g., perfectly uniform images)
        return 0.0
